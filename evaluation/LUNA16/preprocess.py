import os
from glob import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

import torch
import torch.nn.functional as F

for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")


dataset_root = "/scratch/VM/radio-foundation/essential/LUNA16"
output_root = "/scratch/VM/radio-foundation/preprocessed/LUNA16"
os.makedirs(output_root, exist_ok=True)

target_size = 224
min_crop_mm, max_crop_mm = 40, 240
num_augmentations = 6
clearance = 20
max_attempts = 100

SEED = 4
np.random.seed(SEED)
random.seed(SEED)

device_idx = 1
device = torch.device(f"cuda:{device_idx}")
print(f"Using Device {device_idx}: {torch.cuda.get_device_name(device_idx)}")


img_paths = glob(os.path.join(dataset_root, "subset*/**/*.mhd"), recursive=True)

id_to_path = {p.split("/")[-1].replace(".mhd", ""): p for p in img_paths}
print(f"Found {len(id_to_path)} scans.")

annotations_path = os.path.join(dataset_root, "annotations.csv")

annotations_df = pd.read_csv(annotations_path)
print(f"Loaded {len(annotations_df)} total annotations.")


def world_to_voxel(image_sitk, world_coord):
    """
    Convert world (x,y,z) -> voxel (z,y,x) continuous (float) coordinates.

    This function is robust and uses SimpleITK's trusted method.
    It returns a (z, y, x) ordered NumPy array.
    """
    continuous_index_xyz = image_sitk.TransformPhysicalPointToContinuousIndex(
        world_coord
    )

    return np.array(continuous_index_xyz)[::-1]


def resample_to_isotropic(image_sitk, new_spacing=[1.0, 1.0, 1.0]):
    """
    Resample a SimpleITK image to isotropic spacing using PyTorch (on CUDA).
    """
    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()

    original_physical_size = [
        sz * sp for sz, sp in zip(original_size, original_spacing)
    ]
    new_size_xyz = [
        int(round(phys_sz / sp))
        for phys_sz, sp in zip(original_physical_size, new_spacing)
    ]

    target_size_zyx = new_size_xyz[::-1]

    volume_np = sitk.GetArrayFromImage(image_sitk)
    volume_tensor = torch.from_numpy(volume_np).float().to(device)

    volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)

    resampled_tensor = F.interpolate(
        input=volume_tensor, size=target_size_zyx, mode="trilinear", align_corners=True
    )

    resampled_volume_np = resampled_tensor.squeeze().cpu().numpy()

    iso_image_sitk = sitk.GetImageFromArray(resampled_volume_np)
    iso_image_sitk.SetOrigin(image_sitk.GetOrigin())
    iso_image_sitk.SetDirection(image_sitk.GetDirection())

    iso_image_sitk.SetSpacing(new_spacing)

    return iso_image_sitk


def process_nodule(idx, row):
    img_path = id_to_path[row.seriesuid]
    image = sitk.ReadImage(img_path)

    iso_image = resample_to_isotropic(image, new_spacing=[1.0, 1.0, 1.0])

    volume = sitk.GetArrayFromImage(iso_image)

    world_coord_xyz = (row.coordX, row.coordY, row.coordZ)

    nodule_voxel_zyx = world_to_voxel(iso_image, world_coord_xyz)
    nodule_diameter = row.diameter_mm

    nodule_voxel_int_zyx = np.round(nodule_voxel_zyx).astype(int)
    z, y, x = nodule_voxel_int_zyx

    assert (
        0 <= z < volume.shape[0]
        and 0 <= y < volume.shape[1]
        and 0 <= x < volume.shape[2]
    )

    scan_nodules = annotations_df[annotations_df.seriesuid == row.seriesuid]
    other_coords = np.array(
        [
            world_to_voxel(iso_image, [r.coordX, r.coordY, r.coordZ])
            for i, r in scan_nodules.iterrows()
            if i != idx
        ]
    )
    other_coords = np.floor(other_coords).astype(int)

    return volume, nodule_voxel_int_zyx, nodule_diameter, other_coords


def generate_crop(volume, nodule_coords, other_coords, nodule_diameter):
    z, y, x = nodule_coords
    volume_shape = volume.shape

    crop_mm = int(random.randint(max(min_crop_mm, nodule_diameter * 2), max_crop_mm))

    z_lower = int(z - crop_mm + nodule_diameter)
    z_upper = int(z - nodule_diameter)
    y_lower = int(y - crop_mm + nodule_diameter)
    y_upper = int(y - nodule_diameter)
    x_lower = int(x - crop_mm + nodule_diameter)
    x_upper = int(x - nodule_diameter)

    z0 = random.randint(z_lower, z_upper)
    y0 = random.randint(y_lower, y_upper)
    x0 = random.randint(x_lower, x_upper)

    zf = z0 + crop_mm
    yf = y0 + crop_mm
    xf = x0 + crop_mm

    assert z0 >= 0 and zf < volume_shape[0]
    assert y0 >= 0 and yf < volume_shape[1]
    assert x0 >= 0 and xf < volume_shape[2]

    for z_o, y_o, x_o in other_coords:
        assert not (
            (z0 > z_o + clearance and zf < z_o - clearance)
            and (y0 > y_o + clearance and yf < y_o - clearance)
            and (x0 > x_o + clearance and xf < x_o - clearance)
        )

    cropped_volume = torch.from_numpy(volume[z0:zf, y0:yf, x0:xf]).float()

    rel_z = (z - z0) / crop_mm
    rel_y = (y - y0) / crop_mm
    rel_x = (x - x0) / crop_mm

    resize_crop_volume = (
        F.interpolate(
            cropped_volume.unsqueeze(0).unsqueeze(0),
            size=(target_size, target_size, target_size),
            mode="trilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    return resize_crop_volume.numpy(), (rel_z, rel_y, rel_x)


records = []

for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df)):
    scan_id = row.seriesuid

    try:
        volume, nodule_coords, nodule_diameter, other_coords = process_nodule(idx, row)
    except Exception:
        print(f"Failed to process {row.seriesuid}")
        continue

    attempts = 0
    count = 0
    while attempts < max_attempts and count < num_augmentations:
        try:
            new_volume, (rel_z, rel_y, rel_x) = generate_crop(
                volume, nodule_coords, other_coords, nodule_diameter
            )

            count += 1
        except Exception:
            attempts += 1

            continue

        series_uid = f"{idx:05}_{count}"
        out_path = os.path.join(output_root, f"{series_uid}.npy")
        np.save(out_path, new_volume.astype(np.float16))

        records.append(
            {
                "series_uid": series_uid,
                "scan_id": scan_id,
                "node_id": idx,
                "aug_id": count,
                "rel_x": float(rel_x),
                "rel_y": float(rel_y),
                "rel_z": float(rel_z),
            }
        )

records_df = pd.DataFrame(records)
records_df.to_csv(os.path.join(output_root, "targets.csv"), index=False)
