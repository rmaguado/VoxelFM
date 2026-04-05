import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import SimpleITK as sitk


DATA_ROOT = "/mnt/typhon/data/AI/DeepRDT/Mediastinal/manifest-1724359862867/"
OUTPUT_PATH = "/scratch/VM/radio-foundation/preprocessed/Mediastinal"


def load_data(ct_path, seg_path):
    img_reader = sitk.ImageSeriesReader()
    series_ids = img_reader.GetGDCMSeriesIDs(ct_path)
    series_files = img_reader.GetGDCMSeriesFileNames(ct_path, series_ids[0])
    img_reader.SetFileNames(series_files)
    img = img_reader.Execute()

    seg_reader = sitk.ImageSeriesReader()
    series_ids = seg_reader.GetGDCMSeriesIDs(seg_path)
    series_files = seg_reader.GetGDCMSeriesFileNames(seg_path, series_ids[0])
    seg_reader.SetFileNames(series_files)
    seg = seg_reader.Execute()

    if seg.GetDimension() == 4 and seg.GetSize()[-1] == 1:
        seg = sitk.Extract(seg, seg.GetSize()[:3] + (0,))
    assert seg.GetDimension() == 3

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    iso_spacing = min(original_spacing)

    new_spacing = (iso_spacing, iso_spacing, iso_spacing)
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    img_resampler = sitk.ResampleImageFilter()
    img_resampler.SetInterpolator(sitk.sitkLinear)
    img_resampler.SetOutputSpacing(new_spacing)
    img_resampler.SetSize(new_size)
    img_resampler.SetOutputDirection(img.GetDirection())
    img_resampler.SetOutputOrigin(img.GetOrigin())
    img_resampler.SetTransform(sitk.Transform())
    img_resampler.SetDefaultPixelValue(0)

    img_iso = img_resampler.Execute(img)

    seg_resampler = sitk.ResampleImageFilter()
    seg_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_resampler.SetOutputSpacing(new_spacing)
    seg_resampler.SetSize(new_size)
    seg_resampler.SetOutputDirection(img.GetDirection())
    seg_resampler.SetOutputOrigin(img.GetOrigin())
    seg_resampler.SetTransform(sitk.Transform())
    seg_resampler.SetDefaultPixelValue(0)

    seg_iso = seg_resampler.Execute(seg)

    img_np = sitk.GetArrayFromImage(img_iso)
    seg_np = sitk.GetArrayFromImage(seg_iso) / 255.0

    img_torch = torch.from_numpy(img_np).float()
    img_torch = img_torch.clip(-1000.0, 1900)
    seg_torch = torch.from_numpy(seg_np).bool()

    return img_torch, seg_torch


def random_resize_crops_3d_isotropic(
    ct: torch.Tensor,
    mask: torch.Tensor,
    output_size=(112, 112, 112),
    num_augs=8,
    scale_range=(0.5, 1.0),
    max_tries=100,
):
    if ct.dim() == 3:
        ct = ct.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)

    _, D, H, W = ct.shape
    pos = mask > 0

    if pos.sum() == 0:
        raise ValueError("Segmentation mask contains no positive voxels.")

    covered = torch.zeros_like(pos)
    ct_augs, mask_augs = [], []

    min_dim = min(D, H, W)
    tries = 0

    def sample_crop(center=None):
        scale = random.uniform(*scale_range)
        crop_size = max(1, int(scale * min_dim))

        if center is None:
            zc = random.randrange(D)
            yc = random.randrange(H)
            xc = random.randrange(W)
        else:
            _, zc, yc, xc = center

        z0 = max(0, min(zc - crop_size // 2, D - crop_size))
        y0 = max(0, min(yc - crop_size // 2, H - crop_size))
        x0 = max(0, min(xc - crop_size // 2, W - crop_size))

        ct_crop = ct[:, z0 : z0 + crop_size, y0 : y0 + crop_size, x0 : x0 + crop_size]
        mask_crop = mask[
            :, z0 : z0 + crop_size, y0 : y0 + crop_size, x0 : x0 + crop_size
        ]

        ct_resized = F.interpolate(
            ct_crop.unsqueeze(0),
            size=output_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        mask_resized = (
            F.interpolate(
                mask_crop.unsqueeze(0).float(),
                size=output_size,
                mode="nearest",
            )
            .squeeze(0)
            .bool()
        )

        return ct_resized, mask_resized, mask_crop.bool(), (z0, y0, x0, crop_size)

    while (covered & pos).sum() < pos.sum():
        if tries >= max_tries:
            raise RuntimeError("Failed to cover all positive voxels.")
        tries += 1

        uncovered = (pos & ~covered).nonzero(as_tuple=False)
        center = uncovered[random.randrange(uncovered.shape[0])]

        ct_r, mask_r, mask_crop, (z0, y0, x0, cs) = sample_crop(center)

        ct_augs.append(ct_r.squeeze(0))
        mask_augs.append(mask_r.squeeze(0))
        covered[:, z0 : z0 + cs, y0 : y0 + cs, x0 : x0 + cs] |= mask_crop

    while len(ct_augs) < num_augs:
        ct_r, mask_r, _, _ = sample_crop()
        ct_augs.append(ct_r.squeeze(0))
        mask_augs.append(mask_r.squeeze(0))

    return ct_augs, mask_augs


meta_df = pd.read_csv(os.path.join(DATA_ROOT, "metadata.csv"))
meta_df = meta_df[["Series UID", "Subject ID", "Modality", "File Location"]]

pat_ids = []
for pat_id, pat_df in meta_df.groupby("Subject ID"):
    seg_row = pat_df[pat_df["Modality"] == "SEG"].iloc[0]
    ct_row = pat_df[pat_df["Modality"] == "CT"].iloc[0]
    pat_ids.append(
        {
            "pat_id": pat_id,
            "seg_path": seg_row["File Location"],
            "ct_path": ct_row["File Location"],
        }
    )


ct_output_path = os.path.join(OUTPUT_PATH, "image")
seg_output_path = os.path.join(OUTPUT_PATH, "segmentations")
os.makedirs(ct_output_path, exist_ok=True)
os.makedirs(seg_output_path, exist_ok=True)

metadata_csv = os.path.join(OUTPUT_PATH, "metadata.csv")

rows = []
for patid_data in tqdm(pat_ids):
    patid = patid_data["pat_id"]
    ct_path = os.path.join(DATA_ROOT, patid_data["ct_path"])
    seg_path = os.path.join(DATA_ROOT, patid_data["seg_path"])

    try:
        img, seg = load_data(ct_path, seg_path)
    except Exception as e:
        print(f"Failed to load data for image: {patid}")
        continue

    try:
        ct_augs, seg_augs = random_resize_crops_3d_isotropic(
            img, seg, scale_range=(0.2, 0.8), num_augs=16
        )
    except Exception as e:
        print(f"Failed to generate augmentations for image: {patid}")
        continue

    for idx, (ct_aug, seg_aug) in enumerate(zip(ct_augs, seg_augs)):
        crop_id = f"{idx:02}"
        series_uid = f"{patid}_{crop_id}"
        output_name = f"{series_uid}.npy"

        ct_aug_np = ct_aug.numpy()
        seg_aug_np = seg_aug.numpy()

        img_path = os.path.join(ct_output_path, output_name)
        seg_path = os.path.join(seg_output_path, output_name)

        np.save(img_path, ct_aug_np)
        np.save(seg_path, seg_aug_np)

        rows.append(
            {
                "series_uid": series_uid,
                "crop_id": crop_id,
                "volume_name": patid,
                "img_path": img_path,
                "seg_path": seg_path,
            }
        )

df = pd.DataFrame(rows)
df.to_csv(metadata_csv, index=False)

print(f"Done. Saved {len(df)} crops.")
