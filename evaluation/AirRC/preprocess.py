import os
from glob import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

import torch
import torch.nn.functional as F

torch.manual_seed(4)


images_root = "/scratch/VM/radio-foundation/essential/LUNA16"
segmentation_root = "/scratch/VM/radio-foundation/AirRC/labelsTr"
output_root = "/scratch/VM/radio-foundation/preprocessed/AirRC"

output_img = os.path.join(output_root, "images")
output_seg = os.path.join(output_root, "segmentations")
os.makedirs(output_img, exist_ok=True)
os.makedirs(output_seg, exist_ok=True)


img_paths = glob(os.path.join(images_root, "subset*/**/*.mhd"), recursive=True)
seg_paths = glob(os.path.join(segmentation_root, "*.nii.gz"), recursive=True)

id_to_impath = {p.split("/")[-1].replace(".mhd", ""): p for p in img_paths}
id_to_segpath = {p.split("/")[-1].replace(".nii.gz", ""): p for p in seg_paths}

id_paths = {id: (id_to_impath[id], seg_path) for id, seg_path in id_to_segpath.items()}
print(f"{len(id_paths)} paired volumes and segmentations.")


def resample_img(img, new_spacing=(1.0, 1.0, 1.0)):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(img.GetPixelID())

    return resampler.Execute(img)


def resample_seg_from_img(seg, reference_img):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(seg)


def isotropic_random_resize_crop_3d(
    img: torch.Tensor,
    seg: torch.Tensor,
    output_size: int = 112,
    num_augs: int = 16,
    scale_range: Tuple[float, float] = (0.2, 0.6),
    pad_value: int = -1000,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    assert img.shape == seg.shape
    assert img.ndim == 3

    D, H, W = img.shape
    img_dtype = img.dtype
    seg_dtype = seg.dtype

    results = []

    for _ in range(num_augs):
        scale = torch.empty(1).uniform_(*scale_range).item()

        crop_D = max(1, int(round(D * scale)))
        crop_H = max(1, int(round(H * scale)))
        crop_W = max(1, int(round(W * scale)))

        pad_d = max(crop_D - D, 0)
        pad_h = max(crop_H - H, 0)
        pad_w = max(crop_W - W, 0)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_d // 2,
                pad_d - pad_d // 2,
            )
            img_p = F.pad(img, padding, value=pad_value)
            seg_p = F.pad(seg, padding, value=0)
        else:
            img_p = img
            seg_p = seg

        Dp, Hp, Wp = img_p.shape

        z0 = torch.randint(0, Dp - crop_D + 1, (1,)).item()
        y0 = torch.randint(0, Hp - crop_H + 1, (1,)).item()
        x0 = torch.randint(0, Wp - crop_W + 1, (1,)).item()

        img_crop = img_p[
            z0 : z0 + crop_D,
            y0 : y0 + crop_H,
            x0 : x0 + crop_W,
        ]
        seg_crop = seg_p[
            z0 : z0 + crop_D,
            y0 : y0 + crop_H,
            x0 : x0 + crop_W,
        ]

        img_rs = F.interpolate(
            img_crop.unsqueeze(0).unsqueeze(0),
            size=(output_size, output_size, output_size),
            mode="trilinear",
            align_corners=False,
        )
        seg_rs = F.interpolate(
            seg_crop.unsqueeze(0).unsqueeze(0).float(),
            size=(output_size, output_size, output_size),
            mode="nearest",
        )

        img_rs = img_rs.squeeze(0).squeeze(0).to(img_dtype)
        seg_rs = seg_rs.squeeze(0).squeeze(0).to(seg_dtype)

        results.append((img_rs, seg_rs))

    return results


records = []

for idx, (sid, (img_path, seg_path)) in tqdm(enumerate(id_paths.items())):
    img = sitk.ReadImage(img_path)
    img = resample_img(img)
    seg = sitk.ReadImage(seg_path)
    seg = resample_seg_from_img(seg, img)

    img_pt = torch.from_numpy(sitk.GetArrayFromImage(img)).float()
    seg_pt = torch.from_numpy(sitk.GetArrayFromImage(seg)).long()

    img_pt = img_pt.clip(-1000, 1900)

    augmentations = isotropic_random_resize_crop_3d(
        img_pt, seg_pt, num_augs=16, output_size=112
    )

    for aug_id, (img_aug, seg_aug) in enumerate(augmentations):
        series_uid = f"{idx:05}_{aug_id}"
        out_name = f"{series_uid}.npy"
        out_img = os.path.join(output_img, out_name)
        out_seg = os.path.join(output_seg, out_name)
        np.save(out_img, img_aug.numpy())
        np.save(out_seg, seg_aug.numpy())

        records.append(
            {
                "crop_id": series_uid,
                "volume_name": sid,
                "img_path": out_img,
                "seg_path": out_seg,
            }
        )

records_df = pd.DataFrame(records)
records_df.to_csv(os.path.join(output_root, "metadata.csv"), index=False)
