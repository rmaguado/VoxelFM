import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F
from einops import rearrange

from dinov2.inference import load_nifti


data_root = "/scratch/VM/radio-foundation/datasets-nodicom/Totalsegmentator"
volumes_path_root = os.path.join(data_root, "TS")

output_root = "/scratch/VM/radio-foundation/preprocessed/TotalSegmentator"
img_out_dir = os.path.join(output_root, "images")
seg_out_dir = os.path.join(output_root, "segmentations")
os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(seg_out_dir, exist_ok=True)

metadata_csv = os.path.join(output_root, "metadata.csv")

cube_size = 112
stride = 56
device_id = 0


with open("evaluation/TotalSegmentator/map_to_labels.json", "r") as f:
    map_to_labels = json.load(f)

n_labels = 117
labels_ordered = [map_to_labels[str(l)] for l in range(1, n_labels + 1)]


def load_segmentation(path: str) -> torch.Tensor:
    nifti = nib.load(path)  # type: ignore
    seg = nifti.get_fdata()  # type: ignore
    seg = torch.from_numpy(seg).float()
    seg = rearrange(seg, "x y z -> z y x")
    return seg


def resize_isotropic(img, spacing, device, min_spacing=0.75, mode="trilinear"):
    img_device = img.device
    D, H, W = img.shape
    sD, sH, sW = spacing

    if sD == sH == sW:
        return img

    img = img.to(device)
    ts = max(min(spacing), min_spacing)

    new_D = round(D * sD / ts)
    new_H = round(H * sH / ts)
    new_W = round(W * sW / ts)

    args = {
        "input": img.unsqueeze(0).unsqueeze(0),
        "size": (new_D, new_H, new_W),
        "mode": mode,
    }
    if mode != "nearest":
        args["align_corners"] = False

    img_resampled = F.interpolate(**args).squeeze(0).squeeze(0)
    return img_resampled.to(img_device)


def generate_crops(img, seg, cube_size, stride):
    assert stride <= cube_size

    D, H, W = img.shape
    min_voxels = (cube_size**3) * 0.15

    pad_D = max(cube_size - D, 0)
    pad_H = max(cube_size - H, 0)
    pad_W = max(cube_size - W, 0)

    if pad_D > 0 or pad_H > 0 or pad_W > 0:
        img = (
            F.pad(
                img.unsqueeze(0).unsqueeze(0),
                (
                    pad_W // 2,
                    pad_W - pad_W // 2,
                    pad_H // 2,
                    pad_H - pad_H // 2,
                    pad_D // 2,
                    pad_D - pad_D // 2,
                ),
                mode="constant",
                value=0,
            )
            .squeeze(0)
            .squeeze(0)
        )
        seg = (
            F.pad(
                seg.unsqueeze(0).unsqueeze(0),
                (
                    pad_W // 2,
                    pad_W - pad_W // 2,
                    pad_H // 2,
                    pad_H - pad_H // 2,
                    pad_D // 2,
                    pad_D - pad_D // 2,
                ),
                mode="constant",
                value=0,
            )
            .squeeze(0)
            .squeeze(0)
        )
        D, H, W = img.shape

    def starts(dim):
        if dim <= cube_size:
            return [0]
        s = list(range(0, dim - cube_size + 1, stride))
        if s[-1] != dim - cube_size:
            s.append(dim - cube_size)
        return s

    for z in starts(D):
        for y in starts(H):
            for x in starts(W):
                seg_crop = seg[z : z + cube_size, y : y + cube_size, x : x + cube_size]
                if (seg_crop > 0).sum() < min_voxels:
                    continue
                img_crop = img[z : z + cube_size, y : y + cube_size, x : x + cube_size]
                yield img_crop, seg_crop


volume_names = sorted(os.listdir(volumes_path_root))
crop_num = 0
rows = []

for volume_name in tqdm(volume_names):
    volume_dir = os.path.join(volumes_path_root, volume_name)
    volume_path = os.path.join(volume_dir, "ct.nii.gz")
    segmentations_dir = os.path.join(volume_dir, "segmentations")

    img, spacing = load_nifti(volume_path)

    seg_total = torch.zeros_like(img)

    for fname in os.listdir(segmentations_dir):
        label_name = fname.replace(".nii.gz", "")
        if label_name not in labels_ordered:
            continue
        label_id = labels_ordered.index(label_name) + 1
        mask = load_segmentation(os.path.join(segmentations_dir, fname)) > 0
        seg_total[mask] = label_id

    img = resize_isotropic(img, spacing, device_id)
    seg_total = resize_isotropic(seg_total, spacing, device_id, mode="nearest")

    for img_crop, seg_crop in generate_crops(img, seg_total, cube_size, stride):
        img_np = img_crop.cpu().numpy().astype(np.float32)
        seg_np = seg_crop.cpu().numpy().astype(np.int16)

        crop_id = f"{crop_num:07d}"

        img_path = os.path.join(img_out_dir, f"{crop_id}.npy")
        seg_path = os.path.join(seg_out_dir, f"{crop_id}.npy")

        np.save(img_path, img_np)
        np.save(seg_path, seg_np)

        rows.append(
            {
                "crop_id": crop_id,
                "volume_name": volume_name,
                "img_path": img_path,
                "seg_path": seg_path,
            }
        )

        crop_num += 1

df = pd.DataFrame(rows)
df.to_csv(metadata_csv, index=False)

print(f"Done. Saved {len(df)} crops.")
