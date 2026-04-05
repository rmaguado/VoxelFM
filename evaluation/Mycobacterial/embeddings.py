import os
from typing import List, Tuple
import torch
import numpy as np
from einops import rearrange
import nibabel as nib

from dinov2.inference import *


SeriesUID = str
Path = str


def reverse_norm(img, window_center=-600, window_width=1500):
    return img * window_width + window_center - window_width / 2


def load_mbnifti(path):
    nifti = nib.loadsave.load(path)
    image = nifti.get_fdata()  # type: ignore
    affine = nifti.affine  # type: ignore

    s = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    spacing = (float(s[2]), float(s[1]), float(s[0]))
    assert abs(spacing[2] - spacing[1]) < 0.001

    image = torch.from_numpy(image).float()
    image = rearrange(image, "x y z -> z y x")
    image = reverse_norm(image)
    image = image.clip(-1000, 1900)

    assert is_spacing_valid(spacing)

    return image, spacing


def discover_series(root: Path) -> List[Tuple[SeriesUID, Path]]:
    series = []

    NTM_PATH = os.path.join(root, "NTMNiFTi")
    TB_PATH = os.path.join(root, "TBNifTI")

    ntm_paths = [
        os.path.join(NTM_PATH, x) for x in os.listdir(NTM_PATH) if x.endswith(".nii")
    ]
    tb_paths = [
        os.path.join(TB_PATH, x) for x in os.listdir(TB_PATH) if x.endswith(".nii")
    ]

    all_paths = ntm_paths + tb_paths

    for series_path in all_paths:
        series_uid = os.path.basename(series_path).replace(".nii", "")

        series.append((series_uid, series_path))

    return series


if __name__ == "__main__":
    DATA_ROOT = (
        "/mnt/typhon/data/AI/DeepRDT/Mycobacterial/damianhan/nifti-dataset/versions/5/"
    )
    OUTPUT_PATH = "/scratch/VM/radio-foundation/embeddings/DINO3D/Mycobacterial/chp100k"
    CONFIG_PATH = (
        "/home/48078029W/projects/radio-orig/runs/vitb_3d_multires/config.yaml"
    )
    CHECKPOINT_PATH = (
        "/home/48078029W/projects/radio-orig/"
        "runs/vitb_3d_multires/eval/training_99999/teacher_checkpoint.pth"
    )

    DEVICES = [0]
    MAX_PATCHES = 25000

    series_paths = discover_series(DATA_ROOT)

    success_uids, failed_uids = run_embedding_pipeline(
        series_paths=series_paths,
        output_path=OUTPUT_PATH,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        max_patches=MAX_PATCHES,
        devices=DEVICES,
        load_fn=load_mbnifti,
    )
