import os
from glob import glob
from typing import List, Tuple

from dinov2.inference import *


SeriesUID = str
Path = str


def discover_series(root: Path) -> List[Tuple[SeriesUID, Path]]:

    nib_file_paths = glob(os.path.join(root, "**/*.nii.gz"), recursive=True)
    uids_to_paths = [
        (os.path.basename(p).replace(".nii.gz", ""), p) for p in nib_file_paths
    ]

    return uids_to_paths


if __name__ == "__main__":
    DATA_ROOT = "/mnt/logos/scratch/h501uvma/CT-RATE/dataset/train_fixed"
    OUTPUT_PATH = "/scratch/VM/radio-foundation/embeddings/DINO3D/CT_RATE/chp100k/train"
    CONFIG_PATH = (
        "/home/48078029W/projects/radio-orig/runs/vitb_3d_multires/config.yaml"
    )
    CHECKPOINT_PATH = (
        "/home/48078029W/projects/radio-orig/"
        "runs/vitb_3d_multires/eval/training_99999/teacher_checkpoint.pth"
    )

    DEVICES = [0, 1]
    MAX_PATCHES = 25000

    series_paths = discover_series(DATA_ROOT)

    success_uids, failed_uids = run_embedding_pipeline(
        series_paths=series_paths,
        output_path=OUTPUT_PATH,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        max_patches=MAX_PATCHES,
        devices=DEVICES,
    )
