import os
from typing import List, Tuple

from dinov2.inference import *


SeriesUID = str
Path = str


def discover_series(root: Path) -> List[Tuple[SeriesUID, Path]]:

    series = []

    for f in os.listdir(root):
        if not f.endswith(".nii.gz"):
            continue

        path = os.path.join(root, f)
        series_uid = f.replace(".nii.gz", "")

        series.append((series_uid, path))

    return series


if __name__ == "__main__":
    DATA_ROOT = "/mnt/typhon/data/AI/DeepRDT/OSIC/ct"
    OUTPUT_PATH = "/scratch/VM/radio-foundation/embeddings/DINO3D/OSIC/chp100k"
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
    )
