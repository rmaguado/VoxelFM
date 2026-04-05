import os
from typing import List, Tuple
import polars as pl

from dinov2.inference import *


SeriesUID = str
Path = str
Job = Tuple[SeriesUID, Path]


def discover_series(root: Path) -> List[Job]:
    img_paths = [os.path.join(root, x) for x in os.listdir(root)]

    series = []

    for img_path in img_paths:
        series_uid = os.path.basename(img_path).replace(".npy", "")
        series.append((series_uid, img_path))

    return series


if __name__ == "__main__":
    DATA_ROOT = "/scratch/VM/radio-foundation/preprocessed/Mediastinal/image"
    OUTPUT_PATH = "/scratch/VM/radio-foundation/embeddings/DINO3D/Mediastinal/chp100k"
    CONFIG_PATH = (
        "/home/48078029W/projects/radio-orig/runs/vitb_3d_multires/config.yaml"
    )
    CHECKPOINT_PATH = (
        "/home/48078029W/projects/radio-orig/"
        "runs/vitb_3d_multires/eval/training_99999/teacher_checkpoint.pth"
    )

    DEVICES = [0, 1, 2, 3]
    MAX_PATCHES = 25000

    series_paths = discover_series(DATA_ROOT)

    success_uids, failed_uids = run_embedding_pipeline(
        series_paths=series_paths,
        output_path=OUTPUT_PATH,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        max_patches=MAX_PATCHES,
        devices=DEVICES,
        do_preprocessing=False,
    )
