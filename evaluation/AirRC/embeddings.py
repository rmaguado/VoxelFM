import os
from typing import List, Tuple
import polars as pl

from dinov2.inference import *


SeriesUID = str
Path = str
Job = Tuple[SeriesUID, Path]


def discover_series(root: Path) -> List[Job]:
    csv_path = os.path.join(root, "metadata.csv")

    metadata = pl.read_csv(csv_path)
    img_paths = metadata["img_path"].to_list()

    jobs: List[Job] = []

    for img_path in img_paths:
        series_uid = os.path.basename(img_path).replace(".npy", "")
        jobs.append((series_uid, img_path))

    jobs.sort(key=lambda x: x[0])
    return jobs


if __name__ == "__main__":
    DATA_ROOT = "/scratch/VM/radio-foundation/preprocessed/AirRC"
    OUTPUT_PATH = "/scratch/VM/radio-foundation/embeddings/DINO3D/AirRC/chp100k"
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
