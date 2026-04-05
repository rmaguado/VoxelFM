import os
from typing import List, Tuple

from dinov2.inference import *

SeriesUID = str
Path = str


def discover_series(root: Path) -> List[Tuple[SeriesUID, Path]]:
    files = [x for x in os.listdir(root) if x.endswith(".npy")]

    jobs = []

    for f in files:
        series_uids = f.replace(".npy", "")
        series_path = os.path.join(root, f)

        jobs.append((series_uids, series_path))

    jobs.sort(key=lambda x: x[0])
    return jobs


if __name__ == "__main__":
    DATA_ROOT = "/scratch/VM/radio-foundation/preprocessed/LUNA16"
    OUTPUT_PATH = "/scratch/VM/radio-foundation/embeddings/DINO3D/LUNA16/chp100k"
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
        do_preprocessing=False,
    )
