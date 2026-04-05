import os
from typing import List, Tuple
import polars as pl

from dinov2.inference import *


SeriesUID = str
Path = str
Job = Tuple[SeriesUID, Path]


def discover_series(root: Path) -> List[Job]:
    """
    Prepares (SeriesUID, series_dir_path) pairs from RSNA-PE dataset.
    """
    dicom_root = os.path.join(root, "train")

    jobs: List[Job] = []

    study_paths = [os.path.join(dicom_root, p) for p in os.listdir(dicom_root)]
    for study_path in study_paths:
        series_uids = os.listdir(study_path)

        for series_uid in series_uids:
            series_path = os.path.join(study_path, series_uid)
            if any(f.endswith(".dcm") for f in os.listdir(series_path)):
                jobs.append((series_uid, series_path))

    jobs.sort(key=lambda x: x[0])
    return jobs


if __name__ == "__main__":
    DATA_ROOT = "/scratch/VM/radio-foundation/datasets/RSNA-PE"
    OUTPUT_PATH = "/scratch/VM/radio-foundation/embeddings/DINO3D/RSNA_PE/chp100k"
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
