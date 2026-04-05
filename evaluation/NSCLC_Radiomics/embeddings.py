import os
from glob import glob
from typing import List, Tuple

from dinov2.inference import *


SeriesUID = str
Path = str


def discover_series(root: Path) -> List[Tuple[SeriesUID, Path]]:

    series = []

    for series_uid in os.listdir(root):
        path = os.path.join(root, series_uid)
        if os.path.isfile(path):
            continue

        dcm_paths_all = glob(os.path.join(path, "**/*.dcm"), recursive=True)
        dcm_paths = list(set([os.path.dirname(x) for x in dcm_paths_all]))

        count = 0
        for idx, dcm_path in enumerate(dcm_paths):
            dcm_files = [x for x in os.listdir(dcm_path) if x.endswith(".dcm")]
            if len(dcm_files) > 1:
                series.append((series_uid, dcm_path))
                count += 1
        if count == 0:
            print(f"{series_uid} has no data.")
        assert count <= 1, count

    return series


if __name__ == "__main__":
    DATA_ROOT = "/mnt/typhon/data/AI/DeepRDT/NSCLC-Radiomics"
    OUTPUT_PATH = (
        "/scratch/VM/radio-foundation/embeddings/DINO3D/NSCLC_Radiomics/chp100k"
    )
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
