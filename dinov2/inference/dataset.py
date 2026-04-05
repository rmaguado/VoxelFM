import os
from typing import List, Tuple

import pydicom


SeriesUID = str
Path = str


def discover_dicom_series(root: Path) -> List[Tuple[SeriesUID, Path]]:
    """
    Discovers DICOM series under `root`.

    Returns:
        List of (SeriesInstanceUID, series_dir_path)
    """
    uid_to_path = {}
    discovered = []

    for dirpath, _, filenames in os.walk(root):
        dcm_files = [f for f in filenames if not f.startswith(".")]
        if not dcm_files:
            continue

        first_file = os.path.join(dirpath, dcm_files[0])

        try:
            ds = pydicom.dcmread(
                first_file,
                stop_before_pixels=True,
                specific_tags=["SeriesInstanceUID"],
            )
            uid = str(ds.SeriesInstanceUID)
        except Exception:
            continue

        if uid in uid_to_path:
            raise ValueError("Duplicate SeriesInstanceUID detected")

        uid_to_path[uid] = dirpath
        discovered.append((uid, dirpath))

    discovered.sort(key=lambda x: x[0])
    return discovered


def discover_nifti_series(root: Path) -> List[Tuple[SeriesUID, Path]]:
    """
    Discovers NIfTI volumes under `root`.

    UID is derived from filename without extension.
    """
    uid_to_path = {}
    discovered = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
                continue

            path = os.path.join(dirpath, fname)

            uid = fname
            if uid.endswith(".nii.gz"):
                uid = uid[:-7]
            elif uid.endswith(".nii"):
                uid = uid[:-4]

            if uid in uid_to_path:
                raise ValueError("Duplicate NIfTI UID detected")

            uid_to_path[uid] = path
            discovered.append((uid, path))

    discovered.sort(key=lambda x: x[0])
    return discovered
