import os
import torch
import numpy as np
from einops import rearrange
from typing import Tuple, Sequence

import pydicom
import nibabel as nib
import SimpleITK as sitk
import warnings

Path = str
Spacing = Sequence[float]


def is_spacing_valid(spacing):
    return all(x > 0 for x in spacing) and len(spacing) == 3


def is_HU(img):
    img = torch.clip(img, -1000, 1900)
    return img.min() < 800 and img.max() > -100


def load_dicom(folder_path: Path) -> Tuple[torch.Tensor, Spacing]:
    dicom_files = []
    for f in os.listdir(folder_path):
        p = os.path.join(folder_path, f)
        if os.path.isfile(p):
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=False)
                dicom_files.append(ds)
            except Exception:
                continue

    if not dicom_files:
        raise ValueError("No DICOM files found.")

    ct_series = {}
    fallback_series = []

    for ds in dicom_files:
        modality = getattr(ds, "Modality", None)
        if modality and modality != "CT":
            continue

        series_uid = getattr(ds, "SeriesInstanceUID", None)
        if series_uid:
            ct_series.setdefault(series_uid, []).append(ds)
        else:
            fallback_series.append(ds)

    if ct_series:
        series_ds = max(ct_series.values(), key=len)
    elif fallback_series:
        series_ds = fallback_series
    else:
        raise ValueError("No CT slices found.")

    if hasattr(series_ds[0], "ImagePositionPatient"):
        series_ds.sort(key=lambda d: d.ImagePositionPatient[2])
    elif hasattr(series_ds[0], "InstanceNumber"):
        series_ds.sort(key=lambda d: int(d.InstanceNumber))
    else:
        raise ValueError("Cannot sort DICOM slices.")

    pixel_spacings = {
        tuple(map(float, d.PixelSpacing))
        for d in series_ds
        if hasattr(d, "PixelSpacing")
    }
    if len(pixel_spacings) > 1:
        warnings.warn(
            f"Inhomogeneous in-plane PixelSpacing detected: {pixel_spacings}",
            RuntimeWarning,
        )

    z_positions = [
        d.ImagePositionPatient[2]
        for d in series_ds
        if hasattr(d, "ImagePositionPatient")
    ]
    if len(z_positions) == len(series_ds):
        dz = np.diff(z_positions)
        if not np.allclose(dz, dz[0], atol=1e-3):
            warnings.warn(
                "Inhomogeneous slice spacing detected based on ImagePositionPatient.",
                RuntimeWarning,
            )
    else:
        warnings.warn(
            "Missing ImagePositionPatient for some slices; slice spacing consistency cannot be verified.",
            RuntimeWarning,
        )

    volume = np.stack([d.pixel_array for d in series_ds]).astype(np.float32)

    slope_values = {float(getattr(d, "RescaleSlope", 1.0)) for d in series_ds}
    intercept_values = {float(getattr(d, "RescaleIntercept", 0.0)) for d in series_ds}

    if len(slope_values) > 1 or len(intercept_values) > 1:
        warnings.warn(
            f"Inhomogeneous RescaleSlope/Intercept detected: "
            f"slopes={slope_values}, intercepts={intercept_values}",
            RuntimeWarning,
        )

    slope = float(getattr(series_ds[0], "RescaleSlope", 1.0))
    intercept = float(getattr(series_ds[0], "RescaleIntercept", 0.0))
    volume = volume * slope + intercept

    image_tensor = torch.from_numpy(volume).float()
    image_tensor = torch.clamp(image_tensor, -1000, 1000)

    pixel_spacing = series_ds[0].PixelSpacing
    if hasattr(series_ds[0], "SpacingBetweenSlices"):
        slice_spacing = float(series_ds[0].SpacingBetweenSlices)
    elif len(z_positions) == len(series_ds):
        slice_spacing = float(np.mean(np.diff(z_positions)))
    else:
        raise ValueError("Cannot determine slice spacing.")
    slice_spacing = abs(slice_spacing)

    spacing: Spacing = (
        slice_spacing,
        float(pixel_spacing[0]),
        float(pixel_spacing[1]),
    )

    assert is_spacing_valid(spacing), spacing
    assert is_HU(image_tensor)

    return image_tensor, spacing


def load_mhd(path: Path) -> Tuple[torch.Tensor, Spacing]:
    image_obj = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image_obj)
    spacing = image_obj.GetSpacing()
    spacing = list(np.array(spacing)[::-1])
    assert abs(spacing[2] - spacing[1]) < 0.001

    image = torch.from_numpy(image).float()
    image = image.clip(-1000, 1900)

    assert is_spacing_valid(spacing)
    assert is_HU(image)

    return image, spacing


def load_nifti(path: Path) -> Tuple[torch.Tensor, Spacing]:
    nifti = nib.loadsave.load(path)
    image = nifti.get_fdata()  # type: ignore
    affine = nifti.affine  # type: ignore

    s = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    spacing = (float(s[2]), float(s[1]), float(s[0]))
    assert abs(spacing[2] - spacing[1]) < 0.001

    image = torch.from_numpy(image).float()
    image = rearrange(image, "x y z -> z y x")
    image = image.clip(-1000, 1900)

    assert is_spacing_valid(spacing)
    assert is_HU(image)

    return image, spacing


def load_npy(path: Path) -> Tuple[torch.Tensor, Spacing]:
    image = torch.from_numpy(np.load(path)).float()
    spacing = [1.0, 1.0, 1.0]
    return image, spacing


def autoload_ct(path_or_folder: Path) -> Tuple[torch.Tensor, Spacing]:
    """
    Automatically load a CT scan from a folder or file by guessing the type.
    Supports DICOM folders, .mhd, NIfTI (.nii, .nii.gz), and .npy files.
    Returns: (image_tensor, spacing)
    """
    if os.path.isdir(path_or_folder):
        try:
            return load_dicom(path_or_folder)
        except Exception as e:
            raise RuntimeError(f"Failed to load DICOM folder: {e}")
    elif os.path.isfile(path_or_folder):
        ext = os.path.splitext(path_or_folder)[-1].lower()
        if ext in [".mhd"]:
            try:
                return load_mhd(path_or_folder)
            except Exception as e:
                raise RuntimeError(f"Failed to load MHD file: {e}")
        elif ext in [".nii", ".gz"]:
            try:
                return load_nifti(path_or_folder)
            except Exception as e:
                raise RuntimeError(f"Failed to load NIfTI file: {e}")
        elif ext in [".npy"]:
            try:
                return load_npy(path_or_folder)
            except Exception as e:
                raise RuntimeError(f"Failed to load .npy file: {e}")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        raise FileNotFoundError(f"No such file or directory: {path_or_folder}")
