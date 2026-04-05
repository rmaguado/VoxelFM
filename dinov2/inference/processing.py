import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from omegaconf import DictConfig, ListConfig
from einops import rearrange
from dinov2.models import DinoVisionTransformer
from dinov2.inference import load_nifti, load_dicom


@dataclass
class VoxelFeatures:
    cls: torch.Tensor
    patch: torch.Tensor
    img: torch.Tensor


def crop_volume(img, device, k=13):
    """
    removes empty areas around the patient
    input image must be in hounsfield units
    """

    img_device = img.device
    img = img.to(device)

    x_mask = img > -800
    vol_f = x_mask.float().unsqueeze(0).unsqueeze(0)

    kernel = torch.ones((1, 1, k, k, k), dtype=torch.float32).to(device)

    conv_sum = F.conv3d(vol_f, kernel, padding="same")

    out = conv_sum > k * k * k * 0.5

    mask = out[0, 0].bool()

    coords = mask.nonzero(as_tuple=False)

    if coords.numel() == 0:
        raise ValueError("Mask is empty — no bounding box to extract.")

    z_min, y_min, x_min = coords.min(dim=0).values
    z_max, y_max, x_max = coords.max(dim=0).values

    z_min = max(z_min.item() - k // 2, 0)
    y_min = max(y_min.item() - k // 2, 0)
    x_min = max(x_min.item() - k // 2, 0)

    z_max = min(z_max.item() + k // 2, mask.shape[0] - 1)
    y_max = min(y_max.item() + k // 2, mask.shape[1] - 1)
    x_max = min(x_max.item() + k // 2, mask.shape[2] - 1)

    bbox = (slice(z_min, z_max + 1), slice(y_min, y_max + 1), slice(x_min, x_max + 1))

    cropped_img = img[bbox]
    return cropped_img.to(img_device)


def resize_isotropic(img, spacing, device, min_spacing=0.75):
    img_device = img.device

    D, H, W = img.shape
    sD, sH, sW = spacing

    if sD == sH == sW:
        return img

    img = img.to(device)

    ts = max(min(spacing), min_spacing)

    new_D = round(D * sD / ts)
    new_H = round(H * sH / ts)
    new_W = round(W * sW / ts)

    img_resampled = (
        F.interpolate(
            img.unsqueeze(0).unsqueeze(0),
            size=(new_D, new_H, new_W),
            mode="trilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    return img_resampled.to(img_device)


def patch_crop(img, patch_size):
    D, H, W = img.shape

    mod_D, mod_H, mod_W = D % patch_size, H % patch_size, W % patch_size

    return img[
        mod_D // 2 : D - mod_D + mod_D // 2,
        mod_H // 2 : H - mod_H + mod_H // 2,
        mod_W // 2 : W - mod_W + mod_W // 2,
    ]


def resize_max_length(img, target_size):
    D, H, W = img.shape

    r = target_size / max(img.shape)

    new_D, new_H, new_W = int(D * r), int(H * r), int(W * r)

    return (
        F.interpolate(
            img.unsqueeze(0).unsqueeze(0),
            (new_D, new_H, new_W),
            align_corners=False,
            mode="trilinear",
        )
        .squeeze(0)
        .squeeze(0)
    )


def resize_max_patches(img, patch_size, max_patches, device):
    D, H, W = img.shape
    pD, pH, pW = D // patch_size, H // patch_size, W // patch_size
    num_patches = pD * pH * pW
    if num_patches <= max_patches:
        return img

    img_device = img.device
    img = img.to(device)

    scale = (max_patches / num_patches) ** (1 / 3)

    def num_patches_for(scale):
        new_D, new_H, new_W = int(D * scale), int(H * scale), int(W * scale)
        return (
            (new_D // patch_size) * (new_H // patch_size) * (new_W // patch_size),
            new_D,
            new_H,
            new_W,
        )

    best = None
    for delta in [0.98, 0.99, 1.0, 1.01, 1.02]:
        patches, new_D, new_H, new_W = num_patches_for(scale * delta)
        if patches <= max_patches:
            if best is None or patches > best[0]:
                best = (patches, new_D, new_H, new_W)

    if best is None:
        best = num_patches_for(scale * 0.95)

    _, new_D, new_H, new_W = best

    img = (
        torch.nn.functional.interpolate(
            img.unsqueeze(0).unsqueeze(0),
            size=(new_D, new_H, new_W),
            mode="trilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    return img.to(img_device)


def generate_embeddings(
    img,
    model,
    patch_size,
):
    _, D, H, W = img.shape

    with torch.inference_mode():
        features = model(img)

    D_patch = D // patch_size
    H_patch = H // patch_size
    W_patch = W // patch_size

    features = {k: v.cpu() for k, v in features.items() if isinstance(v, torch.Tensor)}

    cls_tokens = features["x_norm_clstoken"]
    patch_tokens = features["x_norm_patchtokens"]
    patch_tokens = rearrange(
        patch_tokens, "1 (d h w) e -> d h w e", d=D_patch, h=H_patch, w=W_patch
    )

    return {
        "cls": cls_tokens,
        "patch": patch_tokens,
    }


def extract_features(
    model: DinoVisionTransformer,
    config: DictConfig | ListConfig,
    image: Union[str, np.ndarray, torch.Tensor],
    device: torch.device,
    crop_background: bool = False,
    max_patches: Optional[int] = None,
) -> VoxelFeatures:
    patch_size = config.student.patch_size
    fmean = config.datasets[0].norm.mean
    fstd = config.datasets[0].norm.std

    if isinstance(image, (np.ndarray, torch.Tensor)):
        img = (
            torch.as_tensor(image, dtype=torch.float32)
            if isinstance(image, np.ndarray)
            else image
        )
        spacing = (1.0, 1.0, 1.0)
    elif isinstance(image, str):
        if image.endswith((".nii.gz", ".nii")):
            img, spacing = load_nifti(image)
        elif image.endswith(".npy"):
            img = torch.from_numpy(np.load(image))
            spacing = (1.0, 1.0, 1.0)
        elif os.path.isdir(image) and any(
            x.endswith((".dcm", ".DCM")) for x in os.listdir(image)
        ):
            img, spacing = load_dicom(image)
        else:
            raise ValueError(f"Could not identify image type for path: {image!r}")
    else:
        raise TypeError(
            f"image must be a file path, numpy array, or torch.Tensor; got {type(image).__name__!r}"
        )

    if img.ndim != 3:
        raise ValueError(
            f"Input volume must have exactly 3 dimensions (D, H, W); got shape {tuple(img.shape)}"
        )

    if any(dim % patch_size > 0 for dim in img.shape):
        warnings.warn(
            f"Volume dimensions are not perfectly divisible by patch_size={patch_size}. The volume will be cropped to the nearest multiple.",
            stacklevel=2,
        )

    if crop_background:
        img = crop_volume(img, device, 21)

    img = resize_isotropic(img, spacing, device)

    if max_patches:
        img = resize_max_patches(img, patch_size, max_patches, device)

    img = patch_crop(img, patch_size)

    x_prep = (img - fmean) / fstd  # type: ignore
    x_prep = x_prep.unsqueeze(0).to(device=device, dtype=torch.float)

    with torch.no_grad():
        features = generate_embeddings(x_prep, model, patch_size)

    return VoxelFeatures(cls=features["cls"], patch=features["patch"], img=img)
