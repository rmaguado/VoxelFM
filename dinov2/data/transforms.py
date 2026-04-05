import torch
import random
from typing import List, Tuple, Callable
import logging

logger = logging.getLogger("dinov2")


class RandomCrop3D:
    """
    Crops a 3D image to a random size that preserves the same proportions
    (aspect ratios) as `size`, then resamples it to the exact target `size`.

    Additionally, the mapping of target dimensions (D,H,W) to input axes
    is randomly permuted before cropping, then permuted back afterwards.

    Input tensor shape: (D, H, W)
    `size` can be an int (cubic output) or a tuple of three ints (D_out, H_out, W_out).
    """

    def __init__(
        self, size: int | Tuple[int, int, int], scale: Tuple[float, float] = (0.7, 1.0)
    ) -> None:
        if not (0 < scale[0] <= scale[1]):
            raise ValueError(
                f"Scale range must be positive and ordered, but got {scale}"
            )

        if isinstance(size, int):
            if size <= 0:
                raise ValueError(f"size must be a positive integer, but got {size}")
            self.crop_size = (size, size, size)
        elif (
            hasattr(size, "__len__")
            and len(size) == 3
            and all(isinstance(s, int) and s > 0 for s in size)
        ):
            self.crop_size = tuple(size)
        else:
            raise ValueError(
                "size must be a positive integer or a tuple of three positive integers"
            )

        self.scale = scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(
                f"Input image must be 3D (D, H, W), but got shape {img.shape}"
            )

        img_shape = tuple(int(x) for x in img.shape)  # (D,H,W)

        perm = list(torch.randperm(3).tolist())
        inv_perm = [perm.index(i) for i in range(3)]

        p_perm = [float(self.crop_size[i]) for i in perm]

        s = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())

        min_p = min(p_perm)
        min_img_dim = min(img_shape)

        desired_alpha = (min_img_dim * s) / min_p

        max_allowed_alpha = min(img_shape[i] / p_perm[i] for i in range(3))
        alpha = min(desired_alpha, max_allowed_alpha)

        crop_dims_perm = [max(1, int(round(alpha * p))) for p in p_perm]
        crop_dims_perm = [min(crop_dims_perm[i], img_shape[i]) for i in range(3)]

        starts = [random.randint(0, img_shape[i] - crop_dims_perm[i]) for i in range(3)]

        cropped = img[
            starts[0] : starts[0] + crop_dims_perm[0],
            starts[1] : starts[1] + crop_dims_perm[1],
            starts[2] : starts[2] + crop_dims_perm[2],
        ].float()

        cropped_for_resample = cropped.permute(inv_perm).contiguous()

        resampled = torch.nn.functional.interpolate(
            cropped_for_resample.unsqueeze(0).unsqueeze(0),
            size=self.crop_size,
            mode="trilinear",
            align_corners=False,
        )

        return resampled.squeeze(0).squeeze(0)


class Resize:
    def __init__(self, output_size: int) -> None:
        self.output_size = (output_size, output_size, output_size)

    def _resize_3d(self, img: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.interpolate(
                img.unsqueeze(0).unsqueeze(0),
                size=self.output_size,
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self._resize_3d(img)


class Permute:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        dims_order = torch.randperm(3).tolist()
        return img.permute(dims_order)


class Flip:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        flip_dims = [dim for dim in [0, 1, 2] if torch.rand(1).item() < 0.5]
        return img.flip(dims=flip_dims) if flip_dims else img


class Norm:
    def __init__(self, mean: float, std: float, vmin: float, vmax: float) -> None:
        self.mean = mean
        self.std = std
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.clip(self.vmin, self.vmax)
        return (img - self.mean) / self.std


class Window:
    def __init__(
        self,
        p: float = 0.5,
        percentiles: tuple[float, float] = (1.0, 99.0),
        level_std_ratio: float = 0.2,
        width_range_ratio: tuple[float, float] = (0.5, 2.0),
        hist_bins: int = 512,
        hist_range: tuple[int, int] = (-1000, 1900),
    ):
        self.p = p
        self.percentiles = torch.tensor(
            [percentiles[0] / 100.0, 0.25, 0.50, 0.75, percentiles[1] / 100.0],
            dtype=torch.float32,
        )
        self.level_std_ratio = level_std_ratio
        self.width_range_ratio = width_range_ratio
        self.hist_bins = hist_bins
        self.hist_range = hist_range

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img

        hist = torch.histogram(
            img.float(),
            bins=self.hist_bins,
            range=self.hist_range,
        ).hist

        cdf = torch.cumsum(hist, dim=0)
        total_pixels = cdf[-1]

        q_indices = torch.searchsorted(cdf, self.percentiles * total_pixels)
        q_indices = torch.clamp(q_indices, 0, self.hist_bins - 1)

        bin_width = (self.hist_range[1] - self.hist_range[0]) / self.hist_bins
        hu_values = self.hist_range[0] + q_indices * bin_width

        p_low, q1, median, q3, p_high = hu_values

        iqr = q3 - q1
        if iqr < 1.0:
            iqr = torch.clamp(p_high - p_low, min=1.0)

        level_std = iqr * self.level_std_ratio
        window_level = torch.normal(mean=median, std=level_std).item()

        min_width = iqr * self.width_range_ratio[0]
        max_width = iqr * self.width_range_ratio[1]
        window_width = (
            torch.empty(1).uniform_(min_width.item(), max_width.item()).item()
        )
        window_width = max(window_width, 10.0)

        window_min = window_level - (window_width / 2)
        window_max = window_level + (window_width / 2)

        return torch.clip(img, window_min, window_max)


class ImageTransforms:
    def __init__(self) -> None:
        self.transforms = []

    def __iadd__(self, new_transform: Callable):
        self.transforms.append(new_transform)
        return self

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            img = transform(img)
        return img


class SlabAwareRandomCrop3D(RandomCrop3D):
    def __init__(
        self,
        size: int | Tuple[int, int, int],
        scale: Tuple[float, float] = (0.7, 1.0),
        slab_axis: int = 0,
    ) -> None:
        super().__init__(size=size, scale=scale)
        if slab_axis not in (0, 1, 2):
            raise ValueError(f"slab_axis must be 0, 1, or 2, got {slab_axis}")
        self.slab_axis = slab_axis

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(
                f"Input image must be 3D (D, H, W), but got shape {img.shape}"
            )

        img_shape = tuple(int(x) for x in img.shape)
        slab_ax = self.slab_axis
        in_plane = [a for a in range(3) if a != slab_ax]  # two wide axes

        slab_crop_dim = img_shape[slab_ax]  # always take all available slices
        slab_out_dim = self.crop_size[slab_ax]  # target output depth

        if torch.rand(1).item() < 0.5:
            in_plane = in_plane[::-1]

        out_axes_order = [a for a in range(3) if a != slab_ax]
        ip_out_sizes = [
            float(self.crop_size[a]) for a in out_axes_order
        ]  # e.g. [224, 224]

        # img dims for the two chosen in-plane input axes
        ip_img_dims = [float(img_shape[a]) for a in in_plane]

        s = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())

        # Isotropic alpha over in-plane only:
        #   desired: smaller in-plane input dim * s == smaller in-plane output target
        min_ip_out = min(ip_out_sizes)
        min_ip_img = min(ip_img_dims)
        desired_alpha = (min_ip_img * s) / min_ip_out
        # clamp so neither in-plane crop exceeds its input dimension
        max_alpha = min(ip_img_dims[i] / ip_out_sizes[i] for i in range(2))
        alpha = min(desired_alpha, max_alpha)

        ip_crop_dims = [
            min(max(1, int(round(alpha * ip_out_sizes[i]))), int(ip_img_dims[i]))
            for i in range(2)
        ]

        crop_dims = [0, 0, 0]
        crop_dims[slab_ax] = slab_crop_dim
        crop_dims[in_plane[0]] = ip_crop_dims[0]
        crop_dims[in_plane[1]] = ip_crop_dims[1]

        starts = [0, 0, 0]
        starts[slab_ax] = 0  # slab already extracted to exact thickness
        starts[in_plane[0]] = random.randint(
            0, img_shape[in_plane[0]] - ip_crop_dims[0]
        )
        starts[in_plane[1]] = random.randint(
            0, img_shape[in_plane[1]] - ip_crop_dims[1]
        )

        cropped = img[
            starts[0] : starts[0] + crop_dims[0],
            starts[1] : starts[1] + crop_dims[1],
            starts[2] : starts[2] + crop_dims[2],
        ].float()

        perm = [0, 0, 0]
        perm[slab_ax] = slab_ax
        perm[out_axes_order[0]] = in_plane[0]
        perm[out_axes_order[1]] = in_plane[1]

        cropped_permuted = cropped.permute(perm).contiguous()

        resampled = torch.nn.functional.interpolate(
            cropped_permuted.unsqueeze(0).unsqueeze(0),
            size=self.crop_size,
            mode="trilinear",
            align_corners=False,
        )

        return resampled.squeeze(0).squeeze(0)


class SlabSampler:
    def __init__(
        self,
        slab_thickness: int,
        slab_axis: int | None = None,
    ) -> None:
        if slab_thickness < 1:
            raise ValueError(f"slab_thickness must be ≥ 1, got {slab_thickness}")
        if slab_axis is not None and slab_axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2 (or None), got {axis}")

        self.slab_thickness = slab_thickness
        self.slab_axis = slab_axis

    def sample_range(self, volume_dim: int) -> Tuple[int, int]:
        thickness = min(self.slab_thickness, volume_dim)
        max_start = volume_dim - thickness

        if max_start == 0:
            return 0, thickness

        start = random.randint(0, max_start)

        return start, start + thickness

    def __call__(self, image) -> torch.Tensor:

        shape = image.shape  # (D, H, W)
        if len(shape) != 3:
            raise ValueError(f"SlabSampler expects a 3-D volume, got shape {shape}")

        axis = self.slab_axis if self.slab_axis is not None else random.randint(0, 2)
        start, stop = self.sample_range(int(shape[axis]))

        slices = [slice(None), slice(None), slice(None)]
        slices[axis] = slice(start, stop)
        slices = tuple(slices)

        slab_raw = image[slices]
        slab = slab_raw.float().contiguous()

        perm = [i for i in range(3) if i != axis] + [axis]
        slab = slab.permute(*perm)

        return slab
