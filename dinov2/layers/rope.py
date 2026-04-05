# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn


class RopePositionEmbedding3D(nn.Module):
    periods: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        rescale_coords: Optional[float] = None,
    ):
        super().__init__()
        assert (
            embed_dim % (6 * num_heads) == 0
        ), "embed_dim must be divisible by 6 * num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.base = base
        self.rescale_coords = rescale_coords

        self.rot_pair_count = self.head_dim // 6
        # register periods/invfreq buffer (float32)
        periods = self.base ** (
            2.0
            * torch.arange(self.rot_pair_count, dtype=torch.float32)
            / (self.head_dim // 3)
        )
        self.register_buffer("periods", periods)

    def forward(self, *, D: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = torch.float32

        zd = torch.arange(0.5, D, device=device, dtype=dtype) / float(D)
        yh = torch.arange(0.5, H, device=device, dtype=dtype) / float(H)
        xw = torch.arange(0.5, W, device=device, dtype=dtype) / float(W)

        coords = torch.stack(
            torch.meshgrid(zd, yh, xw, indexing="ij"), dim=-1
        )  # [D,H,W,3]
        coords = coords.reshape(-1, 3)  # [N, 3]
        coords = 2.0 * coords - 1.0  # [-1, +1]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = (
                torch.empty(1, device=device, dtype=dtype)
                .uniform_(rescale_min, rescale_max)
                .exp()
            )
            coords *= rescale_hw

        # angles per axis: [N, 3, rot_pair_count]
        # periods is [rot_pair_count]; do 2*pi * coord / period
        periods = self.periods.to(device=device, dtype=dtype)  # ensure fp32
        angles = (
            2.0 * math.pi * coords[:, :, None] / periods[None, None, :]
        )  # [N, 3, half]

        angles = torch.remainder(angles, 2.0 * math.pi)

        sin = torch.sin(angles).to(dtype=dtype)
        cos = torch.cos(angles).to(dtype=dtype)

        return sin, cos
