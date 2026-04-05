# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


def _rotate_block(x_block: Tensor, sin_block: Tensor, cos_block: Tensor) -> Tensor:
    # x_block: [B, nHeads, N_patches, block_dim] where block_dim = head_dim // 3
    # sin_block / cos_block: [N_patches, half_per_axis]
    B, nH, Np, block_dim = x_block.shape
    half = block_dim // 2  # guaranteed even by init assert
    # reshape to pair-wise: [..., half, 2]
    xr = x_block.reshape(B, nH, Np, half, 2)
    x_even = xr[..., 0]  # [B, nH, Np, half]
    x_odd = xr[..., 1]  # [B, nH, Np, half]

    # expand cos/sin to broadcast: [1,1,Np,half]
    cos_b = cos_block[None, None, :, :].to(x_block.dtype)
    sin_b = sin_block[None, None, :, :].to(x_block.dtype)

    rot_even = x_even * cos_b - x_odd * sin_b
    rot_odd = x_even * sin_b + x_odd * cos_b

    out = torch.stack([rot_even, rot_odd], dim=-1).reshape(B, nH, Np, block_dim)
    return out


def apply_rope_to_patches(
    q: Tensor, k: Tensor, sin: Tensor, cos: Tensor, num_special: int = 1
):
    """
    q,k: [B, nHeads, N_total, head_dim]
    sin,cos: [N_patches, 3, half_per_axis]
    num_special: number of leading tokens to skip (cls + reg)
    """
    B, nH, N_total, head_dim = q.shape
    N_patches = sin.shape[0]
    # split special / patch tokens
    q_special, q_patches = (
        q[..., :num_special, :],
        q[..., num_special:, :],
    )  # q_patches: [B,nH,N_patches,head_dim]
    k_special, k_patches = k[..., :num_special, :], k[..., num_special:, :]

    # split each head into 3 axis-blocks (z, y, x)
    block_dim = head_dim // 3
    qz, qy, qx = q_patches.split(block_dim, dim=-1)
    kz, ky, kx = k_patches.split(block_dim, dim=-1)

    # sin/cos per axis: sin[:, axis_index, :] -> [N_patches, half_per_axis]
    sin_z = sin[:, 0, :]
    sin_y = sin[:, 1, :]
    sin_x = sin[:, 2, :]
    cos_z = cos[:, 0, :]
    cos_y = cos[:, 1, :]
    cos_x = cos[:, 2, :]

    # rotate each block
    qz = _rotate_block(qz, sin_z, cos_z)
    qy = _rotate_block(qy, sin_y, cos_y)
    qx = _rotate_block(qx, sin_x, cos_x)

    kz = _rotate_block(kz, sin_z, cos_z)
    ky = _rotate_block(ky, sin_y, cos_y)
    kx = _rotate_block(kx, sin_x, cos_x)

    q_patches = torch.cat([qz, qy, qx], dim=-1)
    k_patches = torch.cat([kz, ky, kx], dim=-1)

    q = torch.cat([q_special, q_patches], dim=2)
    k = torch.cat([k_special, k_patches], dim=2)
    return q, k


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, sin: Tensor, cos: Tensor, num_special: int) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = apply_rope_to_patches(q, k, sin, cos, num_special=num_special)

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
