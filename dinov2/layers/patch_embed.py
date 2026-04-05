# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor, Size
import torch.nn as nn
from einops import rearrange


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size

        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor):
        x = self.proj(x.unsqueeze(1))  # B C D H W
        patch_dims = x.shape[2:]

        x = rearrange(x, "b c d h w -> b (d h w) c")
        x = self.norm(x)

        return x, patch_dims
