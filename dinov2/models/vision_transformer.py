# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from einops import repeat

from dinov2.layers import (
    Mlp,
    PatchEmbed3D,
    SwiGLUFFN,
    Block,
    RopePositionEmbedding3D,
)


logger = logging.getLogger("dinov2")


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        rope_base=100.0,
        rope_rescale_coords=None,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        num_register_tokens=0,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.rope_embed = RopePositionEmbedding3D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=rope_base,
            rescale_coords=rope_rescale_coords,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFN
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError(
                f"FFN layer '{ffn_layer}' is not implemented. Use 'mlp', 'swiglu', or 'identity'."
            )

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        # trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, d, w, h = x.shape
        x, patch_dims = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        cls_tokens = repeat(self.cls_token, "1 1 c -> b 1 c", b=B)

        if self.register_tokens is not None:
            register_tokens = repeat(self.register_tokens, "1 r c -> b r c", b=B)
            x = torch.cat((cls_tokens, register_tokens, x), dim=1)

        else:
            x = torch.cat((cls_tokens, x), dim=1)

        return x, patch_dims

    def forward(self, x, masks=None):
        x, (d, h, w) = self.prepare_tokens_with_masks(x, masks)

        sin, cos = self.rope_embed(D=d, H=h, W=w)
        num_special = 1 + self.num_register_tokens

        for blk in self.blocks:
            x = blk(x, sin=sin, cos=cos, num_special=num_special)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_model(cfg) -> Tuple[DinoVisionTransformer, DinoVisionTransformer]:
    args = cfg.student

    vit_kwargs = dict(
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        patch_size=args.patch_size,
        init_values=args.layerscale,
        rope_base=args.rope_base,
        rope_rescale_coords=args.rope_rescale_coords,
        ffn_layer=args.ffn_layer,
        qkv_bias=args.qkv_bias,
        proj_bias=args.proj_bias,
        ffn_bias=args.ffn_bias,
        num_register_tokens=args.num_register_tokens,
    )

    teacher = DinoVisionTransformer(**vit_kwargs)
    student = DinoVisionTransformer(
        **vit_kwargs,
        drop_path_rate=args.drop_path_rate,
        drop_path_uniform=args.drop_path_uniform,
    )
    return student, teacher
