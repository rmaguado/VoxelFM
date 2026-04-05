# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import logging


logger = logging.getLogger("dinov2")


class iBOTPatchLoss(nn.Module):
    center: torch.Tensor
    async_batch_center: torch.Tensor

    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.register_buffer(
            "async_batch_center", torch.zeros(1, 1, patch_out_dim), persistent=False
        )
        self.updated = True
        self.reduce_handle = None
        self.token_count = 0

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        # loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.async_batch_center += torch.sum(
            teacher_patch_tokens.mean(1), dim=0, keepdim=True
        )
        self.token_count += len(teacher_patch_tokens)

    @torch.no_grad()
    def flush_center_update(self):
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(
                self.async_batch_center,
                async_op=True,
            )
        self.updated = False

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated:
            return

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        mom = self.center_momentum if torch.any(self.center) else 0.0

        if self.reduce_handle is not None:
            self.reduce_handle.wait()
        _t = self.async_batch_center / (self.token_count * world_size)

        self.center = self.center * mom + _t * (1 - mom)

        self.async_batch_center.zero_()
        self.token_count = 0
        self.updated = True
