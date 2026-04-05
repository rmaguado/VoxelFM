# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    center: torch.Tensor
    async_batch_center: torch.Tensor

    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer(
            "async_batch_center", torch.zeros(1, out_dim), persistent=False
        )
        self.updated = True
        self.reduce_handle = None
        self.token_count = 0

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.async_batch_center += torch.sum(teacher_output, dim=0, keepdim=True)
        self.token_count += len(teacher_output)

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
