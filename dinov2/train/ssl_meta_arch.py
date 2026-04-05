# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn
from typing import Dict
from torch.nn.parallel import DistributedDataParallel as DDP

import dinov2.distributed as dist
from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model, DinoVisionTransformer
from dinov2.layers import DINOHead
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups


logger = logging.getLogger("dinov2")


class DinoModel(nn.ModuleDict):
    backbone: DinoVisionTransformer
    dino_head: DINOHead
    ibot_head: DINOHead


class SSLMetaArch(nn.Module):
    student: DinoModel
    teacher: DinoModel

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        student_model_dict: Dict[str, nn.Module] = dict()
        teacher_model_dict: Dict[str, nn.Module] = dict()

        student_backbone, teacher_backbone = build_model(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = cfg.student.embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        if self.do_dino:
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=self.embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                self.koleo_loss = KoLeoLoss()

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()  # type: ignore
            teacher_model_dict["dino_head"] = dino_head()  # type: ignore

        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert (
                max(cfg.ibot.mask_ratio_min_max) > 0
            ), "please provide a positive mask ratio tuple for ibot"
            assert (
                cfg.ibot.mask_sample_probability > 0
            ), "please provide a positive mask probability for ibot"
            self.ibot_out_dim = (
                cfg.ibot.head_n_prototypes
                if self.ibot_separate_head
                else cfg.dino.head_n_prototypes
            )
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                ibot_head = partial(
                    DINOHead,
                    in_dim=self.embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()

        self.student = DinoModel(student_model_dict)
        self.teacher = DinoModel(teacher_model_dict)

        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_teacher_output(
        self,
        global_crops,
        mask_indices_list,
        teacher_temp,
        upperbound,
        n_global_crops,
        n_masked_patches,
    ):
        teacher_backbone_output = self.teacher.backbone(global_crops)
        teacher_cls_tokens = teacher_backbone_output["x_norm_clstoken"]
        teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops)

        # reverse order: ensure global crop A matches crop B for DINO loss
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))

        ibot_teacher_patch_tokens = teacher_backbone_output["x_norm_patchtokens"]
        _dim = ibot_teacher_patch_tokens.shape[-1]
        n_cls_tokens = teacher_cls_tokens.shape[0]

        if self.do_ibot and not self.ibot_separate_head:
            buffer = ibot_teacher_patch_tokens.new_zeros(
                upperbound + n_cls_tokens, _dim
            )
            buffer[:n_cls_tokens].copy_(teacher_cls_tokens)
            torch.index_select(
                ibot_teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer[n_cls_tokens : n_cls_tokens + n_masked_patches],
            )
            tokens_after_head = self.teacher.dino_head(buffer)
            teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
            masked_teacher_patch_tokens_after_head = tokens_after_head[
                n_cls_tokens : n_cls_tokens + n_masked_patches
            ]
        elif self.do_ibot and self.ibot_separate_head:
            buffer = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
            torch.index_select(
                ibot_teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer[:n_masked_patches],
            )
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer)[
                :n_masked_patches
            ]
        else:
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            masked_teacher_ibot_softmaxed_centered = None

        teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
            teacher_cls_tokens_after_head, teacher_temp=teacher_temp
        ).view(n_global_crops, -1, *teacher_cls_tokens_after_head.shape[1:])
        self.dino_loss.update_center(teacher_cls_tokens_after_head)

        if self.do_ibot:
            masked_teacher_patch_tokens_after_head = (
                masked_teacher_patch_tokens_after_head.unsqueeze(0)  # type: ignore
            )
            masked_teacher_ibot_softmaxed_centered = (
                self.ibot_patch_loss.softmax_center_teacher(
                    masked_teacher_patch_tokens_after_head[:, :n_masked_patches],
                    teacher_temp=teacher_temp,
                ).squeeze(0)
            )
            self.ibot_patch_loss.update_center(
                masked_teacher_patch_tokens_after_head[:n_masked_patches]
            )
        else:
            masked_teacher_ibot_softmaxed_centered = None

        return (
            teacher_dino_softmaxed_centered_list,
            masked_teacher_ibot_softmaxed_centered,
        )

    def get_student_output(
        self,
        global_crops,
        local_crops,
        masks,
        mask_indices_list,
        upperbound,
    ):
        n_masked_patches = mask_indices_list.shape[0]

        # Backbone forward
        student_global_out = self.student.backbone(global_crops, masks=masks)
        student_local_out = self.student.backbone(local_crops, masks=None)

        inputs_for_head = []

        # Local CLS tokens
        student_local_cls_tokens = student_local_out["x_norm_clstoken"]
        inputs_for_head.append(student_local_cls_tokens.unsqueeze(0))

        # Global CLS tokens
        student_global_cls_tokens = student_global_out["x_norm_clstoken"]
        inputs_for_head.append(student_global_cls_tokens.unsqueeze(0))

        # Masked patch tokens (IBOT)
        if self.do_ibot:
            _dim = student_global_cls_tokens.shape[-1]
            ibot_patch_tokens = student_global_out["x_norm_patchtokens"]
            buffer_patches = ibot_patch_tokens.new_zeros(upperbound, _dim)
            buffer_patches[:n_masked_patches].copy_(
                torch.index_select(
                    ibot_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )
            if not self.ibot_separate_head:
                inputs_for_head.append(buffer_patches.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(
                    buffer_patches
                )[:n_masked_patches]
        else:
            student_global_masked_patch_tokens_after_head = None

        flat_inputs = [x.squeeze(0) for x in inputs_for_head]
        sizes = [x.shape[0] for x in flat_inputs]

        cat_inputs = torch.cat(flat_inputs, dim=0)
        cat_outputs = self.student.dino_head(cat_inputs)
        outputs_list = list(torch.split_with_sizes(cat_outputs, sizes, dim=0))
        outputs_list = [x.unsqueeze(0) for x in outputs_list]

        # Unpack
        student_local_cls_after_head = outputs_list.pop(0).squeeze(0)
        student_global_cls_after_head = outputs_list.pop(0).squeeze(0)

        if self.do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(
                0
            )[:n_masked_patches]

        return dict(
            local_cls_before=student_local_cls_tokens,
            global_cls_before=student_global_cls_tokens,
            local_cls_after=student_local_cls_after_head,
            global_cls_after=student_global_cls_after_head,
            masked_patch_after=student_global_masked_patch_tokens_after_head,  # type: ignore
        )

    def forward(self, images, teacher_temp):
        n_global_crops = 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)
        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]

        teacher_dino_out, masked_teacher_ibot_out = self.get_teacher_output(
            global_crops,
            mask_indices_list,
            teacher_temp,
            upperbound,
            n_global_crops,
            n_masked_patches,
        )

        student_outputs = self.get_student_output(
            global_crops,
            local_crops,
            masks,
            mask_indices_list,
            upperbound,
        )

        loss_dict = {}
        loss_accumulator = 0

        n_local_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_loss_terms = (n_global_crops - 1) * n_global_crops
        loss_scales = 2
        ibot_loss_scale = 1.0 / n_global_crops

        # Local DINO loss
        if n_local_crops > 0:
            dino_local_loss = self.dino_loss(
                student_output_list=student_outputs["local_cls_after"].chunk(
                    n_local_crops
                ),
                teacher_out_softmaxed_centered_list=teacher_dino_out,
            ) / (n_global_loss_terms + n_local_loss_terms)
            loss_dict["dino_local_crops_loss"] = dino_local_loss
            loss_accumulator += self.dino_loss_weight * dino_local_loss

        # Global DINO loss
        if self.do_dino:
            dino_global_loss = (
                self.dino_loss(
                    student_output_list=[student_outputs["global_cls_after"]],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_out.flatten(0, 1)
                    ],
                )
                * loss_scales
                / (n_global_loss_terms + n_local_loss_terms)
            )
            loss_dict["dino_global_crops_loss"] = dino_global_loss
            loss_accumulator += self.dino_loss_weight * dino_global_loss

        # KoLeo loss
        if self.do_koleo:
            student_cls_tokens = student_outputs[
                "global_cls_before"
            ]  # expecting shape [2 * batch, dim]

            koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
            )
            loss_accumulator += koleo_loss
            loss_dict["koleo_loss"] = koleo_loss / loss_scales

        # IBOT loss
        if self.do_ibot:
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_outputs["masked_patch_after"],
                    masked_teacher_ibot_out,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )
            loss_dict["ibot_loss"] = ibot_patch_loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        return loss_dict, loss_accumulator

    def broadcast_teacher(self):
        for param in self.teacher.parameters():
            torch.distributed.broadcast(param.data, src=0)

    def backprop_loss(self, loss):
        loss.backward()
        torch.cuda.synchronize()

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                student_module = self.student[k]
                teacher_module = self.teacher[k]

                if hasattr(student_module, "module"):
                    student_module: nn.Module = student_module.module  # type: ignore

                student_param_list += list(student_module.parameters())
                teacher_param_list += list(teacher_module.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

        self.dino_loss.flush_center_update()
        self.ibot_patch_loss.flush_center_update()
        self.broadcast_teacher()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def get_maybe_fused_params_for_submodel(self, m):
        # For DDP, we need to access the underlying module
        if hasattr(m, "module"):
            m = m.module
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True  # type: ignore
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        for k, v in self.student.items():
            self.student[k] = DDP(
                self.student[k],
                device_ids=[dist.get_local_rank()],
                output_device=dist.get_local_rank(),
                find_unused_parameters=False,
                broadcast_buffers=True,
            )
        self.broadcast_teacher()
