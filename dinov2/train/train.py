# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import time
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from dinov2.logging import MetricLogger
from dinov2.utils.config import setup

from dinov2.checkpointer import (
    save_checkpoint,
    keep_last_n_checkpoints,
    keep_checkpoint_copy,
)
from dinov2.train.utils import (
    update_schedules,
    apply_gradient_operations,
    log_training_step,
    do_test,
)
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.parser import get_args_parser
from dinov2.train.setup import (
    setup_training_components,
    setup_dataloader_multi_resolution,
)
import dinov2.distributed as dist

torch.backends.cuda.matmul.fp32_precision = "tf32"
logger = logging.getLogger("dinov2")


def should_reset_grad(cfg, grad_accum_counter):
    return grad_accum_counter % cfg.train.grad_accum_steps == 0


def should_apply_training_step(cfg, grad_accum_counter, accum_steps):
    return (grad_accum_counter + 1) % accum_steps == 0


def should_eval_model(cfg, iteration):
    return (
        cfg.evaluation.eval_period_iterations > 0
        and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
    )


def log_param_signature(model):
    rank = dist.get_local_rank()

    teacher_checksum = 0.0
    for p in model.teacher.parameters():
        teacher_checksum += float(p.detach().float().sum().cpu().item())
    logger.debug(f"RANK {rank} TEACHER_CHECKSUM {teacher_checksum}")

    student_checksum = 0.0
    for m in model.student.values():
        mod = m.module if hasattr(m, "module") else m
        for p in mod.parameters():
            student_checksum += float(p.detach().float().sum().cpu().item())
    logger.debug(f"RANK {rank} STUDENT_CHECKSUM {student_checksum}")


def train(cfg, model, resume=False):
    model.train()
    model.broadcast_teacher()
    inputs_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[cfg.compute_precision]

    (
        optimizer,
        schedulers,
        start_iter,
        max_iter,
    ) = setup_training_components(cfg, model, resume)

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(
        output_file=os.path.join(cfg.train.output_dir, "training_metrics.jsonl"),
    )

    data_loader = setup_dataloader_multi_resolution(
        cfg, inputs_dtype, start_iter=iteration
    )
    metric_logger.set_dataloader(data_loader)

    grad_accum_counter = 0
    iteration = start_iter

    accum_steps = cfg.train.grad_accum_steps
    ckpt_dir = cfg.train.output_dir

    teacher_temp = None
    mom = None
    step_duration = 0

    for data in metric_logger.log_every(
        cfg.train.print_freq,
        "Training",
        max_iter,
        start_iter,
        accum_steps,
    ):
        if iteration > max_iter:
            return

        if should_reset_grad(cfg, grad_accum_counter):
            mom, teacher_temp = update_schedules(optimizer, schedulers, iteration)
            optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=inputs_dtype, enabled=True):
            loss_dict, loss_accumulator = model.forward(data, teacher_temp=teacher_temp)

        model.backprop_loss(loss_accumulator)
        step_duration += time.time() - t0

        if should_apply_training_step(cfg, grad_accum_counter, accum_steps):
            apply_gradient_operations(cfg, model, optimizer, accum_steps)
            model.update_teacher(mom)

            log_training_step(
                metric_logger, loss_dict, schedulers, iteration, step_duration
            )

            if (iteration + 1) % cfg.checkpointing.period == 0:
                torch.cuda.synchronize()
                ckpt_path = os.path.join(ckpt_dir, f"{iteration + 1:06}")

                save_checkpoint(
                    ckpt_path,
                    iteration=iteration + 1,
                    model=model,
                    optimizer=optimizer,
                    overwrite=True,
                )
                if dist.is_main_process():
                    if getattr(cfg.checkpointing, "keep_every") is not None:
                        if (iteration + 1) % cfg.checkpointing.keep_every == 0:
                            keep_checkpoint_copy(ckpt_path)
                    keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)

            step_duration = 0
            iteration += 1

            if should_eval_model(cfg, iteration):
                do_test(cfg, model, f"training_{iteration}")
                torch.cuda.synchronize()

        grad_accum_counter += 1

    do_test(cfg, model, f"training_{iteration}")
    logger.info("Finished training.")


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    assert (
        cfg.train.batch_size_per_gpu
        * cfg.train.grad_accum_steps
        * dist.get_global_size()
        == cfg.train.total_batch_size
    ), f"batch_size_per_gpu * grad_accum_steps * num_gpus should equal total_batch_size"

    logger.debug("Model:\n{}".format(model))

    train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    if os.environ.get("PYTHONPATH") is not None and not os.path.exists("dinov2"):
        os.chdir(os.environ["PYTHONPATH"])

    args = get_args_parser(add_help=True).parse_args()
    try:
        main(args)
    finally:
        torch.distributed.destroy_process_group()
