import os
import math
import torch
import logging

import dinov2.distributed as distributed


logger = logging.getLogger("dinov2")


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def update_schedules(optimizer, schedulers, iteration):
    lr = schedulers["lr"][iteration]
    wd = schedulers["wd"][iteration]
    momentum = schedulers["momentum"][iteration]
    teacher_temp = schedulers["teacher_temp"][iteration]
    last_layer_lr = schedulers["last_layer_lr"][iteration]
    apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

    return momentum, teacher_temp


def apply_gradient_operations(cfg, model, optimizer, accum_steps):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.div_(accum_steps)

    if cfg.optim.clip_grad:
        all_params = []
        for v in model.student.values():
            module = v
            if hasattr(v, "module"):
                module = v.module
            all_params.extend(module.parameters())

        torch.nn.utils.clip_grad_norm_(all_params, cfg.optim.clip_grad)

    optimizer.step()


def log_training_step(metric_logger, loss_dict, schedulers, iteration, step_duration):
    if distributed.get_global_size() > 1:
        for v in loss_dict.values():
            torch.distributed.all_reduce(v)
    loss_dict_reduced = {
        k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()
    }

    if math.isnan(sum(loss_dict_reduced.values())):
        logger.error(f"NaN detected in reduced loss at iteration {iteration}")
        logger.info(f"Reduced loss dict: {loss_dict_reduced}")
        raise AssertionError

    losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    metric_logger.update(step_duration=step_duration)
    metric_logger.update(lr=schedulers["lr"][iteration])
    metric_logger.update(wd=schedulers["wd"][iteration])
    metric_logger.update(mom=schedulers["momentum"][iteration])
    metric_logger.update(last_layer_lr=schedulers["last_layer_lr"][iteration])
    metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)


def do_test(cfg, model, iteration):
    torch.cuda.synchronize()

    new_state_dict = {k: v.cpu() for k, v in model.teacher.state_dict().items()}

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)

        torch.cuda.empty_cache()

        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)
