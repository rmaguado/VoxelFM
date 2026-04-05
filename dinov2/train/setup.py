import logging
import torch
from functools import partial
import numpy as np
import copy

from dinov2.checkpointer import find_latest_checkpoint, load_checkpoint
from dinov2.data import (
    collate_data_and_cast,
    MaskingGenerator3D,
    CombinedDataLoader,
    SamplerType,
    make_data_loader,
    make_train_dataset,
)


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)
    )


def linear_warmup_cosine_decay(
    start: float,
    peak: float,
    end: float,
    warmup_iterations: int,
    total_iterations: int,
    cosine_iterations: int | None = None,
) -> np.ndarray:
    linear = np.linspace(start, peak, warmup_iterations, endpoint=False)
    if cosine_iterations is None:
        cosine_iterations = total_iterations - warmup_iterations
    cosine = np.cos(np.linspace(0, np.pi, cosine_iterations))
    cosine = (cosine + 1) / 2
    cosine = (peak - end) * cosine + end
    remaining_iterations = total_iterations - cosine_iterations - warmup_iterations
    assert remaining_iterations >= 0
    constant = np.full((remaining_iterations,), fill_value=end)
    return np.concatenate([linear, cosine, constant])


def build_schedulers(cfg):
    iter_per_epoch = cfg.train.official_epoch_length
    total_iterations = iter_per_epoch * cfg.optim.epochs

    lr = linear_warmup_cosine_decay(
        start=cfg.schedules.lr.start,
        peak=cfg.schedules.lr.peak,
        end=cfg.schedules.lr.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.lr.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.lr.cosine_epochs
            if "cosine_epochs" in cfg.schedules.lr
            else None
        ),
    )
    last_layer_lr = lr.copy()
    last_layer_lr[: int(iter_per_epoch * cfg.schedules.lr.freeze_last_layer_epochs)] = 0
    weight_decay = linear_warmup_cosine_decay(
        start=cfg.schedules.weight_decay.start,
        peak=cfg.schedules.weight_decay.peak,
        end=cfg.schedules.weight_decay.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.weight_decay.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.weight_decay.cosine_epochs
            if "cosine_epochs" in cfg.schedules.weight_decay
            else None
        ),
    )
    momentum = linear_warmup_cosine_decay(
        start=cfg.schedules.momentum.start,
        peak=cfg.schedules.momentum.peak,
        end=cfg.schedules.momentum.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.momentum.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.momentum.cosine_epochs
            if "cosine_epochs" in cfg.schedules.momentum
            else None
        ),
    )
    teacher_temp = linear_warmup_cosine_decay(
        start=cfg.schedules.teacher_temp.start,
        peak=cfg.schedules.teacher_temp.peak,
        end=cfg.schedules.teacher_temp.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.teacher_temp.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in cfg.schedules.teacher_temp
            else None
        ),
    )
    return {
        "lr": lr,
        "wd": weight_decay,
        "momentum": momentum,
        "teacher_temp": teacher_temp,
        "last_layer_lr": last_layer_lr,
    }


def setup_dataloader(cfg, inputs_dtype, start_iter, num_workers):

    image_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size

    input_size = [im_s // patch_size for im_s in image_size]
    n_tokens = input_size[0] * input_size[1] * input_size[2]

    mask_generator = MaskingGenerator3D(
        input_size=tuple(input_size),
        max_num_patches=0.5 * n_tokens,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    dataset, weights = make_train_dataset(cfg)

    batch_size = cfg.train.batch_size_per_gpu
    if weights is not None:
        sampler_type = SamplerType.WEIGHTED_INFINITE
    else:
        sampler_type = SamplerType.INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=cfg.train.seed,
        weights=weights,
        sampler_type=sampler_type,
        drop_last=True,
        start_iter=start_iter,
        collate_fn=collate_fn,
    )

    return data_loader


def setup_dataloader_multi_resolution(
    cfg,
    inputs_dtype,
    start_iter,
):
    global_crops_sizes = (
        [cfg.crops.global_crops_size]
        if isinstance(cfg.crops.global_crops_size[0], int)
        else cfg.crops.global_crops_size
    )
    local_crops_sizes = (
        [cfg.crops.local_crops_size]
        if isinstance(cfg.crops.local_crops_size[0], int)
        else cfg.crops.local_crops_size
    )
    loader_ratios = (
        [cfg.crops.crop_pairs_ratios]
        if type(cfg.crops.crop_pairs_ratios) in [int, float]
        else cfg.crops.crop_pairs_ratios
    )
    assert len(global_crops_sizes) == len(local_crops_sizes) == len(loader_ratios)

    total_workers = cfg.train.num_workers
    loader_workers = [int(r * total_workers) for r in loader_ratios]

    loaders = []
    for increment, (
        global_crops_size_i,
        local_crops_size_i,
        num_workers_i,
    ) in enumerate(zip(global_crops_sizes, local_crops_sizes, loader_workers)):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.crops.global_crops_size = global_crops_size_i
        cfg_i.crops.local_crops_size = local_crops_size_i
        cfg_i.train.seed = cfg.train.seed + increment + 1
        loaders.append(
            setup_dataloader(
                cfg=cfg_i,
                inputs_dtype=inputs_dtype,
                start_iter=start_iter,
                num_workers=num_workers_i,
            )
        )

    if len(loaders) == 1:
        data_loader = loaders[0]
    else:
        data_loader = CombinedDataLoader(
            loaders_with_ratios=zip(loaders, loader_ratios),
            batch_size=cfg.train.batch_size_per_gpu,
            combining_mode=0,
            seed=cfg.train.seed,
            name="MultiResDL",
        )
    return data_loader


def setup_training_components(cfg, model, resume):
    logger = logging.getLogger("dinov2")

    optimizer = build_optimizer(cfg, model.get_params_groups())
    logger.info("Optimizer ready.")
    schedulers = build_schedulers(cfg)
    logger.info("Schedulers ready.")

    start_iter = 0
    if resume and (last_checkpoint_dir := find_latest_checkpoint(cfg.train.output_dir)):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        start_iter = load_checkpoint(
            last_checkpoint_dir,
            model=model,
            optimizer=optimizer,
            strict_loading=True,
        )

    num_epochs = cfg.optim.epochs
    epoch_len = cfg.train.official_epoch_length
    max_iter = epoch_len * num_epochs

    return (
        optimizer,
        schedulers,
        start_iter,
        max_iter,
    )
