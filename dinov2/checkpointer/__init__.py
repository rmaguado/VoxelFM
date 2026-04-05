# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import shutil
import subprocess
from pathlib import Path

import torch
import dinov2.distributed as dist

logger = logging.getLogger(__name__)


def save_checkpoint(
    ckpt_dir: str,  # e.g. output_dir/ckpt/199
    *,
    iteration: int | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    overwrite: bool = True,
    **others,
):
    """Save a model, its optimizer, iteration, and other stateful objects to a single file."""
    rank = dist.get_global_rank()
    _ckpt_dir = Path(ckpt_dir)
    ckpt_file = _ckpt_dir / "checkpoint.pth"

    if rank == 0:
        if ckpt_file.exists() and not overwrite:
            raise RuntimeError(f"Checkpoint already exists: {ckpt_file}")

        _ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {"iteration": iteration, "model": model.state_dict()}
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        state.update(others)

        torch.save(state, ckpt_file)
        logger.info(f"Saved: {ckpt_file}")

    torch.distributed.barrier()


def load_checkpoint(
    ckpt_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict_loading: bool = True,
    **others,
) -> int:
    """Load a model, its optimizer, iteration, and other stateful objects from a single file."""
    rank = dist.get_global_rank()
    ckpt_file = Path(ckpt_dir) / "checkpoint.pth"

    # Ensure all ranks agree if file exists
    exists = [ckpt_file.exists() if rank == 0 else None]
    torch.distributed.broadcast_object_list(exists, src=0)
    if not exists[0]:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model"], strict=strict_loading)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    for k, v in others.items():
        if k in checkpoint:
            v.load_state_dict(checkpoint[k])

    logger.info(f"Loaded: {ckpt_file}")
    return checkpoint.get("iteration", 0)


def find_all_checkpoints(
    ckpt_dir: Path | str, include_keeps: bool = True
) -> list[Path]:
    """
    Find checkpoint directories in ckpt_dir.
    - Normal checkpoints: "123/"
    - Optional keep copies: "123_keep/"
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []

    checkpoints = []
    for p in ckpt_dir.iterdir():
        if not p.is_dir():
            continue

        name = p.name
        if name.isdigit():
            checkpoints.append((int(name), p))
        elif name.endswith("_keep") and include_keeps:
            base = name[:-5]
            if base.isdigit():
                checkpoints.append((int(base), p))

    checkpoints.sort(key=lambda x: x[0])
    return [ckpt for _, ckpt in checkpoints]


def find_latest_checkpoint(ckpt_dir: Path | str) -> Path | None:
    checkpoints = find_all_checkpoints(ckpt_dir)
    return checkpoints[-1] if checkpoints else None


def keep_last_n_checkpoints(ckpt_dir: Path | str, n: int | None):
    """Keep only the last n checkpoints (by integer index)."""
    if n is None:
        return
    checkpoints = find_all_checkpoints(ckpt_dir, include_keeps=False)
    for old_ckpt in checkpoints[:-n]:
        try:
            shutil.rmtree(old_ckpt)
            logger.info(f"Deleted: {old_ckpt}")
        except Exception:
            logger.exception(f"Failed to delete: {old_ckpt}")


def keep_checkpoint_copy(src: Path | str):
    """Copy a checkpoint dir next to itself with a _keep suffix."""
    src = Path(src)
    dst = src.parent / f"{src.name}_keep"
    try:
        subprocess.check_output(
            ["cp", "--recursive", "--link", str(src), str(dst)],
            stderr=subprocess.STDOUT,
        )
        logger.info(f"Copied: {src} -> {dst}")
    except subprocess.CalledProcessError as e:
        pass
