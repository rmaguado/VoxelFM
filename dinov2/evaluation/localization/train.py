import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dinov2.evaluation.utils import *
from .dataset import EmbeddingLocalizationDataset
from .model import PositionalAttentionRegressor3D


def load_split(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, dtype={"series_uid": str})
    return df["series_uid"].tolist()


def load_coords(csv_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    CSV columns:
        series_uid, rel_z, rel_y, rel_x
    """
    df = pd.read_csv(csv_path, dtype={"series_uid": str})
    return {
        r["series_uid"]: (r["rel_z"], r["rel_y"], r["rel_x"]) for _, r in df.iterrows()
    }


def mean_localization_error(y_true, y_pred):
    """
    Euclidean distance in normalized coordinate space
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.linalg.norm(y_true - y_pred, axis=1).mean()


def make_dataset(cfg, split_csv):
    uids = load_split(split_csv)
    coords_map = load_coords(cfg.data.labels_path)

    valid_uids = []
    valid_coords = []

    for uid in uids:
        path = os.path.join(cfg.data.embeddings_path, f"{uid}.pth")
        if os.path.exists(path) and uid in coords_map:
            valid_uids.append(uid)
            valid_coords.append(coords_map[uid])

    dropped = len(uids) - len(valid_uids)
    if dropped > 0:
        print(
            f"Dropped {dropped} samples from "
            f"{os.path.basename(split_csv)} (missing embeddings or coords)"
        )

    return EmbeddingLocalizationDataset(
        uids=valid_uids,
        coords=valid_coords,
        embeddings_path=cfg.data.embeddings_path,
        feature_key=cfg.data.feature_key,
        noise_p=cfg.data.noise_p,
        noise_sigma=cfg.data.noise_sigma,
    )


def build_model(cfg, device):
    model = PositionalAttentionRegressor3D(
        embed_dim=cfg.model.embed_dim,
        **cfg.model.params,
    )

    return model.to(device)


def build_optimizer(cfg, model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.scheduler.max_lr,
        weight_decay=cfg.training.weight_decay,
    )


def forward_batch(model, batch, loss_fn, device):
    x, targets = [b.to(device) for b in batch]
    preds = model(x)
    loss = loss_fn(preds, targets)
    return loss, preds, targets


def run_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    optimizer=None,
    grad_accum_steps: int = 1,
    scheduler=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    preds_all, targets_all = [], []

    if is_train:
        optimizer.zero_grad()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(dataloader, leave=False)
        for step, batch in enumerate(pbar):
            loss, preds, targets = forward_batch(model, batch, loss_fn, device)

            if is_train:
                loss.backward()
                if (step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

            total_loss += loss.item()
            preds_all.append(preds.detach().cpu())
            targets_all.append(targets.detach().cpu())

        step = len(dataloader) - 1
        if is_train and (step + 1) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    preds = torch.cat(preds_all).numpy()
    targets = torch.cat(targets_all).numpy()

    mle = mean_localization_error(targets, preds)

    return {
        "loss": total_loss / len(dataloader),
        "mae": float(mle),
    }


def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    num_epochs,
    grad_accum_steps,
    device,
    scheduler,
) -> Tuple[List[Dict], Dict]:
    best_val_error = float("inf")
    best_state = dict()
    history = []

    for epoch in range(num_epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            grad_accum_steps=grad_accum_steps,
            scheduler=scheduler,
        )
        val_metrics = run_epoch(model, val_loader, loss_fn, device)

        history.append(
            {
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if val_metrics["mae"] < best_val_error:
            best_val_error = val_metrics["mae"]
            best_state: Dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    return history, best_state


def run_localization(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    exp_dir = os.path.join(cfg.output_dir, f"{timestamp}_{cfg.name}")
    os.makedirs(exp_dir, exist_ok=True)

    OmegaConf.save(cfg, os.path.join(exp_dir, "config.yaml"))

    train_ds = make_dataset(cfg, os.path.join(cfg.data.splits_path, "train.csv"))
    val_ds = make_dataset(cfg, os.path.join(cfg.data.splits_path, "valid.csv"))
    test_ds = make_dataset(cfg, os.path.join(cfg.data.splits_path, "test.csv"))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=int(cfg.num_workers * 0.7),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=int(cfg.num_workers * 0.15),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers - int(cfg.num_workers * 0.85),
    )

    epoch_len = len(train_loader) // cfg.training.grad_accum_steps

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = get_cosine_scheduler_with_warmup(optimizer, cfg.scheduler, epoch_len)

    loss_fn = torch.nn.SmoothL1Loss()

    history, best_state = train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.scheduler.num_epochs,
        grad_accum_steps=cfg.training.grad_accum_steps,
        device=device,
        scheduler=scheduler,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(exp_dir, "history.csv"), index=False)

    ckpt_path = os.path.join(exp_dir, "model.pt")
    torch.save(best_state, ckpt_path)

    model.load_state_dict(best_state)
    test_metrics = run_epoch(
        model,
        test_loader,
        loss_fn,
        device,
        optimizer=None,
    )

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Final Test Mean Localization Error: " f"{test_metrics['mae']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run localization training with precomputed embeddings."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to localization config YAML",
    )

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    run_localization(cfg)
