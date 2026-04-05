import os
import json
import numpy as np
import pandas as pd
from sklearn.externals.array_api_compat.numpy import float16
from tqdm import tqdm
from typing import Dict
from dataclasses import dataclass
from omegaconf import OmegaConf
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from lifelines.utils import concordance_index

import torch
from torch.utils.data import DataLoader

from dinov2.evaluation.utils import *
from .dataset import *
from .model import *


class CoxPHLoss(torch.nn.Module):
    def forward(self, risks, times, events):
        order = torch.argsort(times, descending=True)
        times = times[order]
        risks = risks[order]
        events = events[order]

        log_cumsum = torch.logcumsumexp(risks, dim=0)
        loss = -(risks - log_cumsum) * events
        return loss.sum() / (events.sum() + 1e-8)


@dataclass
class EpochMetrics:
    loss: float
    cindex: float
    rocauc: float
    rocauc_ci: List[float]
    accuracy: float
    f1: float


def binary_survival_metrics(times, events, risks, horizon) -> Dict[str, float]:
    """
    risks: higher = worse survival
    """
    y, valid = binary_survival_labels(times, events, horizon)

    if y.size == 0:
        return {"auc": np.nan, "accuracy": np.nan}

    scores = -risks[valid]
    preds = scores > np.median(scores)

    rocauc, rocauc_ci = roc_auc_ci_hanley(y, scores)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    return {
        "rocauc": rocauc,
        "rocauc_ci": rocauc_ci,
        "accuracy": float(acc),
        "f1": float(f1),
    }


def binary_survival_labels(times, events, horizon):
    """
    Returns:
        y: binary labels (1 = survived beyond horizon)
        mask: boolean mask of valid samples
    """
    times = np.asarray(times)
    events = np.asarray(events)

    valid = (events == 1) | (times >= horizon)

    y = (times >= horizon).astype(np.int32)
    return y[valid], valid


def load_split(csv_path):
    df = pd.read_csv(csv_path, dtype={"series_uid": str})
    return df["series_uid"].tolist()


def load_labels(csv_path):
    df = pd.read_csv(csv_path)

    series_uids = df["series_uid"].to_list()
    times = df["time"].to_list()
    events = df["event"].to_list()

    return {
        series_uid: (time, event)
        for series_uid, time, event in zip(series_uids, times, events)
    }


def load_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    return cfg


def prepare_output_dir(base_output_path, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_output_path, f"{timestamp}_{experiment_name}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "figures"), exist_ok=True)
    return exp_dir


def build_training_model(cfg, device):
    model = build_head(cfg)
    return model.to(device)


def build_optimizer(cfg, model):
    if cfg.optimizer.name == "adamw":
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.scheduler.max_lr,
            weight_decay=cfg.training.weight_decay,
        )

    elif cfg.optimizer.name == "sgd":
        return torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.scheduler.max_lr,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum,
        )

    else:
        raise ValueError(f"Optimizer not regonized: {cfg.optimizer.name}")


def forward_batch(model, batch, loss_fn, device, padding):
    if padding:
        padded, mask, times, events = [b.to(device, non_blocking=True) for b in batch]
        risks = model(padded, mask).flatten()
    else:
        embeddings, times, events = [b.to(device, non_blocking=True) for b in batch]
        risks = model(embeddings).flatten()

    loss = loss_fn(risks, times, events)
    return loss, risks, times, events


def run_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    padding,
    optimizer=None,
    grad_accum_steps=1,
    desc="Epoch",
    horizon=0.0,
    scheduler=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    times_all, events_all, risks_all = [], [], []

    if is_train:
        optimizer.zero_grad()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for step, batch in enumerate(pbar):

            loss, risks, times, events = forward_batch(
                model, batch, loss_fn, device, padding
            )

            if is_train:
                loss.backward()
                if (step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

            total_loss += loss.item()
            times_all.append(times.detach().cpu())
            events_all.append(events.detach().cpu())
            risks_all.append(risks.detach().cpu())

        step = len(dataloader) - 1
        if is_train and (step + 1) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    times = torch.cat(times_all).numpy()
    events = torch.cat(events_all).numpy()
    risks = torch.cat(risks_all).numpy()

    avg_loss = total_loss / len(dataloader)
    cindex = concordance_index(times, -risks, events)

    bin_metrics = binary_survival_metrics(times, events, risks, horizon)
    rocauc = bin_metrics["rocauc"]
    rocauc_ci = bin_metrics["rocauc_ci"]
    accuracy = bin_metrics["accuracy"]
    f1 = bin_metrics["f1"]

    return EpochMetrics(avg_loss, cindex, rocauc, rocauc_ci, accuracy, f1)


def test(model, loss_fn, dataloader, device, padding, horizon):
    metrics = run_epoch(
        model,
        dataloader,
        loss_fn,
        device,
        padding,
        optimizer=None,
        desc="Test",
        horizon=horizon,
    )

    return {
        "loss": metrics.loss,
        "cindex": metrics.cindex,
        "rocauc": metrics.rocauc,
        "rocauc_ci": metrics.rocauc_ci,
        "accuracy": metrics.accuracy,
        "f1": metrics.f1,
    }


def run_survival(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    exp_name = f"{timestamp}_{cfg.name}"
    exp_dir = os.path.join(cfg.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    history_path = os.path.join(exp_dir, "history.jsonl")

    OmegaConf.save(cfg, os.path.join(exp_dir, "config.yaml"))

    labels_map = load_labels(cfg.data.labels_path)

    def make_dataset(splits_path):
        uids = load_split(splits_path)

        exist_uids = []
        for u in uids:
            file_path = os.path.join(cfg.data.embeddings_path, f"{u}.pth")
            if os.path.exists(file_path):
                exist_uids.append(u)

        missing_embed = len(uids) - len(exist_uids)
        if missing_embed > 0:
            print(
                f"Dropped {missing_embed} series IDs from {os.path.basename(splits_path)} because .pth files were missing."
            )

        uids = exist_uids

        uid_with_label = [x for x in list(labels_map.keys()) if x in uids]
        missing_label = len(uids) - len(uid_with_label)

        if missing_label > 0:
            print(
                f"Dropped {missing_label} series IDs from {os.path.basename(splits_path)} because labels were missing."
            )

        uids = uid_with_label
        times = [labels_map[u][0] for u in uids]
        events = [labels_map[u][1] for u in uids]

        return EmbeddingDataset(
            uids=uids,
            times=times,
            events=events,
            embeddings_path=cfg.data.embeddings_path,
            feature_key=cfg.data.feature_key,
            pooling=cfg.data.pooling,
            noise_p=cfg.data.noise_p,
            noise_sigma=cfg.data.noise_sigma,
        )

    train_ds = make_dataset(os.path.join(cfg.data.splits_path, "train.csv"))
    val_ds = make_dataset(os.path.join(cfg.data.splits_path, "valid.csv"))
    test_ds = make_dataset(os.path.join(cfg.data.splits_path, "test.csv"))

    collate_fn = collate_fn_pad if cfg.training.padding else None
    # train_sampler = StratifiedBatchSampler(train_ds.labels, cfg.training.batch_size)

    train_loader = DataLoader(
        train_ds,
        # batch_sampler=train_sampler,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        num_workers=int(cfg.num_workers * 0.7),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(cfg.num_workers * 0.15),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers - int(cfg.num_workers * 0.85),
    )

    model = build_training_model(cfg, device)
    optimizer = build_optimizer(cfg, model)

    epoch_len = len(train_loader) // cfg.training.grad_accum_steps
    scheduler = get_cosine_scheduler_with_warmup(optimizer, cfg.scheduler, epoch_len)

    horizon = cfg.evaluation.binary_horizon

    loss_fn = CoxPHLoss()
    best_val_cindex = 0.0
    best_state_dict = dict()

    for epoch in tqdm(range(cfg.scheduler.num_epochs), desc="Training"):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            cfg.training.padding,
            optimizer=optimizer,
            grad_accum_steps=cfg.training.grad_accum_steps,
            desc=f"Train Epoch {epoch+1}",
            horizon=horizon,
            scheduler=scheduler,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            cfg.training.padding,
            optimizer=None,
            desc=f"Val Epoch {epoch+1}",
            horizon=horizon,
        )

        if history_path is not None:
            epoch_log = {
                "epoch": epoch + 1,
                "train": {
                    "loss": train_metrics.loss,
                    "cindex": train_metrics.cindex,
                    "rocauc": train_metrics.rocauc,
                    "accuracy": train_metrics.accuracy,
                    "f1": train_metrics.f1,
                },
                "val": {
                    "loss": val_metrics.loss,
                    "cindex": val_metrics.cindex,
                    "rocauc": val_metrics.rocauc,
                    "accuracy": val_metrics.accuracy,
                    "f1": val_metrics.f1,
                },
            }
            with open(history_path, "a") as f:
                f.write(json.dumps(epoch_log) + "\n")

        if val_metrics.cindex > best_val_cindex:
            best_val_cindex = val_metrics.cindex
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        tqdm.write(
            f"Epoch {epoch+1}/{cfg.scheduler.num_epochs} | "
            f"Val Loss: {val_metrics.loss:.4f} | "
            f"Val C-index: {val_metrics.cindex:.4f} | "
            f"Val F1: {val_metrics.f1:.4f} | "
            f"Best Val C-index: {best_val_cindex:.4f}"
        )

    checkpoint_path = os.path.join(exp_dir, "model.pt")
    torch.save(best_state_dict, checkpoint_path)

    model.load_state_dict(best_state_dict)
    test_metrics = test(
        model, loss_fn, test_loader, device, cfg.training.padding, horizon
    )

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Final Test C-Index: {test_metrics['cindex']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run embedding survival training.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the yaml configuration file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")

    cfg = OmegaConf.load(args.config)
    run_survival(cfg)
