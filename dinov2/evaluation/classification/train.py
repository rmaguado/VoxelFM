import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from dataclasses import dataclass
from omegaconf import OmegaConf
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from dinov2.evaluation.utils import *
from .dataset import *
from .model import *


@dataclass
class EpochMetrics:
    loss: float
    rocauc: float
    rocauc_ci: Tuple[float, float]
    f1: float


def load_split(csv_path):
    df = pd.read_csv(csv_path, dtype={"series_uid": str})
    return df["series_uid"].tolist()


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["series_uid"], df["label"]))


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
    if cfg.training.input_mode == "embeddings":
        model = build_head(cfg)
    elif cfg.training.input_mode == "volumes":
        from dinov2.inference import build_model

        backbone_config = OmegaConf.load(cfg.model.backbone_config)
        backbone = build_model(cfg.backbone_ckpt, backbone_config, device)
        adapter = DinoV2Adapter(backbone)
        model = LoRABackboneWithHead(adapter, cfg)
    else:
        raise ValueError(f"Unknown input_mode: {cfg.training.input_mode}")

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
        padded, mask, labels = [b.to(device, non_blocking=True) for b in batch]
        logits = model(padded, mask)
    else:
        embeddings, labels = [b.to(device, non_blocking=True) for b in batch]
        logits = model(embeddings).flatten()

    labels = labels.float().view_as(logits)
    loss = loss_fn(logits, labels)
    return loss, logits, labels


def run_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    padding,
    optimizer=None,
    grad_accum_steps=1,
    desc="Epoch",
    scheduler=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    labels_all, preds_all = [], []

    if is_train:
        optimizer.zero_grad()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for step, batch in enumerate(pbar):
            loss, logits, labels = forward_batch(model, batch, loss_fn, device, padding)

            if is_train:
                loss.backward()
                if (step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

            total_loss += loss.item()
            labels_all.append(labels.detach().cpu())
            preds_all.append(logits.detach().cpu())

        step = len(dataloader) - 1
        if is_train and (step + 1) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    labels = torch.cat(labels_all).numpy()
    probs = torch.sigmoid(torch.cat(preds_all)).cpu().numpy()

    rocauc, rocauc_ci = roc_auc_ci_hanley(labels, probs)
    f1 = compute_f1(labels, probs)
    avg_loss = total_loss / len(dataloader)

    return EpochMetrics(avg_loss, rocauc, rocauc_ci, f1)


def test(model, loss_fn, dataloader, device, padding):
    metrics = run_epoch(
        model,
        dataloader,
        loss_fn,
        device,
        padding,
        optimizer=None,
        desc="Test",
    )

    return {
        "loss": metrics.loss,
        "rocauc": metrics.rocauc,
        "rocauc_ci": metrics.rocauc_ci,
        "f1": metrics.f1,
    }


def run_classification(cfg):
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
        labels = [labels_map[u] for u in uids]

        if cfg.training.input_mode == "embeddings":
            return EmbeddingDataset(
                uids=uids,
                labels=labels,
                embeddings_path=cfg.data.embeddings_path,
                feature_key=cfg.data.feature_key,
                pooling=cfg.data.pooling,
                noise_p=cfg.data.noise_p,
                noise_sigma=cfg.data.noise_sigma,
            )

        elif cfg.training.input_mode == "volumes":
            return VolumeDataset(
                uids=uids,
                labels=labels,
                volumes_path=cfg.data.volumes_path,
                fmean=cfg.data.fmean,
                fstd=cfg.data.fstd,
            )

        else:
            raise ValueError(f"`input_mode` not recognized: {cfg.training.input_mode}")

    train_ds = make_dataset(os.path.join(cfg.data.splits_path, "train.csv"))
    val_ds = make_dataset(os.path.join(cfg.data.splits_path, "valid.csv"))
    test_ds = make_dataset(os.path.join(cfg.data.splits_path, "test.csv"))

    collate_fn = collate_fn_pad if cfg.training.padding else None
    train_sampler = StratifiedBatchSampler(train_ds.labels, cfg.training.batch_size)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
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

    epoch_len = len(train_loader) // cfg.training.grad_accum_steps

    model = build_training_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = get_cosine_scheduler_with_warmup(optimizer, cfg.scheduler, epoch_len)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_val_rocauc = 0.0
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
        )

        if history_path is not None:
            epoch_log = {
                "epoch": epoch + 1,
                "train": {
                    "loss": train_metrics.loss,
                    "rocauc": train_metrics.rocauc,
                    "rocauc_ci": train_metrics.rocauc_ci,
                    "f1": train_metrics.f1,
                },
                "val": {
                    "loss": val_metrics.loss,
                    "rocauc": val_metrics.rocauc,
                    "rocauc_ci": val_metrics.rocauc_ci,
                    "f1": val_metrics.f1,
                },
            }
            with open(history_path, "a") as f:
                f.write(json.dumps(epoch_log) + "\n")

        if val_metrics.rocauc > best_val_rocauc:
            best_val_rocauc = val_metrics.rocauc
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        tqdm.write(
            f"Epoch {epoch+1}/{cfg.scheduler.num_epochs} | "
            f"Val ROC-AUC: {val_metrics.rocauc:.4f} | "
            f"Val F1: {val_metrics.f1:.4f} | "
            f"Best: {best_val_rocauc:.4f}"
        )

    checkpoint_path = os.path.join(exp_dir, "model.pt")
    torch.save(best_state_dict, checkpoint_path)

    model.load_state_dict(best_state_dict)
    test_metrics = test(model, loss_fn, test_loader, device, cfg.training.padding)

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Final Test ROC-AUC: {test_metrics['rocauc']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run embedding classification training."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the yaml configuration file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")

    cfg = OmegaConf.load(args.config)
    run_classification(cfg)
