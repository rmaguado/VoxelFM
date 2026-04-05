import os
import json
import math
import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm

from dinov2.evaluation.utils import *
from .dataset import SegmentationEmbeddingDataset
from .model import PatchToVoxelDecoder


def dice_per_class(logits, target, num_classes, ignore_index=0, eps=1e-6):
    """
    logits: (B, C, Z, Y, X)
    target: (B, Z, Y, X)
    returns: dict {class_id: dice}
    """
    pred = logits.argmax(dim=1)
    dices = {}
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        dices[cls] = (
            float("nan") if denom == 0 else ((2 * inter + eps) / (denom + eps)).item()
        )
    return dices


def class_voxel_counts(target, num_classes, ignore_index=0):
    """
    target: (B, Z, Y, X)
    returns: dict {class_id: voxel_count}
    """
    return {
        cls: (target == cls).sum().item()
        for cls in range(num_classes)
        if cls != ignore_index
    }


class MetricAccumulator:
    def __init__(self):
        self._scan_dices: list[dict[int, float]] = []
        self._scan_weights: list[dict[int, float]] = []

    def update(self, dice: dict[int, float], weights: dict[int, float]):
        """Register one batch's worth of per-class Dice and voxel counts."""
        self._scan_dices.append(dice)
        self._scan_weights.append(weights)

    @staticmethod
    def _per_class(dices):
        """Unweighted mean Dice per class, ignoring NaNs."""
        sums = defaultdict(float)
        counts = defaultdict(int)
        for d in dices:
            for cls, v in d.items():
                if not math.isnan(v):
                    sums[cls] += v
                    counts[cls] += 1
        return {cls: sums[cls] / counts[cls] for cls in sums if counts[cls] > 0}

    @staticmethod
    def _macro(per_class: dict[int, float]) -> float:
        vals = [v for v in per_class.values() if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    @staticmethod
    def _micro(dices, weights) -> float:
        """Voxel-weighted mean Dice across all classes and scans."""
        num, denom = 0.0, 0.0
        for d, w in zip(dices, weights):
            for cls, v in d.items():
                if not math.isnan(v):
                    wt = w.get(cls, 0.0)
                    num += v * wt
                    denom += wt
        return num / denom if denom > 0 else float("nan")

    def summary(self) -> dict:
        per_class = self._per_class(self._scan_dices)
        return {
            "dice_per_class": per_class,
            "dice_macro": self._macro(per_class),
            "dice_micro": self._micro(self._scan_dices, self._scan_weights),
        }

    def bootstrap_segmentation_ci(
        self, n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 4
    ) -> dict:
        rng = np.random.default_rng(seed)
        n = len(self._scan_dices)
        alpha = (1 - ci) / 2

        boot_macro = np.empty(n_bootstrap)
        boot_micro = np.empty(n_bootstrap)
        boot_per_class = defaultdict(lambda: np.empty(n_bootstrap))

        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            b_dices = [self._scan_dices[j] for j in idx]
            b_weights = [self._scan_weights[j] for j in idx]

            pc = self._per_class(b_dices)
            boot_macro[i] = self._macro(pc)
            boot_micro[i] = self._micro(b_dices, b_weights)
            for cls, v in pc.items():
                boot_per_class[cls][i] = v

        def ci_tuple(arr):
            lo, hi = np.quantile(arr, [alpha, 1 - alpha])
            return (float(lo), float(hi))

        return {
            "dice_macro_ci": ci_tuple(boot_macro),
            "dice_micro_ci": ci_tuple(boot_micro),
            "dice_per_class_ci": {
                cls: ci_tuple(arr) for cls, arr in boot_per_class.items()
            },
        }


def forward_batch(model, batch, loss_fn, device):
    x, y = [b.to(device) for b in batch]
    logits = model(x)
    loss = loss_fn(logits, y)
    return loss, logits, y


def run_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    optimizer=None,
    grad_accum_steps: int = 1,
    desc: str = "Epoch",
    scheduler=None,
    compute_ci: bool = False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    metric_acc = MetricAccumulator()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for step, batch in enumerate(pbar):
            loss, logits, targets = forward_batch(model, batch, loss_fn, device)

            if is_train:
                (loss / grad_accum_steps).backward()
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            total_loss += loss.item()

            dice = dice_per_class(logits, targets, num_classes=model.num_classes)
            counts = class_voxel_counts(targets, num_classes=model.num_classes)
            metric_acc.update(dice, counts)

    result = {
        "loss": total_loss / len(dataloader),
        **metric_acc.summary(),
    }
    if compute_ci:
        result.update(metric_acc.bootstrap_segmentation_ci())

    return result


def run_segmentation(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    exp_dir = os.path.join(cfg.output_dir, f"{timestamp}_{cfg.name}")
    os.makedirs(exp_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(exp_dir, "config.yaml"))

    def load_split(csv_path):
        df = pd.read_csv(csv_path, dtype={"series_uid": str})
        return df["series_uid"].tolist()

    train_uids = load_split(os.path.join(cfg.data.splits_path, "train.csv"))
    val_uids = load_split(os.path.join(cfg.data.splits_path, "valid.csv"))
    test_uids = load_split(os.path.join(cfg.data.splits_path, "test.csv"))

    train_ds = SegmentationEmbeddingDataset(
        uids=train_uids,
        embeddings_path=cfg.data.embeddings_path,
        masks_path=cfg.data.masks_path,
        feature_key=cfg.data.feature_key,
        noise_p=cfg.data.noise_p,
        noise_sigma=cfg.data.noise_sigma,
    )

    val_ds = SegmentationEmbeddingDataset(
        uids=val_uids,
        embeddings_path=cfg.data.embeddings_path,
        masks_path=cfg.data.masks_path,
        feature_key=cfg.data.feature_key,
    )

    test_ds = SegmentationEmbeddingDataset(
        uids=test_uids,
        embeddings_path=cfg.data.embeddings_path,
        masks_path=cfg.data.masks_path,
        feature_key=cfg.data.feature_key,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 4),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 4),
    )

    model = PatchToVoxelDecoder(
        **cfg.model,
    ).to(device)

    epoch_len = len(train_loader) // cfg.training.grad_accum_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.scheduler.max_lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = get_cosine_scheduler_with_warmup(optimizer, cfg.scheduler, epoch_len)

    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_dice = 0.0
    best_state_dict = dict()

    history_path = os.path.join(exp_dir, "history.jsonl")

    for epoch in tqdm(range(cfg.scheduler.num_epochs), desc="Segmentation Training"):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            grad_accum_steps=cfg.training.grad_accum_steps,
            desc=f"Train Epoch {epoch+1}",
            scheduler=scheduler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            optimizer=None,
            desc=f"Val Epoch {epoch+1}",
        )

        epoch_log = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        }

        with open(history_path, "a") as f:
            f.write(json.dumps(epoch_log) + "\n")

        if val_metrics["dice_macro"] > best_val_dice:
            best_val_dice = val_metrics["dice_macro"]
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        tqdm.write(
            f"Epoch {epoch+1} | Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Dice: {val_metrics['dice_macro']:.4f} | Best Dice: {best_val_dice:.4f}"
        )

    torch.save(best_state_dict, os.path.join(exp_dir, "best_model.pt"))

    model.load_state_dict(best_state_dict)
    test_metrics = run_epoch(
        model,
        test_loader,
        loss_fn,
        device,
        optimizer=None,
        desc="Test",
        compute_ci=True,
    )
    with open(os.path.join(exp_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Final Test Dice (macro): {test_metrics['dice_macro']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run 3D Segmentation Training with Patch Embeddings"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = OmegaConf.load(args.config)
    run_segmentation(cfg)
