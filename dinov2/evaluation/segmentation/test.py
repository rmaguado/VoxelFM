import os
import json
import math
import argparse
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

from .train import MetricAccumulator, dice_per_class, class_voxel_counts


def forward_batch(model, batch, loss_fn, device):
    x, y = [b.to(device) for b in batch]
    logits = model(x)
    loss = loss_fn(logits, y)
    return loss, logits, y


def run_inference(exp_dir: str, output_filename: str = "predictions.csv"):
    config_path = os.path.join(exp_dir, "config.yaml")
    checkpoint_path = os.path.join(exp_dir, "best_model.pt")
    output_path = os.path.join(exp_dir, output_filename)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if os.path.exists(output_path):
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            "Remove it manually or choose a different --output_filename."
        )

    cfg = OmegaConf.load(config_path)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint_path)

    def load_split(csv_path):
        df = pd.read_csv(csv_path, dtype={"series_uid": str})
        return df["series_uid"].tolist()

    test_uids = load_split(os.path.join(cfg.data.splits_path, "test.csv"))
    test_ds = SegmentationEmbeddingDataset(
        uids=test_uids,
        embeddings_path=cfg.data.embeddings_path,
        masks_path=cfg.data.masks_path,
        feature_key=cfg.data.feature_key,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = PatchToVoxelDecoder(
        **cfg.model,
    ).to(device)
    model.load_state_dict(state_dict)

    loss_fn = torch.nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    metric_acc = MetricAccumulator()
    with torch.no_grad():
        pbar = tqdm(test_loader, leave=False)
        for step, batch in enumerate(pbar):
            loss, logits, targets = forward_batch(model, batch, loss_fn, device)

            total_loss += loss.item()

            dice = dice_per_class(logits, targets, num_classes=model.num_classes)
            counts = class_voxel_counts(targets, num_classes=model.num_classes)
            metric_acc.update(dice, counts)

    test_metrics = {
        "loss": total_loss / len(test_loader),
        **metric_acc.summary(),
        **metric_acc.bootstrap_segmentation_ci(),
    }
    with open(os.path.join(exp_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Final Test Dice (macro): {test_metrics['dice_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on the test set.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to the experiment folder containing config.yaml and model.pt.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        raise NotADirectoryError(f"Experiment directory not found: {args.exp_dir}")

    run_inference(args.exp_dir)
