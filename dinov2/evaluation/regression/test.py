import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dinov2.evaluation.utils import *
from .dataset import *
from .model import *


def load_split(csv_path):
    df = pd.read_csv(csv_path, dtype={"series_uid": str})
    return df["series_uid"].tolist()


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["series_uid"].astype(str), df["label"]))


def build_inference_model(cfg, device):
    model = build_head(cfg)
    return model.to(device)


def run_inference(exp_dir: str, output_filename: str = "predictions.csv"):
    config_path = os.path.join(exp_dir, "config.yaml")
    checkpoint_path = os.path.join(exp_dir, "model.pt")
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

    labels_map = load_labels(cfg.data.labels_path)
    test_uids_raw = load_split(os.path.join(cfg.data.splits_path, "test.csv"))

    test_uids = []
    for u in test_uids_raw:
        file_path = os.path.join(cfg.data.embeddings_path, f"{u}.pth")
        if os.path.exists(file_path):
            test_uids.append(u)

    missing_embed = len(test_uids_raw) - len(test_uids)
    if missing_embed > 0:
        print(f"Dropped {missing_embed} series IDs because .pth files were missing.")

    uids_with_label = [u for u in test_uids if u in labels_map]
    missing_label = len(test_uids) - len(uids_with_label)
    if missing_label > 0:
        print(f"Dropped {missing_label} series IDs because labels were missing.")

    test_uids = uids_with_label
    labels = [labels_map[u] for u in test_uids]

    test_ds = EmbeddingDataset(
        uids=test_uids,
        targets=labels,
        embeddings_path=cfg.data.embeddings_path,
        feature_key=cfg.data.feature_key,
        pooling=cfg.data.pooling,
        noise_p=0.0,
        noise_sigma=0.0,
    )

    collate_fn = collate_fn_pad if cfg.training.padding else None
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
    )

    model = build_inference_model(cfg, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if cfg.training.padding:
                padded, mask, batch_labels = [
                    b.to(device, non_blocking=True) for b in batch
                ]
                preds = model(padded, mask).flatten()
            else:
                embeddings, batch_labels = [
                    b.to(device, non_blocking=True) for b in batch
                ]
                preds = model(embeddings).flatten()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    errors = np.abs(all_preds - all_labels)

    results_df = pd.DataFrame(
        {
            "series_uid": test_uids,
            "label": all_labels,
            "prob": all_preds,
            "error": errors,
        }
    )
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions ({len(results_df)} samples) → {output_path}")

    mse = float(np.mean(errors**2))
    mae = float(np.mean(errors))
    print(f"Test MSE: {mse} | Test MAE: {mae}")

    # Bootstrap confidence intervals for MAE
    mae_ci_lower, mae_ci_upper, mae_se = bootstrap_ci(
        errors, n_bootstrap=10000, ci=0.95
    )
    print(f"MAE Bootstrap 95% CI: [{mae_ci_lower}, {mae_ci_upper}]")
    print(f"MAE Standard Error (Bootstrap): {mae_se}")

    # Bootstrap confidence intervals for MSE
    mse_ci_lower, mse_ci_upper, mse_se = bootstrap_ci(
        errors**2, n_bootstrap=10000, ci=0.95
    )
    print(f"MSE Bootstrap 95% CI: [{mse_ci_lower}, {mse_ci_upper}]")
    print(f"MSE Standard Error (Bootstrap): {mse_se}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run regression inference on the test set."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to the experiment folder containing config.yaml and model.pt.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="predictions.csv",
        help="Name of the output CSV file (saved inside exp_dir).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        raise NotADirectoryError(f"Experiment directory not found: {args.exp_dir}")

    run_inference(args.exp_dir, args.output_filename)
