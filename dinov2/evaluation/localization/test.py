import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
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


def build_inference_model(cfg, device):
    model = PositionalAttentionRegressor3D(
        embed_dim=cfg.model.embed_dim,
        **cfg.model.params,
    )
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

    coords_map = load_coords(cfg.data.labels_path)
    test_uids_raw = load_split(os.path.join(cfg.data.splits_path, "test.csv"))

    test_uids = []
    test_coords = []
    for u in test_uids_raw:
        file_path = os.path.join(cfg.data.embeddings_path, f"{u}.pth")
        if os.path.exists(file_path) and u in coords_map:
            test_uids.append(u)
            test_coords.append(coords_map[u])

    dropped = len(test_uids_raw) - len(test_uids)
    if dropped > 0:
        print(
            f"Dropped {dropped} series IDs because .pth files or coords were missing."
        )

    test_ds = EmbeddingLocalizationDataset(
        uids=test_uids,
        coords=test_coords,
        embeddings_path=cfg.data.embeddings_path,
        feature_key=cfg.data.feature_key,
        noise_p=0.0,
        noise_sigma=0.0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = build_inference_model(cfg, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            x, targets = [b.to(device, non_blocking=True) for b in batch]
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)  # (N, 3) — pred_z, pred_y, pred_x
    all_targets = np.concatenate(all_targets)  # (N, 3) — z, y, x

    errors = np.linalg.norm(all_targets - all_preds, axis=1)

    results_df = pd.DataFrame(
        {
            "series_uid": test_uids,
            "z": all_targets[:, 0],
            "y": all_targets[:, 1],
            "x": all_targets[:, 2],
            "pred_z": all_preds[:, 0],
            "pred_y": all_preds[:, 1],
            "pred_x": all_preds[:, 2],
            "error": errors,
        }
    )
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions ({len(results_df)} samples) → {output_path}")
    print(f"Test Mean Localization Error: {errors.mean()}")

    ci_lower, ci_upper, se = bootstrap_ci(errors, n_bootstrap=10000, ci=0.95)
    print(f"Bootstrap 95% CI: [{ci_lower}, {ci_upper}]")
    print(f"Standard Error (Bootstrap): {se:.4f}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run localization inference on the test set."
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
