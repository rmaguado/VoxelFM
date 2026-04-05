import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dinov2.evaluation.utils import *
from .dataset import *
from .model import *


def load_split(csv_path):
    df = pd.read_csv(csv_path, dtype={"series_uid": str})
    return df["series_uid"].tolist()


def bootstrap_cindex(
    times: np.ndarray,
    risks: np.ndarray,
    events: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the Concordance Index.
    """
    n = len(times)
    boot_indices = np.random.randint(0, n, (n_bootstrap, n))
    boot_stats = []

    for i in range(n_bootstrap):
        idx = boot_indices[i]
        # lifelines C-index: higher risk should mean shorter time,
        # so we pass -risks if risks represent 'hazard'
        c = concordance_index(times[idx], -risks[idx], events[idx])
        boot_stats.append(c)

    boot_stats = np.array(boot_stats)
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)

    return lower, upper


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    series_uids = df["series_uid"].astype(str).to_list()
    times = df["time"].to_list()
    events = df["event"].to_list()
    return {uid: (time, event) for uid, time, event in zip(series_uids, times, events)}


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
    times = [labels_map[u][0] for u in test_uids]
    events = [labels_map[u][1] for u in test_uids]

    test_ds = EmbeddingDataset(
        uids=test_uids,
        times=times,
        events=events,
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

    all_risks, all_times, all_events = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if cfg.training.padding:
                padded, mask, batch_times, batch_events = [
                    b.to(device, non_blocking=True) for b in batch
                ]
                risks = model(padded, mask).flatten()
            else:
                embeddings, batch_times, batch_events = [
                    b.to(device, non_blocking=True) for b in batch
                ]
                risks = model(embeddings).flatten()

            all_risks.append(risks.cpu().numpy())
            all_times.append(batch_times.cpu().numpy())
            all_events.append(batch_events.cpu().numpy())

    all_risks = np.concatenate(all_risks)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)

    three_year_label = np.logical_and(all_events == 1, all_times <= 3).astype(int)

    results_df = pd.DataFrame(
        {
            "series_uid": test_uids,
            "time": all_times,
            "event": all_events.astype(int),
            "pred_risk": all_risks,
            "death_3y": three_year_label,
        }
    )
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions ({len(results_df)} samples) → {output_path}")

    cindex = concordance_index(all_times, -all_risks, all_events)
    c_low, c_high = bootstrap_cindex(all_times, all_risks, all_events)

    print(f"Test C-Index: {cindex}")
    print(f"Test C-Index 95% CI: [{c_low}, {c_high}]")

    rocauc, (rocauc_ci_lower, rocauc_ci_upper) = roc_auc_ci_hanley(
        three_year_label, all_risks
    )
    print(f"Test ROC-AUC (death within 3 years): {rocauc}")
    print(f"Test ROC-AUC 95% CI: [{rocauc_ci_lower}, {rocauc_ci_upper}]")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run survival analysis inference on the test set."
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
