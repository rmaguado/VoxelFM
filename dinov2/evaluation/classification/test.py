import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
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


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["series_uid"].astype(str), df["label"]))


def build_inference_model(cfg, device):
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

    if cfg.training.input_mode == "embeddings":
        test_ds = EmbeddingDataset(
            uids=test_uids,
            labels=labels,
            embeddings_path=cfg.data.embeddings_path,
            feature_key=cfg.data.feature_key,
            pooling=cfg.data.pooling,
            noise_p=0.0,
            noise_sigma=0.0,
        )
    elif cfg.training.input_mode == "volumes":
        test_ds = VolumeDataset(
            uids=test_uids,
            labels=labels,
            volumes_path=cfg.data.volumes_path,
            fmean=cfg.data.fmean,
            fstd=cfg.data.fstd,
        )
    else:
        raise ValueError(f"`input_mode` not recognized: {cfg.training.input_mode}")

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

    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if cfg.training.padding:
                padded, mask, batch_labels = [
                    b.to(device, non_blocking=True) for b in batch
                ]
                logits = model(padded, mask)
            else:
                embeddings, batch_labels = [
                    b.to(device, non_blocking=True) for b in batch
                ]
                logits = model(embeddings).flatten()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch_labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    results_df = pd.DataFrame(
        {
            "series_uid": test_uids,
            "label": all_labels.astype(int),
            "prob": all_probs,
        }
    )
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions ({len(results_df)} samples) → {output_path}")

    if np.unique(all_labels).size >= 2:
        rocauc = roc_auc_score(all_labels, all_probs)
        print(f"Test ROC-AUC: {rocauc:.4f}")

    return results_df


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
