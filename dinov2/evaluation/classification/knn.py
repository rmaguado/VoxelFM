import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from dataclasses import dataclass
from omegaconf import OmegaConf
from datetime import datetime
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier

from torch.utils.data import DataLoader

from dinov2.evaluation.utils import *
from .dataset import EmbeddingDataset


@dataclass
class KNNMetrics:
    rocauc: float
    rocauc_ci: Tuple[float, float]
    f1: float
    accuracy: float


def load_split(csv_path):
    df = pd.read_csv(csv_path, dtype={"series_uid": str})
    return df["series_uid"].tolist()


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["series_uid"], df["label"]))


def load_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    return cfg


def compute_accuracy(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return float(np.mean(y_true == y_pred))


def extract_embeddings_from_dataset(dataset, batch_size=32):
    """Extract all embeddings and labels from a dataset."""
    embeddings_list = []
    labels_list = []

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=None
    )

    for batch_emb, batch_labels in tqdm(
        dataloader, desc="Extracting embeddings", leave=False
    ):
        if batch_emb.ndim > 2:
            batch_emb = batch_emb.mean(dim=1)

        embeddings_list.append(batch_emb.cpu().numpy())
        labels_list.append(batch_labels.cpu().numpy())

    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)

    return embeddings, labels


def evaluate_knn(knn_model, X, y_true):
    """Evaluate KNN model and return metrics."""
    y_probs = knn_model.predict_proba(X)[:, 1]  # Probability of class 1

    rocauc, rocauc_ci = roc_auc_ci_hanley(y_true, y_probs)
    f1 = compute_f1(y_true, y_probs)
    accuracy = compute_accuracy(y_true, y_probs)

    return KNNMetrics(rocauc, rocauc_ci, f1, accuracy)


def grid_search_k(X_train, y_train, X_val, y_val, k_values=None):
    """Perform grid search to find the best k value."""
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101]

    best_k = k_values[0]
    best_rocauc = 0.0
    results = []

    print("\nGrid search for optimal k:")
    for k in tqdm(k_values, desc="Testing k values"):
        if k > len(X_train):
            continue

        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        )
        knn.fit(X_train, y_train)

        val_metrics = evaluate_knn(knn, X_val, y_val)

        results.append(
            {
                "k": k,
                "rocauc": val_metrics.rocauc,
                "f1": val_metrics.f1,
                "accuracy": val_metrics.accuracy,
            }
        )

        print(
            f"k={k:3d} | Val ROC-AUC: {val_metrics.rocauc:.4f} | "
            f"F1: {val_metrics.f1:.4f} | Acc: {val_metrics.accuracy:.4f}"
        )

        if val_metrics.rocauc > best_rocauc:
            best_rocauc = val_metrics.rocauc
            best_k = k

    print(f"\nBest k: {best_k} with ROC-AUC: {best_rocauc:.4f}")

    return best_k, results


def run_knn_classification(cfg):
    set_seed(cfg.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    exp_name = f"{timestamp}_{cfg.name}_knn"
    exp_dir = os.path.join(cfg.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

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
                f"Dropped {missing_embed} series IDs from {os.path.basename(splits_path)} "
                f"because .pth files were missing."
            )

        uids = exist_uids

        uid_with_label = [x for x in list(labels_map.keys()) if x in uids]
        missing_label = len(uids) - len(uid_with_label)

        if missing_label > 0:
            print(
                f"Dropped {missing_label} series IDs from {os.path.basename(splits_path)} "
                f"because labels were missing."
            )

        uids = uid_with_label
        labels = [labels_map[u] for u in uids]

        return EmbeddingDataset(
            uids=uids,
            labels=labels,
            embeddings_path=cfg.data.embeddings_path,
            feature_key=cfg.data.feature_key,
            pooling=cfg.data.pooling,
            noise_p=0.0,
            noise_sigma=0.0,
        )

    print("Loading datasets...")
    train_ds = make_dataset(os.path.join(cfg.data.splits_path, "train.csv"))
    val_ds = make_dataset(os.path.join(cfg.data.splits_path, "valid.csv"))
    test_ds = make_dataset(os.path.join(cfg.data.splits_path, "test.csv"))

    print("\nExtracting embeddings from datasets...")
    X_train, y_train = extract_embeddings_from_dataset(train_ds)
    X_val, y_val = extract_embeddings_from_dataset(val_ds)
    X_test, y_test = extract_embeddings_from_dataset(test_ds)

    print(f"\nTrain: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Val:   {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")

    if hasattr(cfg, "knn") and hasattr(cfg.knn, "k"):
        best_k = cfg.knn.k
        print(f"\nUsing specified k={best_k}")
        grid_results = None
    else:
        k_values = (
            cfg.knn.k_values
            if hasattr(cfg, "knn") and hasattr(cfg.knn, "k_values")
            else None
        )
        best_k, grid_results = grid_search_k(X_train, y_train, X_val, y_val, k_values)

        if grid_results is not None:
            with open(os.path.join(exp_dir, "grid_search.json"), "w") as f:
                json.dump(grid_results, f, indent=4)

    print(f"\nTraining final KNN model with k={best_k}...")
    knn_model = KNeighborsClassifier(
        n_neighbors=best_k, weights="distance", metric="euclidean", n_jobs=-1
    )
    knn_model.fit(X_train, y_train)

    print("\nEvaluating on validation set...")
    val_metrics = evaluate_knn(knn_model, X_val, y_val)
    print(
        f"Val ROC-AUC: {val_metrics.rocauc:.4f} "
        f"({val_metrics.rocauc_ci[0]:.4f}, {val_metrics.rocauc_ci[1]:.4f})"
    )
    print(f"Val F1: {val_metrics.f1:.4f}")
    print(f"Val Accuracy: {val_metrics.accuracy:.4f}")

    print("\nEvaluating on test set...")
    test_metrics = evaluate_knn(knn_model, X_test, y_test)
    print(
        f"Test ROC-AUC: {test_metrics.rocauc:.4f} "
        f"({test_metrics.rocauc_ci[0]:.4f}, {test_metrics.rocauc_ci[1]:.4f})"
    )
    print(f"Test F1: {test_metrics.f1:.4f}")
    print(f"Test Accuracy: {test_metrics.accuracy:.4f}")

    results = {
        "best_k": best_k,
        "rocauc": test_metrics.rocauc,
        "rocauc_ci": test_metrics.rocauc_ci,
        "f1": test_metrics.f1,
        "accuracy": test_metrics.accuracy,
    }

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {exp_dir}")
    print(f"Final Test ROC-AUC: {test_metrics.rocauc:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run KNN classification on embeddings."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the yaml configuration file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        exit(1)

    cfg = OmegaConf.load(args.config)
    run_knn_classification(cfg)
