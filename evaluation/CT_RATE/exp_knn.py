import os
import json
import torch
from tqdm import tqdm
import numpy as np
import polars as pl
from einops import rearrange
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def get_embedding(series_uid, embeddings_path, select_feature, pooling):
    features_mmap = torch.load(
        os.path.join(embeddings_path, f"{series_uid}.pth"),
        mmap=True,
    )

    if select_feature == "cls":
        features = features_mmap["cls"]
    elif select_feature == "patch":
        features = rearrange(features_mmap["patch"], "z y x d -> (z y x) d")
    elif select_feature == "none":
        features = rearrange(features_mmap, "z y x d -> (z y x) d")
    else:
        raise ValueError(f"Unknown select_feature: {select_feature}")

    if pooling == "avg":
        features = rearrange(features, "... d -> (...) d")
        features = torch.mean(features, dim=0, keepdim=True)
    elif pooling == "none":
        pass
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    if torch.isnan(features).any():
        print(f"NaN found in {series_uid} at {embeddings_path}")
        return torch.zeros_like(features)

    return features.clone().contiguous()


def launch_experiment_knn(cfg, exp_dir: str):
    print("Loading labels...")
    train_df = pl.read_csv(os.path.join(cfg.labels_path, "train_predicted_labels.csv"))
    test_df = pl.read_csv(os.path.join(cfg.labels_path, "valid_predicted_labels.csv"))

    train_df = train_df.with_columns(
        pl.col("VolumeName")
        .str.split("_")
        .list.slice(0, 2)
        .list.join("_")
        .alias("patient_id")
    )
    train_ids = pl.read_csv(os.path.join(cfg.splits_path, "train.csv"))
    train_df = train_ids.filter(
        pl.col("patient_id").is_in(train_ids["patient_id"].implode())
    )  # remove validation set ids

    def clean_df(df):
        return df.with_columns(
            pl.col("VolumeName")
            .str.replace_all(".nii.gz", "", literal=True)
            .alias("series_uid")
        ).select(["series_uid", cfg.label])

    train_df, test_df = clean_df(train_df), clean_df(test_df)

    train_embeddings_path = os.path.join(cfg.embeddings_path, "train")
    test_embeddings_path = os.path.join(cfg.embeddings_path, "valid")

    train_series_uids = [
        x.replace(".pth", "")
        for x in os.listdir(train_embeddings_path)
        if x.endswith(".pth")
    ]
    test_series_uids = [
        x.replace(".pth", "")
        for x in os.listdir(test_embeddings_path)
        if x.endswith(".pth")
    ]

    train_df = train_df.filter(pl.col("series_uid").is_in(train_series_uids))
    test_df = test_df.filter(pl.col("series_uid").is_in(test_series_uids))

    print(
        f"Train samples with embeddings: {len(train_df)}, Valid samples: {len(test_df)}"
    )

    print("Loading training embeddings...")
    X_train, y_train = [], []
    for row in tqdm(train_df.iter_rows(named=True), total=len(train_df)):
        emb = get_embedding(
            row["series_uid"],
            train_embeddings_path,
            cfg.select_feature,
            cfg.pooling,
        )
        X_train.append(emb.squeeze(0).cpu().numpy())
        y_train.append(row[cfg.label])
    X_train = np.stack(X_train)
    y_train = np.array(y_train)

    print("Loading validation embeddings...")
    X_valid, y_valid = [], []
    for row in tqdm(test_df.iter_rows(named=True), total=len(test_df)):
        emb = get_embedding(
            row["series_uid"],
            test_embeddings_path,
            cfg.select_feature,
            cfg.pooling,
        )
        X_valid.append(emb.squeeze(0).cpu().numpy())
        y_valid.append(row[cfg.label])
    X_valid = np.stack(X_valid)
    y_valid = np.array(y_valid)

    print(f"Embeddings loaded: train {X_train.shape}, valid {X_valid.shape}")

    k = getattr(cfg, "k", 5)
    print(f"Running kNN with k={k} using cosine similarity...")

    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_train)
    dists, idxs = nbrs.kneighbors(X_valid)

    y_pred_scores = np.array([y_train[i].mean() for i in idxs])
    y_pred_binary = (y_pred_scores >= 0.5).astype(int)

    rocauc = roc_auc_score(y_valid, y_pred_scores)
    precision = precision_score(y_valid, y_pred_binary, zero_division=0)
    recall = recall_score(y_valid, y_pred_binary, zero_division=0)
    f1 = f1_score(y_valid, y_pred_binary, zero_division=0)

    print(
        f"Validation ROC-AUC: {rocauc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )

    summary = {
        "rocauc": float(rocauc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "k": k,
        "select_feature": cfg.select_feature,
        "pooling": cfg.pooling,
        "experiment_dir": exp_dir,
    }

    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    with open(os.path.join(exp_dir, "metrics", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {os.path.join(exp_dir, 'metrics', 'summary.json')}")
