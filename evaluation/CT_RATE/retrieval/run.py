import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from einops import rearrange


N_NEGATIVES = 99


@dataclass
class LabelMetrics:
    label: str
    n_positives: int
    num_trials: int
    recall: Dict[int, float] = field(default_factory=dict)
    recall_ci_lower: Dict[int, float] = field(default_factory=dict)
    recall_ci_upper: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        out: Dict = {
            "label": self.label,
            "n_positives": self.n_positives,
            "num_trials": self.num_trials,
        }
        for k in sorted(self.recall):
            out[f"Recall@{k}"] = self.recall[k]
            out[f"Recall@{k}_ci_lower"] = self.recall_ci_lower[k]
            out[f"Recall@{k}_ci_upper"] = self.recall_ci_upper[k]
        return out


def load_embedding(
    uid: str, embeddings_path: str, feature_key: str, pooling: str
) -> torch.Tensor:
    path = os.path.join(embeddings_path, f"{uid}.pth")
    data = torch.load(path, mmap=True)
    x = data[feature_key] if feature_key != "none" else data
    if x.ndim > 1:
        x = rearrange(x, "... d -> (...) d")
        if pooling == "avg":
            x = x.mean(dim=0, keepdim=True)
    return x.contiguous()


def load_all_embeddings(
    uids: List[str],
    embeddings_path: str,
    feature_key: str,
    pooling: str,
    device: torch.device,
) -> torch.Tensor:
    tensors = [
        load_embedding(uid, embeddings_path, feature_key, pooling)
        for uid in tqdm(uids, desc="Loading embeddings")
    ]
    return torch.cat(tensors, dim=0).to(device)


def cosine_similarity_batched(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    batch_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    query_norm = query_embeddings / (query_embeddings.norm(dim=-1, keepdim=True) + eps)
    gallery_norm = gallery_embeddings / (
        gallery_embeddings.norm(dim=-1, keepdim=True) + eps
    )
    rows = []
    for i in range(0, query_norm.shape[0], batch_size):
        rows.append(query_norm[i : i + batch_size] @ gallery_norm.T)
    return torch.cat(rows, dim=0)


def bootstrap_ci(
    scores: np.ndarray,
    n_resamples: int = 1000,
    ci_confidence: float = 0.95,
) -> Tuple[float, float]:
    n = len(scores)
    if n < 2:
        m = float(scores.mean()) if n == 1 else 0.0
        return m, m
    rng = np.random.default_rng(seed=42)
    boot_means = np.array(
        [rng.choice(scores, size=n, replace=True).mean() for _ in range(n_resamples)]
    )
    alpha = 1.0 - ci_confidence
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def load_labels(labels_path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(labels_path, dtype={"series_uid": str})
    sample = df["labels"].dropna().head(20).str.cat(sep=" ")
    result: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        uid = row["series_uid"]
        if not isinstance(row["labels"], str):
            continue
        result[uid] = [l.strip() for l in row["labels"].split(";") if l.strip()]
    return result


def get_valid_uids(embeddings_path: str, labels_map: Dict[str, List[str]]) -> List[str]:
    embed_uids = {
        x.replace(".pth", "") for x in os.listdir(embeddings_path) if x.endswith(".pth")
    }
    valid_uids = sorted(embed_uids & set(labels_map.keys()))
    print(f"Found {len(embed_uids)} embeddings, {len(labels_map)} labelled samples")
    print(f"Valid samples with both: {len(valid_uids)}")
    return valid_uids


def get_all_labels(labels_map: Dict[str, List[str]]) -> List[str]:
    from collections import Counter

    counts = Counter(l for ls in labels_map.values() for l in ls)
    # Return sorted by frequency descending
    return [label for label, _ in counts.most_common()]


def evaluate_label(
    label: str,
    uids: List[str],
    labels_map: Dict[str, List[str]],
    sim_matrix: torch.Tensor,
    recall_ks: List[int],
    n_resamples: int,
    ci_confidence: float,
    rng: np.random.Generator,
) -> LabelMetrics:
    pos_indices = np.array(
        [i for i, uid in enumerate(uids) if label in labels_map.get(uid, [])],
        dtype=np.intp,
    )
    neg_indices = np.array(
        [i for i, uid in enumerate(uids) if label not in labels_map.get(uid, [])],
        dtype=np.intp,
    )

    hit_lists: Dict[int, List[float]] = {k: [] for k in recall_ks}

    for query_idx in pos_indices:
        other_pos = pos_indices[pos_indices != query_idx]
        query_sims = sim_matrix[query_idx].cpu().numpy()

        for pos_idx in other_pos:
            neg_sample = rng.choice(neg_indices, size=N_NEGATIVES, replace=False)
            pool = np.concatenate([[pos_idx], neg_sample])
            pool_sims = query_sims[pool]
            ranked_order = np.argsort(-pool_sims, kind="stable")
            pos_rank = int(np.where(ranked_order == 0)[0][0]) + 1  # 1-indexed

            for k in recall_ks:
                hit_lists[k].append(1.0 if pos_rank <= k else 0.0)

    num_trials = len(hit_lists[recall_ks[0]])
    metrics = LabelMetrics(
        label=label,
        n_positives=len(pos_indices),
        num_trials=num_trials,
    )
    for k in recall_ks:
        arr = np.array(hit_lists[k], dtype=np.float64)
        metrics.recall[k] = float(arr.mean())
        metrics.recall_ci_lower[k], metrics.recall_ci_upper[k] = bootstrap_ci(
            arr, n_resamples, ci_confidence
        )

    return metrics


def print_summary(
    all_metrics: List[LabelMetrics],
    ci_confidence: float,
    recall_ks: List[int],
) -> None:
    ci_pct = int(ci_confidence * 100)
    pool_size = 1 + N_NEGATIVES
    rand_recall = {k: k / pool_size for k in recall_ks}

    # Sort by Recall@K of the first (smallest) K
    primary_k = recall_ks[0]
    sorted_metrics = sorted(
        all_metrics, key=lambda m: m.recall.get(primary_k, 0), reverse=True
    )

    header_ks = "".join(f"  {'R@'+str(k):>10}" for k in recall_ks)
    print(f"\n{'='*80}")
    print(
        f"  Per-label retrieval sweep  "
        f"(1 positive / {N_NEGATIVES} negatives, pool={pool_size})"
    )
    print(f"{'='*80}")
    print(f"  {'Label':<42} {'n':>5}{header_ks}  {'trials':>8}")
    print(f"  {'-'*76}")

    for m in sorted_metrics:
        scores = "".join(f"  {m.recall.get(k, 0):>10.4f}" for k in recall_ks)
        print(f"  {m.label:<42} {m.n_positives:>5}{scores}  {m.num_trials:>8,}")

    # Random baseline row
    rand_scores = "".join(f"  {rand_recall[k]:>10.4f}" for k in recall_ks)
    print(f"  {'-'*76}")
    print(f"  {'random baseline':<42} {'':>5}{rand_scores}")
    print(f"{'='*80}\n")


def compute_macro_average(
    all_metrics: List[LabelMetrics],
    recall_ks: List[int],
    n_resamples: int,
    ci_confidence: float,
) -> Dict:
    """
    Macro-average Recall@K across all labels, with bootstrap CI.

    Bootstrap is over the 18 per-label means (one draw per label per resample),
    which captures uncertainty in the class-average rather than within-class
    trial variance.
    """
    # Shape: (n_labels, n_ks)
    per_label_scores = np.array(
        [[m.recall.get(k, 0.0) for k in recall_ks] for m in all_metrics],
        dtype=np.float64,
    )  # (18, len(recall_ks))

    n_labels = len(all_metrics)
    rng = np.random.default_rng(seed=42)

    out: Dict = {"n_labels": n_labels}

    for ki, k in enumerate(recall_ks):
        scores = per_label_scores[:, ki]
        macro = float(scores.mean())

        boot_means = np.array(
            [
                rng.choice(scores, size=n_labels, replace=True).mean()
                for _ in range(n_resamples)
            ]
        )
        alpha = 1.0 - ci_confidence
        ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

        out[f"Recall@{k}"] = macro
        out[f"Recall@{k}_ci"] = [ci_lo, ci_hi]

    return out


def run_label_sweep(config: DictConfig) -> List[Dict]:
    labels_map = load_labels(config.data.labels_path)
    uids = get_valid_uids(config.data.embeddings_path, labels_map)

    if not uids:
        raise ValueError("No valid samples found with both embeddings and labels!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading embeddings...")
    embeddings = load_all_embeddings(
        uids,
        config.data.embeddings_path,
        config.data.feature_key,
        config.data.pooling,
        device,
    )

    print("Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity_batched(
        embeddings, embeddings, config.data.batch_size
    )

    hn_cfg = getattr(config, "hard_negative", OmegaConf.create({}))
    rng_seed = int(getattr(hn_cfg, "seed", 4))
    recall_ks = list(getattr(hn_cfg, "recall_ks", [1, 5, 10]))
    n_resamples = int(getattr(config, "bootstrap_n_resamples", 1000))
    ci_confidence = float(getattr(config, "ci_confidence", 0.95))

    rng = np.random.default_rng(seed=rng_seed)
    labels = get_all_labels(labels_map)

    print(f"\nRunning sweep over {len(labels)} labels...\n")
    all_metrics: List[LabelMetrics] = []

    for label in labels:
        print(f"  {label}...")
        m = evaluate_label(
            label,
            uids,
            labels_map,
            sim_matrix,
            recall_ks,
            n_resamples,
            ci_confidence,
            rng,
        )
        all_metrics.append(m)

    print_summary(all_metrics, ci_confidence, recall_ks)

    macro = compute_macro_average(all_metrics, recall_ks, n_resamples, ci_confidence)

    os.makedirs(config.output_dir, exist_ok=True)

    results = [m.to_dict() for m in all_metrics]
    per_label_path = os.path.join(config.output_dir, "results_per_label.json")
    with open(per_label_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPer-label results saved to : {per_label_path}")

    macro_path = os.path.join(config.output_dir, "results_macro.json")
    with open(macro_path, "w") as f:
        json.dump(macro, f, indent=2)
    print(f"Macro-average saved to     : {macro_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-label retrieval sweep: 1 positive vs 99 negatives per trial."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (same schema as run.py).",
    )
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    cfg = OmegaConf.load(args.config)
    run_label_sweep(cfg)
