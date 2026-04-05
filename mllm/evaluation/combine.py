import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path


RESULTS_PATHS = [
    "mllm/evaluation/out/dino/generated_abnormalities.csv",
    "mllm/evaluation/out/m3d/generated_abnormalities.csv",
    "mllm/evaluation/out/ctclip/generated_abnormalities.csv",
    "mllm/evaluation/out/merlin/generated_abnormalities.csv",
    "mllm/evaluation/out/radfm/generated_abnormalities.csv",
]

GROUND_TRUTH_PATH = "mllm/evaluation/out/ground_truth_abnormalities.csv"

OUTPUT_DIR = "mllm/evaluation/out/f1_scores"

N_BOOTSTRAP = 1000
CI_ALPHA = 0.95
RANDOM_SEED = 4


def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = CI_ALPHA,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI for the F1 score."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        yt, yp = y_true[indices], y_pred[indices]
        # zero_division=0 avoids warnings when a bootstrap sample has no positives
        scores.append(f1_score(yt, yp, zero_division=0))

    lower_pct = (1 - alpha) / 2 * 100
    upper_pct = (1 + alpha) / 2 * 100
    return float(np.percentile(scores, lower_pct)), float(
        np.percentile(scores, upper_pct)
    )


def load_predictions(csv_path: str) -> pd.DataFrame:
    """Load a prediction CSV, coercing boolean columns to int (0/1)."""
    df = pd.read_csv(csv_path, index_col="series_uid")

    # Convert True/False strings or actual booleans → int
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0})
        elif df[col].dtype == bool:
            df[col] = df[col].astype(int)

    return df


def process_model(csv_path: str, gt_df: pd.DataFrame, output_dir: str) -> None:
    """Compute per-label F1 + CI for one model and write a JSON result file."""
    model_name = Path(csv_path).parent.name  # e.g. "dino"
    print(f"[{model_name}] Loading {csv_path} …")

    pred_df = load_predictions(csv_path)

    # Align on the common series_uids and labels
    common_uids = pred_df.index.intersection(gt_df.index)
    common_labels = pred_df.columns.intersection(gt_df.columns).tolist()

    if len(common_uids) == 0:
        print(f"  [WARN] No overlapping series_uids with ground truth — skipping.")
        return
    if len(common_labels) == 0:
        print(f"  [WARN] No overlapping labels with ground truth — skipping.")
        return

    pred_df = pred_df.loc[common_uids, common_labels]
    gt_aligned = gt_df.loc[common_uids, common_labels]

    print(f"  {len(common_uids)} samples, {len(common_labels)} labels")

    results: dict[str, dict] = {}

    for label in common_labels:
        mask = pred_df[label].notna() & gt_aligned[label].notna()
        if mask.sum() == 0:
            print(f"  [{label}] skipped — no valid samples")
            results[label] = {"f1": None, "f1_ci": [None, None]}
            continue

        y_true = gt_aligned.loc[mask, label].values.astype(int)
        y_pred = pred_df.loc[mask, label].values.astype(int)

        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        lower, upper = bootstrap_f1_ci(y_true, y_pred)

        results[label] = {
            "f1": round(f1, 4),
            "f1_ci": [round(lower, 4), round(upper, 4)],
        }
        print(f"  [{label}]  F1={f1:.4f}  95% CI=[{lower:.4f}, {upper:.4f}]")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_f1_scores.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=4)


if not os.path.exists(GROUND_TRUTH_PATH):
    raise FileNotFoundError(f"Ground truth file not found: {GROUND_TRUTH_PATH}")

print(f"Loading ground truth from {GROUND_TRUTH_PATH} …\n")
gt_df = load_predictions(GROUND_TRUTH_PATH)

for csv_path in RESULTS_PATHS:
    if not os.path.exists(csv_path):
        print(f"File not found, skipping: {csv_path}")
        continue
    process_model(csv_path, gt_df, OUTPUT_DIR)

print("Done.")
