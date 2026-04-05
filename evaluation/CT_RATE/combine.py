import os
import json
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

RESULTS_PATHS = {
    "VoxelFM": "evaluation/CT_RATE/output/dino/patch",
    "M3D": "evaluation/CT_RATE/output/m3d/patch",
    "CT-CLIP": "evaluation/CT_RATE/output/ctclip/patch",
    "Merlin": "evaluation/CT_RATE/output/merlin/patch",
    "RadFM": "evaluation/CT_RATE/output/radfm/patch",
}
OUTPUT_DIR = "evaluation/CT_RATE/output/f1_scores"

N_BOOTSTRAP = 1000
CI_ALPHA = 0.95
RANDOM_SEED = 42


def bootstrap_f1_ci(
    y_true, y_pred, n_bootstrap=N_BOOTSTRAP, alpha=CI_ALPHA, seed=RANDOM_SEED
):
    """Return (lower, upper) bootstrap percentile CI for the F1 score."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
    lo = (1 - alpha) / 2 * 100
    hi = (1 + alpha) / 2 * 100
    return float(np.percentile(scores, lo)), float(np.percentile(scores, hi))


os.makedirs(OUTPUT_DIR, exist_ok=True)

for baseline, results_folder_path in RESULTS_PATHS.items():
    model_results = {}

    if not os.path.exists(results_folder_path):
        print(
            f"WARNING: Results folder not found for {baseline}: {results_folder_path}"
        )
        continue

    for subfolder in os.listdir(results_folder_path):
        subfolder_path = os.path.join(results_folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Load config to get label name
        config_path = os.path.join(subfolder_path, "config.yaml")
        if not os.path.exists(config_path):
            print(f"WARNING: config.yaml not found for {baseline}/{subfolder}")
            continue
        cfg = OmegaConf.load(config_path)
        label = cfg["label"].replace("_", " ").capitalize()

        # Load predictions
        predictions_path = os.path.join(subfolder_path, "predictions.csv")
        if not os.path.exists(predictions_path):
            print(f"WARNING: predictions.csv not found for {baseline}/{subfolder}")
            model_results[label] = {"f1": None, "f1_ci": [None, None]}
            continue

        try:
            df = pd.read_csv(predictions_path)
            if "label" not in df.columns or "prob" not in df.columns:
                print(
                    f"WARNING: predictions.csv missing required columns for {baseline}/{subfolder}"
                )
                model_results[label] = {"f1": None, "f1_ci": [None, None]}
                continue

            y_true = df["label"].values.astype(int)
            y_pred = (df["prob"].values >= 0.5).astype(int)

            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            lower, upper = bootstrap_f1_ci(y_true, y_pred)

            model_results[label] = {
                "f1": round(f1, 4),
                "f1_ci": [round(lower, 4), round(upper, 4)],
            }
            print(
                f"  [{baseline}] [{label}]  F1={f1:.4f}  95% CI=[{lower:.4f}, {upper:.4f}]"
            )

        except Exception as e:
            print(f"WARNING: Error processing {baseline}/{subfolder}: {e}")
            model_results[label] = {"f1": None, "f1_ci": [None, None]}

    # Write per-model JSON
    out_path = os.path.join(
        OUTPUT_DIR, f"{baseline.lower().replace('-', '_')}_f1_scores.json"
    )
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=4)
    print(f"[{baseline}] → saved to {out_path}\n")

print("Done.")
