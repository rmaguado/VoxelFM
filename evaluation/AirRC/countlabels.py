import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

SEG_PATH = "/scratch/VM/radio-foundation/preprocessed/AirRC/segmentations"
SPLITS_PATH = "evaluation/AirRC/splits"
OUTPUT_PATH = "evaluation/AirRC/output/counts.csv"

series_uids = [
    x.replace(".npy", "") for x in os.listdir(SEG_PATH) if x.endswith(".npy")
]

map_to_labels = {
    "1": "Airway Lumen",
    "2": "Airway Wall",
    "3": "Pulmonary Veins",
    "4": "Pulmonary Arteries",
}

train_ids = pd.read_csv(
    os.path.join(SPLITS_PATH, "train.csv"), dtype={"series_uid": str}
)["series_uid"].tolist()

valid_ids = pd.read_csv(
    os.path.join(SPLITS_PATH, "valid.csv"), dtype={"series_uid": str}
)["series_uid"].tolist()

test_ids = pd.read_csv(
    os.path.join(SPLITS_PATH, "test.csv"), dtype={"series_uid": str}
)["series_uid"].tolist()

print(series_uids[:10])
print(train_ids[:10])

exit()

train_counts = {}
valid_counts = {}
test_counts = {}
for uid in tqdm(series_uids):
    if uid in train_ids:
        label_counts = train_counts
    elif uid in valid_ids:
        label_counts = valid_counts
    elif uid in test_ids:
        label_counts = test_counts
    else:
        print(f"skipping uid not in any split: {uid}")
        continue

    path = os.path.join(SEG_PATH, f"{uid}.npy")
    seg = np.load(path)
    unique, counts = np.unique(seg, return_counts=True)
    for val, count in zip(unique, counts):
        if val == 0:
            continue
        val_str = str(val)
        if val_str in label_counts:
            label_counts[val_str] += int(count)
        else:
            label_counts[val_str] = int(count)


output = {"label": [], "train": [], "valid": [], "test": []}
for val in range(1, 5):
    str_val = str(val)

    train_count = train_counts.get(str_val, 0)
    valid_count = valid_counts.get(str_val, 0)
    test_count = test_counts.get(str_val, 0)

    label_name = map_to_labels[str_val]

    output["label"].append(label_name)
    output["train"].append(train_count)
    output["valid"].append(valid_count)
    output["test"].append(test_count)

pd.DataFrame(output).to_csv(OUTPUT_PATH, index=False)
