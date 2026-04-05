import os
import json
import pandas as pd

LABELS_PATH = "evaluation/Merlin/labels"
SPLITS_PATH = "evaluation/Merlin/splits"
OUTPUT_PATH = "evaluation/Merlin/counts.csv"

train_ids = pd.read_csv(os.path.join(SPLITS_PATH, "train.csv"))["series_uid"]
valid_ids = pd.read_csv(os.path.join(SPLITS_PATH, "valid.csv"))["series_uid"]
test_ids = pd.read_csv(os.path.join(SPLITS_PATH, "test.csv"))["series_uid"]

all_labels = [
    x.replace(".csv", "") for x in os.listdir(LABELS_PATH) if x.endswith(".csv")
]
all_labels.sort()

output = {
    "label": [x.replace("_", " ").capitalize() for x in all_labels],
    "train_count": [],
    "valid_count": [],
    "test_count": [],
    "train_frac": [],
    "valid_frac": [],
    "test_frac": [],
}

for label in all_labels:
    label_path = os.path.join(LABELS_PATH, f"{label}.csv")
    label_df = pd.read_csv(label_path)
    train_labels = label_df[label_df["series_uid"].isin(train_ids)]["label"]
    valid_labels = label_df[label_df["series_uid"].isin(valid_ids)]["label"]
    test_labels = label_df[label_df["series_uid"].isin(test_ids)]["label"]

    train_count = train_labels.sum()
    valid_count = valid_labels.sum()
    test_count = test_labels.sum()

    train_frac = train_count / len(train_labels)
    valid_frac = valid_count / len(valid_labels)
    test_frac = test_count / len(test_labels)

    output["train_count"].append(train_count)
    output["valid_count"].append(valid_count)
    output["test_count"].append(test_count)

    output["train_frac"].append(train_frac)
    output["valid_frac"].append(valid_frac)
    output["test_frac"].append(test_frac)

result_df = pd.DataFrame(output)
result_df.to_csv(OUTPUT_PATH, index=False)
