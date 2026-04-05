import os
import json
import pandas as pd

LABELS_PATH = "evaluation/RSNA_PE/labels.csv"
SPLITS_PATH = "evaluation/RSNA_PE/splits"
OUTPUT_PATH = "evaluation/RSNA_PE/counts.csv"

train_ids = pd.read_csv(os.path.join(SPLITS_PATH, "train.csv"))["series_uid"]
valid_ids = pd.read_csv(os.path.join(SPLITS_PATH, "valid.csv"))["series_uid"]
test_ids = pd.read_csv(os.path.join(SPLITS_PATH, "test.csv"))["series_uid"]

output = {
    "train_count": [],
    "valid_count": [],
    "test_count": [],
    "train_frac": [],
    "valid_frac": [],
    "test_frac": [],
}

label_df = pd.read_csv(LABELS_PATH)
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
