import os
import random
import pandas as pd

random.seed(4)

train_frac = 0.7
valid_frac = 0.15

data_path = "/scratch/VM/radio-foundation/preprocessed/TotalSegmentator/metadata.csv"
meta_path = "/scratch/VM/radio-foundation/datasets-nodicom/Totalsegmentator/meta.csv"
output_path = "evaluation/TotalSegmentator/splits"


df_data = pd.read_csv(data_path, dtype={"crop_id": str})
df_meta = pd.read_csv(meta_path, delimiter=";")


study_type_counts = df_meta["study_type"].value_counts().to_dict()

train_scan_ids = []
valid_scan_ids = []
test_scan_ids = []

for study_type, counts in study_type_counts.items():
    study_scan_ids = df_meta[df_meta["study_type"] == study_type]["image_id"].to_list()
    random.shuffle(study_scan_ids)

    if counts < 20:
        train_scan_ids += study_scan_ids
        continue

    n_scans = len(study_scan_ids)
    num_train_scans = int(n_scans * train_frac)
    num_valid_scans = int(n_scans * valid_frac)

    train_scan_ids += study_scan_ids[:num_train_scans]
    valid_scan_ids += study_scan_ids[
        num_train_scans : num_train_scans + num_valid_scans
    ]
    test_scan_ids += study_scan_ids[num_train_scans + num_valid_scans :]


train_series_uids = df_data[df_data["volume_name"].isin(train_scan_ids)]["crop_id"]
valid_series_uids = df_data[df_data["volume_name"].isin(valid_scan_ids)]["crop_id"]
test_series_uids = df_data[df_data["volume_name"].isin(test_scan_ids)]["crop_id"]


train_ids = pd.DataFrame({"series_uid": train_series_uids})
valid_ids = pd.DataFrame({"series_uid": valid_series_uids})
test_ids = pd.DataFrame({"series_uid": test_series_uids})

os.makedirs(output_path, exist_ok=True)
train_ids.to_csv(os.path.join(output_path, "train.csv"), index=False)
valid_ids.to_csv(os.path.join(output_path, "valid.csv"), index=False)
test_ids.to_csv(os.path.join(output_path, "test.csv"), index=False)
