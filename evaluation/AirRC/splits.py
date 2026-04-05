import os
import random
import pandas as pd

random.seed(4)

data_path = "/scratch/VM/radio-foundation/preprocessed/AirRC/metadata.csv"
output_path = "evaluation/AirRC/splits"

df_data = pd.read_csv(data_path, dtype={"crop_id": str})

train_scan_ids = []
valid_scan_ids = []
test_scan_ids = []

volume_names = df_data["volume_name"].unique().tolist()
random.shuffle(volume_names)

total_volumes = len(volume_names)
n_train_vols = int(total_volumes * 0.7)
n_valid_vols = int(total_volumes * 0.15)

train_volumes = volume_names[:n_train_vols]
valid_volumes = volume_names[n_train_vols : n_train_vols + n_valid_vols]
test_volumes = volume_names[n_train_vols + n_valid_vols :]

train_series_uids = df_data[df_data["volume_name"].isin(train_volumes)]["crop_id"]
valid_series_uids = df_data[df_data["volume_name"].isin(valid_volumes)]["crop_id"]
test_series_uids = df_data[df_data["volume_name"].isin(test_volumes)]["crop_id"]


train_ids = pd.DataFrame({"series_uid": train_series_uids})
valid_ids = pd.DataFrame({"series_uid": valid_series_uids})
test_ids = pd.DataFrame({"series_uid": test_series_uids})

os.makedirs(output_path, exist_ok=True)
train_ids.to_csv(os.path.join(output_path, "train.csv"), index=False)
valid_ids.to_csv(os.path.join(output_path, "valid.csv"), index=False)
test_ids.to_csv(os.path.join(output_path, "test.csv"), index=False)
