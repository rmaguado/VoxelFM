import os
import random
import pandas as pd

random.seed(4)

train_frac = 0.7
valid_frac = 0.15

data_path = "/scratch/VM/radio-foundation/preprocessed/Mediastinal/metadata.csv"
output_path = "evaluation/Mediastinal/splits"

df_data = pd.read_csv(data_path, dtype={"crop_id": str})

series_uids = df_data["series_uid"].unique().tolist()
volume_names = df_data["volume_name"].unique().tolist()
random.shuffle(volume_names)

n_train_volumes = int(0.7 * len(volume_names))
n_valid_volumes = int(0.15 * len(volume_names))

train_volumes = volume_names[:n_train_volumes]
valid_volumes = volume_names[n_train_volumes : n_train_volumes + n_valid_volumes]
test_volumes = volume_names[n_train_volumes + n_valid_volumes :]

train_series_uids = df_data[df_data["volume_name"].isin(train_volumes)]["series_uid"]
valid_series_uids = df_data[df_data["volume_name"].isin(valid_volumes)]["series_uid"]
test_series_uids = df_data[df_data["volume_name"].isin(test_volumes)]["series_uid"]

train_df = pd.DataFrame({"series_uid": train_series_uids})
valid_df = pd.DataFrame({"series_uid": valid_series_uids})
test_df = pd.DataFrame({"series_uid": test_series_uids})

os.makedirs(output_path, exist_ok=True)
train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
valid_df.to_csv(os.path.join(output_path, "valid.csv"), index=False)
test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)
