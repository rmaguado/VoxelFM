import os
import random
import pandas as pd

random.seed(4)

DATA_PATH = "/mnt/typhon/data/AI/DeepRDT/OSIC"
OUTPUT_PATH = "evaluation/OSIC/splits"

os.makedirs(OUTPUT_PATH, exist_ok=True)

img_root = os.path.join(DATA_PATH, "ct")
series_uids = [
    x.replace(".nii.gz", "") for x in os.listdir(img_root) if x.endswith(".nii.gz")
]
random.shuffle(series_uids)

n_series = len(series_uids)
n_train = int(n_series * 0.7)
n_valid = int(n_series * 0.15)

train_series_uids = series_uids[:n_train]
valid_series_uids = series_uids[n_train : n_train + n_valid]
test_series_uids = series_uids[n_train + n_valid :]

df_train = pd.DataFrame({"series_uid": train_series_uids})
df_valid = pd.DataFrame({"series_uid": valid_series_uids})
df_test = pd.DataFrame({"series_uid": test_series_uids})

df_train.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
df_valid.to_csv(os.path.join(OUTPUT_PATH, "valid.csv"), index=False)
df_test.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), index=False)
