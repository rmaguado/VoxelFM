import os
import random
import pandas as pd

DATA_PATH = "/scratch/VM/radio-foundation/datasets/CT_RATE/multi_abnormality_labels"
OUTPUT_PATH = "evaluation/CT_RATE/splits"
train_frac = 0.8
random.seed(4)

trainval_df = pd.read_csv(os.path.join(DATA_PATH, "train_predicted_labels.csv"))

trainval_series_uids = trainval_df["VolumeName"].str.replace(".nii.gz", "").to_list()
trainval_patids = ["_".join(x.split("_")[:2]) for x in trainval_series_uids]

random.shuffle(trainval_patids)

n_train = int(train_frac * len(trainval_patids))
train_patids = trainval_patids[:n_train]

train_series_uids = [
    x for x in trainval_series_uids if "_".join(x.split("_")[:2]) in train_patids
]
valid_series_uids = [x for x in trainval_series_uids if x not in train_series_uids]

train_df = pd.DataFrame({"series_uid": train_series_uids})
valid_df = pd.DataFrame({"series_uid": valid_series_uids})

os.makedirs(OUTPUT_PATH, exist_ok=True)
train_df.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
valid_df.to_csv(os.path.join(OUTPUT_PATH, "valid.csv"), index=False)

test_df = pd.read_csv(os.path.join(DATA_PATH, "valid_predicted_labels.csv"))
test_df["series_uid"] = test_df["VolumeName"].str.replace(".nii.gz", "")
test_df = test_df[["series_uid"]]
test_df.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), index=False)
