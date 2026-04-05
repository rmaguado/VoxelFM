import os
import random
import pandas as pd

random.seed(4)

LABELS_PATH = "evaluation/Mycobacterial/labels.csv"
OUTPUT_PATH = "evaluation/Mycobacterial/splits"
os.makedirs(OUTPUT_PATH, exist_ok=True)

labels_df = pd.read_csv(LABELS_PATH)

neg_uids = labels_df[labels_df["label"] == 0]["series_uid"].to_list()
pos_uids = labels_df[labels_df["label"] == 1]["series_uid"].to_list()
random.shuffle(neg_uids)
random.shuffle(pos_uids)

n_train_pos = int(len(pos_uids) * 0.7)
n_train_neg = int(len(neg_uids) * 0.7)

n_valid_pos = int(len(pos_uids) * 0.15)
n_valid_neg = int(len(neg_uids) * 0.15)

train_uids = pos_uids[:n_train_pos] + neg_uids[:n_train_neg]
valid_uids = (
    pos_uids[n_train_pos : n_train_pos + n_valid_pos]
    + neg_uids[n_train_neg : n_train_neg + n_valid_neg]
)
test_uids = (
    pos_uids[n_train_pos + n_valid_pos :] + neg_uids[n_train_neg + n_valid_neg :]
)

random.shuffle(train_uids)
random.shuffle(valid_uids)
random.shuffle(test_uids)


train_df = pd.DataFrame({"series_uid": train_uids})
valid_df = pd.DataFrame({"series_uid": valid_uids})
test_df = pd.DataFrame({"series_uid": test_uids})

train_df.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
valid_df.to_csv(os.path.join(OUTPUT_PATH, "valid.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), index=False)
