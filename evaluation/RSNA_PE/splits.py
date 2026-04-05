import os
import pandas as pd

train_frac = 0.7
valid_frac = 0.15

output_path = "evaluation/RSNA_PE/splits"
labels_path = "evaluation/RSNA_PE/labels.csv"

df = pd.read_csv(labels_path)
ids = pd.DataFrame({"series_uid": df["series_uid"].unique()})

trainval_ids = ids.sample(frac=train_frac + valid_frac, random_state=4)

train_ids = trainval_ids.sample(
    frac=train_frac / (train_frac + valid_frac), random_state=4
)

valid_ids = trainval_ids[~trainval_ids["series_uid"].isin(train_ids["series_uid"])]

test_ids = ids[~ids["series_uid"].isin(trainval_ids["series_uid"])]

os.makedirs(output_path, exist_ok=True)

train_ids.to_csv(os.path.join(output_path, "train.csv"), index=False)
valid_ids.to_csv(os.path.join(output_path, "valid.csv"), index=False)
test_ids.to_csv(os.path.join(output_path, "test.csv"), index=False)
