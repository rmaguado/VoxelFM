import os
import pandas as pd

DATA_PATH = "/mnt/logos/scratch/h501uvma/CT-RATE/dataset/multi_abnormality_labels"
OUTPUT_PATH = "evaluation/CT_RATE/labels"
os.makedirs(OUTPUT_PATH, exist_ok=True)

train_path = os.path.join(DATA_PATH, "train_predicted_labels.csv")
valid_path = os.path.join(DATA_PATH, "valid_predicted_labels.csv")

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)

df = pd.concat([train_df, valid_df])
labels = df.columns[1:]
df["series_uid"] = df["VolumeName"].str.replace(".nii.gz", "")

for label in labels:
    label_df = df[["series_uid", label]].rename(columns={label: "label"})
    label_output_path = os.path.join(
        OUTPUT_PATH, f"{label.lower().replace(" ", "_")}.csv"
    )
    label_df.to_csv(label_output_path, index=False)
