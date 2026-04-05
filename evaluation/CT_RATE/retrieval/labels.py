import os
import pandas as pd

META_PATH = "/scratch/VM/radio-foundation/datasets/CT_RATE/multi_abnormality_labels/valid_predicted_labels.csv"

df = pd.read_csv(META_PATH)
labels = df.columns[1:]

retrieval_labels = {"series_uid": [], "labels": []}

for idx, row in df.iterrows():
    series_uid = row["VolumeName"].replace(".nii.gz", "")
    series_suffix = series_uid.split("_")[-1]
    if series_suffix != "1":
        continue
    series_labels = [l.lower().replace(" ", "_") for l in labels if row[l]]
    if series_labels:
        retrieval_labels["series_uid"].append(series_uid)
        retrieval_labels["labels"].append(";".join(series_labels))

out_df = pd.DataFrame(retrieval_labels)
out_df.to_csv("evaluation/CT_RATE/retrieval/labels.csv", index=False)
