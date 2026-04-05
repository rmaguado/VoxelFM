import os
import pandas as pd

count_threshold = 500
DATA_PATH = "/scratch/VM/radio-foundation/essential/Merlin/merlinabdominalctdataset"
OUTPUT_PATH = "evaluation/Merlin/labels"
os.makedirs(OUTPUT_PATH, exist_ok=True)

df_findings = pd.read_csv(
    os.path.join(DATA_PATH, "zero_shot_findings_disease_cls.csv")
).rename(columns={"study id": "series_uid"})
labels = list(df_findings.columns)[1:]

for label in labels:
    label_df = (
        df_findings[["series_uid", label]].copy().rename(columns={label: "label"})
    )
    label_df["label"] = label_df["label"].replace(-1, 0)

    csv_path = os.path.join(OUTPUT_PATH, f"{label}.csv")

    label_df.to_csv(csv_path, index=False)
