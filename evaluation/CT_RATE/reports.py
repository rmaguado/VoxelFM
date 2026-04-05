import os
import pandas as pd

DATA_PATH = "/scratch/VM/radio-foundation/essential/CT-RATE/dataset/radiology_text_reports/validation_reports.csv"
OUTPUT_PATH = "evaluation/CT_RATE/reports"
OUTPUT_NAME = "valid.csv"

os.makedirs(OUTPUT_PATH, exist_ok=True)

df_reports = pd.read_csv(DATA_PATH).rename(
    columns={"VolumeName": "series_uid", "Findings_EN": "report"}
)[["series_uid", "report"]]

df_reports["series_uid"] = df_reports["series_uid"].str.replace(".nii.gz", "")

df_reports.to_csv(os.path.join(OUTPUT_PATH, OUTPUT_NAME), index=False)
