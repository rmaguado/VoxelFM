import os
import pandas as pd

DATA_PATH = "/scratch/VM/radio-foundation/essential/Merlin/merlinabdominalctdataset"
OUTPUT_PATH = "evaluation/Merlin"

os.makedirs(OUTPUT_PATH, exist_ok=True)

df_reports = pd.read_excel(os.path.join(DATA_PATH, "reports_final.xlsx")).rename(
    columns={"study id": "series_uid", "Findings": "report"}
)[["series_uid", "report"]]

df_reports["report"] = (
    df_reports["report"]
    .str.replace("\n", "", regex=False)
    .str.replace("FINDINGS:", "", regex=False)
    .str.split("IMPRESSION")
    .str[0]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df_reports.to_csv(os.path.join(OUTPUT_PATH, "reports.csv"), index=False)
