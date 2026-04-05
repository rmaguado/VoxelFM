import os
import pandas as pd

DATA_PATH = "/scratch/VM/radio-foundation/essential/Merlin/merlinabdominalctdataset"
OUTPUT_PATH = "evaluation/Merlin/splits"

os.makedirs(OUTPUT_PATH, exist_ok=True)

df_splits = pd.read_excel(os.path.join(DATA_PATH, "reports_final.xlsx")).rename(
    columns={"study id": "series_uid"}
)[["series_uid", "Split"]]

df_train = df_splits[df_splits["Split"] == "train"][["series_uid"]]
df_val = df_splits[df_splits["Split"] == "val"][["series_uid"]]
df_test = df_splits[df_splits["Split"] == "test"][["series_uid"]]

df_train.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
df_val.to_csv(os.path.join(OUTPUT_PATH, "valid.csv"), index=False)
df_test.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), index=False)
