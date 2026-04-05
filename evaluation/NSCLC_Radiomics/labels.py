import os
import numpy as np
import pandas as pd

DATA_ROOT = "/scratch/VM/radio-foundation/datasets/NSCLC-Radiomics"
OUTPUT_PATH = "evaluation/NSCLC_Radiomics/labels.csv"

meta_df = pd.read_csv(os.path.join(DATA_ROOT, "metadata.csv"))
meta_df = meta_df[["PatientID", "age", "gender", "Survival.time", "deadstatus.event"]]

survival = np.array(meta_df["Survival.time"].to_list())
log1p_survival = np.log1p(survival / 100)

labels_df = pd.DataFrame(
    {
        "series_uid": meta_df["PatientID"].to_list(),
        "time": log1p_survival,
        "event": meta_df["deadstatus.event"].to_list(),
    }
)
labels_df.to_csv(OUTPUT_PATH, index=False)
