import os
import pandas as pd


DATA_ROOT = "/mnt/typhon/data/AI/DeepRDT/OSIC"
OUTPUT_PATH = "evaluation/OSIC"

img_root = os.path.join(DATA_ROOT, "ct")
series_uids = [
    x.replace(".nii.gz", "") for x in os.listdir(img_root) if x.endswith(".nii.gz")
]

df = pd.read_csv(os.path.join(DATA_ROOT, "baselines.csv"), dtype={"PATIENT ID": str})
df = df.rename(columns={"PATIENT ID": "series_uid", "FVC PREDICTED": "fvc"})
df["series_uid"] = df["series_uid"].apply(lambda x: x + "_0")
df = df[["series_uid", "fvc"]].dropna()
df = df[df["series_uid"].isin(series_uids)]

fvc = df["fvc"].to_numpy()
fvc_norm = (fvc - fvc.mean()) / fvc.std()
df["fvc"] = fvc_norm

df = df.rename(columns={"fvc": "label"})
df.to_csv(os.path.join(OUTPUT_PATH, "labels.csv"), index=False)
