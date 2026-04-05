import os
import pandas as pd


def norm_df(df):
    values = df["label"].to_numpy()
    fmean = values.mean()
    fstd = values.std()

    values = (values - fmean) / fstd
    df["label"] = values
    return df


DATA_ROOT = "/mnt/typhon/data/AI/DeepRDT/iCTCF"
OUTPUT_PATH = "evaluation/iCTCF/labels"
os.makedirs(OUTPUT_PATH, exist_ok=True)

img_root = os.path.join(DATA_ROOT, "ct")
series_uids = [x for x in os.listdir(img_root)]

df = pd.read_csv(os.path.join(DATA_ROOT, "clinical.csv"))
df["series_uid"] = df["ID"].str.replace(" ", "_")
df = df[df["series_uid"].isin(series_uids)]
df = df[
    [
        "series_uid",
        "SARS-CoV-2 nucleic acids",
        "Morbidity outcome",
        "HGB",
        "WBC",
        "CRP",
        "PCT",
        "CK",
    ]
]

severe_categories = ["Severe", "Critically ill"]
df["severity"] = df["Morbidity outcome"].apply(lambda x: x in severe_categories)

df["covid"] = df["SARS-CoV-2 nucleic acids"].apply(lambda x: x == "Positive")

df_severity = df[["series_uid", "severity"]].rename(columns={"severity": "label"})
df_covid = df[["series_uid", "covid"]].rename(columns={"covid": "label"})
df_HGB = df[["series_uid", "HGB"]].rename(columns={"HGB": "label"}).dropna()
df_WBC = df[["series_uid", "WBC"]].rename(columns={"WBC": "label"}).dropna()
df_CRP = df[["series_uid", "CRP"]].rename(columns={"CRP": "label"}).dropna()
df_PCT = df[["series_uid", "PCT"]].rename(columns={"PCT": "label"}).dropna()
df_CK = df[["series_uid", "CK"]].rename(columns={"CK": "label"}).dropna()

df_HGB = norm_df(df_HGB)
df_WBC = norm_df(df_WBC)
df_CRP = norm_df(df_CRP)
df_PCT = norm_df(df_PCT)
df_CK = norm_df(df_CK)

df_severity.to_csv(os.path.join(OUTPUT_PATH, "severity.csv"), index=False)
df_covid.to_csv(os.path.join(OUTPUT_PATH, "covid.csv"), index=False)
df_HGB.to_csv(os.path.join(OUTPUT_PATH, "hgb.csv"), index=False)
df_WBC.to_csv(os.path.join(OUTPUT_PATH, "wbc.csv"), index=False)
df_CRP.to_csv(os.path.join(OUTPUT_PATH, "crp.csv"), index=False)
df_PCT.to_csv(os.path.join(OUTPUT_PATH, "pct.csv"), index=False)
df_CK.to_csv(os.path.join(OUTPUT_PATH, "ck.csv"), index=False)
