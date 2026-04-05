import os
import pandas as pd

DATA_ROOT = (
    "/mnt/typhon/data/AI/DeepRDT/Mycobacterial/damianhan/nifti-dataset/versions/5/"
)
OUTPUT_PATH = "evaluation/Mycobacterial"
os.makedirs(OUTPUT_PATH, exist_ok=True)

NTM_series_uids = [
    x.replace(".nii", "") for x in os.listdir(os.path.join(DATA_ROOT, "NTMNiFTi"))
]
TB_series_uids = [
    x.replace(".nii", "") for x in os.listdir(os.path.join(DATA_ROOT, "TBNifTI"))
]

series_uids = NTM_series_uids + TB_series_uids
labels = [0] * len(NTM_series_uids) + [1] * len(TB_series_uids)

df = pd.DataFrame({"series_uid": series_uids, "label": labels})

csv_path = os.path.join(OUTPUT_PATH, "labels.csv")
df.to_csv(csv_path, index=False)
