import pandas as pd
import os
from glob import glob


generated_reports_path = "mllm/preprocessing/out"
report_files = glob(os.path.join(generated_reports_path, "restructured_reports_*.csv"))

df = pd.concat(
    [
        pd.read_csv(f, header=None, names=["report_id", "report"], delimiter=";")
        for f in report_files
    ],
    ignore_index=True,
)

mapping_file_path = os.path.join(generated_reports_path, "report_mapping.csv")
mappings = pd.read_csv(mapping_file_path)

merged = pd.merge(mappings, df, on="report_id", how="left")[["VolumeName", "report"]]

output_file = os.path.join(generated_reports_path, "combined_reports.csv")
merged.to_csv(output_file, index=False, sep=";")
