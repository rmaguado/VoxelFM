import json
import pandas as pd


REPORTS_PATH = "mllm/preprocessing/out/ctrate/reports/v1/train.csv"
OUTPUT_PATH = "mllm/preprocessing/out/ctrate/reports/v1/train.json"

reports = pd.read_csv(REPORTS_PATH, delimiter=";")

result = []

reports_uids = reports["series_uid"].to_list()


for idx, row in reports.iterrows():

    cleaned = {}
    cleaned["image"] = row.series_uid
    cleaned["conversations"] = [
        {
            "from": "human",
            "value": "<image>\nGenerate a radiology report for this CT scan.",
        },
        {"from": "gpt", "value": row.report},
    ]

    result.append(cleaned)


with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=4)
