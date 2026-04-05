import os
import json
import pandas as pd


VQA_PATH = "/scratch/VM/radio-foundation/essential/CT-RATE/dataset/vqa/train_vqa.json"
REPORTS_PATH = "mllm/preprocessing/out/ctrate/reports/v1/train.csv"
OUTPUT_PATH = "mllm/preprocessing/out/ctrate/vqa/train.json"

reports = pd.read_csv(REPORTS_PATH, delimiter=";")
report_series_uids = reports["series_uid"].to_list()

with open(VQA_PATH, "r") as f:
    conversations = json.load(f)

result = []

for raw_data in conversations:
    cleaned = {}
    processed_report = ""
    if "image" in raw_data:

        series_uid = raw_data["image"].replace(".nii.gz", "")
        cleaned["image"] = series_uid

        if series_uid not in report_series_uids:
            continue

        if any("<report_generation>" in x["value"] for x in raw_data["conversations"]):
            processed_report = reports[reports["series_uid"] == series_uid].iloc[0][
                "report"
            ]

    conversation = []

    for msg in raw_data["conversations"]:
        source = msg["from"]
        value = msg["value"]

        if source == "gpt" and processed_report:
            conversation.append({"from": source, "value": processed_report})
        else:
            conversation.append({"from": source, "value": value})

    cleaned["conversations"] = conversation

    result.append(cleaned)


with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=4)
