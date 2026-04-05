import os
import json
import time
import logging
import concurrent.futures
from threading import Lock

import pandas as pd
from ollama import Client


MODEL_NAME = "gpt-oss:120b"
OUTPUT_DIR = "mllm/preprocessing/out/ctrate/reports/train"
PROMPT_FILE = "mllm/preprocessing/prompts/v1.txt"
REPORTS_FILE = "evaluation/CT_RATE/reports/train.csv"
LOG_FILE = os.path.join(OUTPUT_DIR, "restructure_reports.log")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "restructured_reports.csv")
REQUEST_DELAY_SEC = 0.2

START_TAG = "<report>"
END_TAG = "</report>"

ports_cc3 = list(range(11434, 11434 + 2))
HOSTS = [f"http://192.168.36.203:{port}" for port in ports_cc3]


class CheckpointManager:
    """Thread-safe JSONL checkpoint manager."""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.lock = Lock()
        self.processed = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self.checkpoint_file):
            return {}

        data = {}
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        key = entry.get("report_normalized")
                        if key:
                            data[key] = entry["structured_report"]
            logging.info(f"Loaded {len(data)} items from checkpoint")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")

        return data

    def is_processed(self, key: str) -> bool:
        with self.lock:
            return key in self.processed

    def add(self, key: str, structured_report: str):
        with self.lock:
            if key in self.processed:
                return

            self.processed[key] = structured_report
            with open(self.checkpoint_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "report_normalized": key,
                            "structured_report": structured_report,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def get_all(self) -> dict:
        with self.lock:
            return dict(self.processed)


def get_response(client: Client, system_prompt: str, query: str) -> str:
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"]


def extract_structured_report(response: str) -> str:
    if START_TAG in response and END_TAG in response:
        response = response.split(START_TAG, 1)[1].split(END_TAG, 1)[0]
    return response.replace("\n", "").replace(";", ",").strip()


def process_report(report_text: str, system_prompt: str, client: Client) -> str:
    try:
        raw_response = get_response(client, system_prompt, report_text)

        if START_TAG not in raw_response or END_TAG not in raw_response:
            logging.error("Missing delimiters in response")
            return ""

        structured = extract_structured_report(raw_response)

        if not structured.strip():
            logging.error("Empty structured report after extraction")
            return ""

        time.sleep(REQUEST_DELAY_SEC)
        return structured

    except Exception as e:
        logging.error(f"Error processing report: {e}")
        return ""


def normalize_text(text: str) -> str:
    """Normalize text for deduplication."""
    return " ".join(str(text).split())


def load_system_prompt() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def get_processed_reports(output_file: str) -> dict:
    """Load already processed reports (text -> structured)."""
    processed = {}
    if not os.path.exists(output_file):
        return processed

    try:
        df = pd.read_csv(output_file, sep=";")
        for _, row in df.iterrows():
            processed[normalize_text(row["report"])] = row["structured_report"]
        logging.info(f"Loaded {len(processed)} processed reports")
    except Exception as e:
        logging.warning(f"Could not load processed reports: {e}")

    return processed


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    df = pd.read_csv(REPORTS_FILE)
    system_prompt = load_system_prompt()

    df["report_normalized"] = df["report"].apply(normalize_text)
    unique_reports = df.drop_duplicates(subset="report_normalized")

    total = len(df)
    unique = len(unique_reports)
    logging.info(f"Total: {total}, Unique: {unique}, Duplicates: {total - unique}")

    checkpoint_file = OUTPUT_FILE.replace(".csv", "_checkpoint.jsonl")
    checkpoint = CheckpointManager(checkpoint_file)

    unique_reports = unique_reports[
        ~unique_reports["report_normalized"].apply(checkpoint.is_processed)
    ].reset_index(drop=True)

    logging.info(f"Processing {len(unique_reports)} unique reports")

    if len(unique_reports) == 0:
        logging.info("No new reports to process")
        return

    clients = [Client(host=host) for host in HOSTS]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = {}

        for i, (_, row) in enumerate(unique_reports.iterrows()):
            client = clients[i % len(clients)]
            future = executor.submit(
                process_report,
                row["report"],
                system_prompt,
                client,
            )
            futures[future] = row["report_normalized"]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            report_norm = futures[future]
            try:
                structured = future.result()
                if structured:
                    checkpoint.add(report_norm, structured)
                    completed += 1
                    logging.info(f"Progress: {completed}/{len(unique_reports)}")
            except Exception as e:
                logging.error(f"Future raised exception: {e}")

    processed = checkpoint.get_all()
    df["structured_report"] = df["report_normalized"].map(processed)
    df = df[df["structured_report"].notna()]

    output_df = df[["series_uid", "structured_report"]].rename(
        columns={"structured_report": "report"}
    )
    output_df.to_csv(OUTPUT_FILE, sep=";", index=False)

    logging.info(f"Saved {len(output_df)} results to {OUTPUT_FILE}")
    print(f"Total processed reports: {len(processed)}")


if __name__ == "__main__":
    main()
