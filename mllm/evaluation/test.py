import os
import re
import json
import pandas as pd
from ollama import Client
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ABNORMALITIES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

EXTRACTION_SYSTEM_PROMPT = f"""Extract the presence or absence of predefined abnormalities from a radiology report.

### OUTPUT

Return a single valid JSON object:

* Keys must exactly match the target abnormalities list below.
* Values must be boolean (`true` or `false`).

### TARGET ABNORMALITIES & DEFINITIONS

1. **Medical material**: implanted or surgical devices (e.g., pacemakers, stents, clips, wires, catheters, prosthetic valves, fixation hardware).
2. **Arterial wall calcification**: calcification or atherosclerosis of the aorta or major arteries.
3. **Cardiomegaly**: enlarged heart size.
4. **Pericardial effusion**: fluid around the heart.
5. **Coronary artery wall calcification**: calcified plaque in the coronary arteries.
6. **Hiatal hernia**: stomach protruding into the chest.
7. **Lymphadenopathy**: enlarged or abnormal hilar or mediastinal lymph nodes.
8. **Emphysema**: emphysematous changes such as bullae or paraseptal/centrilobular destruction.
9. **Atelectasis**: partial or complete lung collapse, including linear or subsegmental forms.
10. **Lung nodule**: focal lung lesions including nodules, granulomas, or masses.
11. **Lung opacity**: nonspecific lung opacities (e.g., ground-glass or infiltrates) not classified as consolidation or atelectasis.
12. **Pulmonary fibrotic sequela**: lung scarring or fibrotic changes (e.g., bands, apical capping).
13. **Pleural effusion**: fluid in the pleural space or blunted costophrenic angles.
14. **Mosaic attenuation pattern**: patchy lung attenuation suggesting air trapping or perfusion differences.
15. **Peribronchial thickening**: thickened bronchial walls or bronchial cuffing.
16. **Consolidation**: dense airspace filling, with or without air bronchograms.
17. **Bronchiectasis**: permanently dilated bronchi (e.g., tram-track or signet-ring appearance).
18. **Interlobular septal thickening**: thickened interlobular septa.
"""


class CheckpointManager:
    """Manages checkpointing using JSONL format."""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.lock = Lock()
        self.processed = self._load()

    def _load(self) -> Dict:
        if not os.path.exists(self.checkpoint_file):
            return {}
        try:
            data = {}
            with open(self.checkpoint_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        key = entry.pop("report_text", entry.pop("series_uid", None))
                        if key:
                            data[key] = entry
            logger.info(f"Loaded {len(data)} processed items from checkpoint")
            return data
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return {}

    def is_processed(self, key: str) -> bool:
        with self.lock:
            return key in self.processed

    def add_result(self, key: str, result: Dict, is_text: bool = False):
        with self.lock:
            self.processed[key] = result
            entry = {"report_text" if is_text else "series_uid": key, **result}
            with open(self.checkpoint_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def get_result(self, key: str) -> Dict:
        with self.lock:
            return self.processed.get(key, {})


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.split())


def find_duplicates(reports_data: List[Dict], report_key: str) -> tuple:
    """Find duplicate reports and return mappings."""
    text_to_uids = {}
    uid_to_text = {}

    for item in reports_data:
        uid = item["series_uid"]
        text = normalize_text(item[report_key])
        uid_to_text[uid] = text
        text_to_uids.setdefault(text, []).append(uid)

    unique = len(text_to_uids)
    total = len(reports_data)
    logger.info(f"Total: {total}, Unique: {unique}, Duplicates: {total - unique}")

    return text_to_uids, uid_to_text


def parse_response(response: str) -> Dict[str, bool]:
    """Parse JSON response from model."""
    try:
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        result = json.loads(match.group(0))
        return {k: result.get(k, False) for k in ABNORMALITIES}
    except Exception as e:
        logger.error(f"Failed to parse: {e}")
        return {}


def extract_abnormalities(report: str, client: Client, model: str) -> Dict[str, bool]:
    """Extract abnormalities using Ollama."""
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Report:\n{report}"},
            ],
        )
        return parse_response(response["message"]["content"])
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {}


def process_unique_report(
    report_text: str,
    client: Client,
    model: str,
    checkpoint: CheckpointManager,
) -> Dict:
    """Process a single unique report."""
    if checkpoint.is_processed(report_text):
        return checkpoint.get_result(report_text)

    abnormalities = extract_abnormalities(report_text, client, model)
    if abnormalities:
        checkpoint.add_result(report_text, abnormalities, is_text=True)
    return abnormalities


def process_reports(
    reports_data: List[Dict],
    ollama_hosts: List[str],
    model: str,
    report_key: str,
    output_csv: str,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Process reports in parallel with deduplication."""
    checkpoint_file = output_csv.replace(".csv", "_checkpoint.jsonl")
    checkpoint = CheckpointManager(checkpoint_file)

    text_to_uids, uid_to_text = find_duplicates(reports_data, report_key)

    remaining = [text for text in text_to_uids if not checkpoint.is_processed(text)]
    logger.info(f"Processing {len(remaining)}/{len(text_to_uids)} unique reports")

    if remaining:
        clients = [Client(host=host) for host in ollama_hosts]
        max_workers = max_workers or len(clients)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, text in enumerate(remaining):
                client = clients[i % len(clients)]
                future = executor.submit(
                    process_unique_report, text, client, model, checkpoint
                )
                futures[future] = text

            completed = len(text_to_uids) - len(remaining)
            for future in as_completed(futures):
                text = futures[future]
                try:
                    if future.result():
                        completed += 1
                        logger.info(
                            f"Progress: {completed}/{len(text_to_uids)} "
                            f"({len(text_to_uids[text])} copies)"
                        )
                except Exception as e:
                    logger.error(f"Error: {e}")

    results = []
    for uid, text in uid_to_text.items():
        abnormalities = checkpoint.get_result(text)
        if abnormalities:
            results.append({"series_uid": uid, **abnormalities})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(results)} results to {output_csv}")
    return df


def calculate_metrics(gt_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate precision, recall, F1, and accuracy."""
    merged = gt_df.merge(gen_df, on="series_uid", suffixes=("_gt", "_gen"))
    metrics = []

    for abn in ABNORMALITIES:
        gt_col, gen_col = f"{abn}_gt", f"{abn}_gen"
        tp = ((merged[gt_col]) & (merged[gen_col])).sum()
        fp = ((~merged[gt_col]) & (merged[gen_col])).sum()
        fn = ((merged[gt_col]) & (~merged[gen_col])).sum()
        tn = ((~merged[gt_col]) & (~merged[gen_col])).sum()

        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn)

        metrics.append(
            {
                "abnormality": abn,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "accuracy": acc,
                "support": tp + fn,
            }
        )

    df = pd.DataFrame(metrics)

    logger.info(f"Macro Precision: {df['precision'].mean():.4f}")
    logger.info(f"Macro Recall: {df['recall'].mean():.4f}")
    logger.info(f"Macro F1: {df['f1_score'].mean():.4f}")

    return df


def main(
    reports_json: str,
    ollama_hosts: List[str],
    model: str,
    groundtruth_dir: str,
    generated_dir: str,
    max_workers: Optional[int] = None,
):
    os.makedirs(groundtruth_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    groundtruth_path = os.path.join(groundtruth_dir, "ground_truth_abnormalities.csv")
    generated_path = os.path.join(generated_dir, "generated_abnormalities.csv")

    logger.info(f"Using {len(ollama_hosts)} Ollama worker(s)")

    with open(reports_json) as f:
        reports_data = json.load(f)
    logger.info(f"Loaded {len(reports_data)} reports")

    if os.path.exists(groundtruth_path):
        logger.info(f"Loading existing ground truth from {groundtruth_path}")
        gt_df = pd.read_csv(groundtruth_path)
    else:
        logger.info("Processing ground truth reports...")
        gt_df = process_reports(
            reports_data,
            ollama_hosts,
            model,
            "ground_truth",
            groundtruth_path,
            max_workers,
        )

    logger.info("Processing generated reports...")
    gen_df = process_reports(
        reports_data, ollama_hosts, model, "generated", generated_path, max_workers
    )

    metrics_path = os.path.join(generated_dir, "evaluation_metrics.csv")

    logger.info("Calculating metrics...")
    metrics_df = calculate_metrics(gt_df, gen_df)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main(
        reports_json="runs/mllm/reports/dino/03_inference/nucleus/all_results.json",
        groundtruth_dir="mllm/evaluation/out",
        generated_dir="mllm/evaluation/out/dino",
        ollama_hosts=[
            "http://127.0.0.1:11434",
            "http://127.0.0.1:11435",
        ],
        model="gpt-oss:120b",
    )
