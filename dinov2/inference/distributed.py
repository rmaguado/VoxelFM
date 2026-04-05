import os
import json
import time
from typing import Callable, List, Optional, Sequence, Tuple

import logging
import logging.handlers

import torch
import torch.multiprocessing as mp
from queue import Empty
from multiprocessing.queues import Queue
from omegaconf import OmegaConf

from .processing import *
from .model import build_model
from .loaders import autoload_ct

SeriesUID = str
Path = str
GPUId = int

Job = Tuple[SeriesUID, Path]
Result = Tuple[SeriesUID, bool]
LoadFn = Callable[[Path], Tuple[torch.Tensor, Sequence[float]]]


def worker(
    gpu_id: GPUId,
    job_queue: Queue[Optional[Job]],
    result_queue: Queue[Result],
    load_fn: LoadFn,
    output_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    max_patches: int,
    log_path: Path,
    log_lock,
    log_queue,
    do_preprocessing=True,
) -> None:
    """Worker process targeting a specific GPU."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.handlers.QueueHandler(log_queue))
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(gpu_id)
        gpu_name = torch.cuda.get_device_name(gpu_id)

        config = OmegaConf.load(config_path)
        model = build_model(checkpoint_path, config, device=device)
        model.eval()

        fmean = config.datasets[0].norm.mean
        fstd = config.datasets[0].norm.std
        patch_size = config.student.patch_size

        logger.info(f"Worker GPU {gpu_id}:{gpu_name} ready")

        while True:
            try:
                job = job_queue.get(timeout=1)
            except Empty:
                continue

            if job is None:
                break

            uid, path = job

            try:
                t0 = time.time()
                img, spacing = load_fn(path)

                if do_preprocessing:
                    img = crop_volume(img, device, 21)
                    img = resize_isotropic(img, spacing, device)
                    img = resize_max_patches(img, patch_size, max_patches, device)
                    img = patch_crop(img, patch_size)

                x = (img - fmean) / fstd
                x = x.unsqueeze(0)

                with torch.no_grad():
                    features = generate_embeddings(x.to(device), model, patch_size)

                embedding_path = os.path.join(output_path, f"{uid}.pth")
                torch.save({k: v.cpu() for k, v in features.items()}, embedding_path)

                result_queue.put((uid, True))

                log_entry = {
                    "uid": uid,
                    "path": embedding_path,
                    "shape": list(img.shape),
                    "min": float(img.min()),
                    "max": float(img.max()),
                }
                with log_lock:
                    with open(log_path, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                tf = time.time()
                logger.info(f"Processed {uid} in {tf-t0:.04f} seconds.")

            except Exception as e:
                logger.exception(f"failed on {uid}.")
                result_queue.put((uid, False))

    except KeyboardInterrupt:
        pass
    finally:
        result_queue.cancel_join_thread()
        logger.info(f"Worker GPU {gpu_id} exiting.")


def run_embedding_pipeline(
    series_paths: Sequence[Job],
    output_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    max_patches: int,
    devices: Sequence[GPUId],
    load_fn: LoadFn = autoload_ct,
    do_preprocessing: bool = True,
) -> Tuple[List[SeriesUID], List[SeriesUID]]:
    mp.set_start_method("spawn", force=True)
    os.makedirs(output_path, exist_ok=True)

    existing = {
        os.path.splitext(f)[0] for f in os.listdir(output_path) if f.endswith(".pth")
    }
    to_process = [(uid, path) for uid, path in series_paths if uid not in existing]
    skipped = [uid for uid, _ in series_paths if uid in existing]

    if not to_process:
        print("Nothing to process. All volumes already processed.")
        return skipped, []

    job_queue = mp.Queue()
    result_queue = mp.Queue()
    log_lock = mp.Lock()
    log_path = os.path.join(output_path, "summary.jsonl")

    log_queue = mp.Queue()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(processName)s] %(message)s")
    handler.setFormatter(formatter)

    listener = logging.handlers.QueueListener(log_queue, handler)
    listener.start()

    workers = []
    for gpu_id in devices:
        p = mp.Process(
            target=worker,
            args=(
                gpu_id,
                job_queue,
                result_queue,
                load_fn,
                output_path,
                config_path,
                checkpoint_path,
                max_patches,
                log_path,
                log_lock,
                log_queue,
                do_preprocessing,
            ),
        )
        p.start()
        workers.append(p)

    for job in to_process:
        job_queue.put(job)

    processed = {}
    try:
        while len(processed) < len(to_process):
            try:
                uid, success = result_queue.get(timeout=1)
                processed[uid] = success
            except Empty:
                if not any(p.is_alive() for p in workers):
                    print("All workers have terminated unexpectedly.")
                    break
                continue
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected")
    finally:
        while not job_queue.empty():
            try:
                job_queue.get_nowait()
            except Empty:
                break

        for _ in range(len(workers)):
            try:
                job_queue.put_nowait(None)
            except:
                pass

        for p in workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join()

        job_queue.close()
        result_queue.close()
        job_queue.join_thread()
        result_queue.cancel_join_thread()

    success_uids = skipped + [u for u, ok in processed.items() if ok]
    failed_uids = [u for u, ok in processed.items() if not ok]

    print(f"Skipped:   {len(skipped)}")
    print(f"Succeeded: {len(success_uids) - len(skipped)}")
    print(f"Failed:    {len(failed_uids)}")

    return success_uids, failed_uids
