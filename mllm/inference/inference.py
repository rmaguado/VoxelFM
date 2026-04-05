import os
import json
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import torch
import torch.multiprocessing as mp
import copy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from mllm.models.vl_model import VisionLanguageModel
from mllm.models.injectors import ImageTokenInjector
from mllm.models.projectors import AttentionalPoolProjector
from mllm.data import VLMDataset, preprocess_conversation


class ReportsDataset(VLMDataset):
    def __init__(
        self,
        conversations_path: str,
        embeddings_path: str,
        tokenizer,
        system_prompt,
        feature_key: str,
        pooling: str,
    ):
        super().__init__(
            conversations_path,
            embeddings_path,
            tokenizer,
            system_prompt,
            feature_key,
            pooling,
        )

    def __getitem__(self, idx):
        data = self.conversations[idx]
        series_uid = data.get("image")
        conversation = data["conversations"]

        assert (
            len(conversation) == 2
        ), "Conversation should have 2 messages: user request and radiology report. "

        image_features = self._load_embedding(series_uid)

        conv = copy.deepcopy(self.conversation)
        conv.reset()

        role = conversation[0]["from"]
        content = conversation[0]["value"]
        assert role == "human", "first message role must be human"
        conv.append_message(role, content)
        data = preprocess_conversation(conv, self.tokenizer)

        ground_truth = conversation[1]["value"]

        return {
            "input_ids": data["input_ids"],
            "image_features": image_features,
            "series_uid": series_uid,
            "ground_truth": ground_truth,
        }


class ReportGenerator:
    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: str,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
        rank: int = 0,
    ):
        """
        Initialize the report generator with a trained checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory containing model files
            device: Device to run inference on
            dtype: Data type for model weights
            rank: Process rank for DDP
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.rank = rank

        self.print(f"Loading model from {checkpoint_path}")
        self._load_model()

    def print(self, msg):
        if self.rank == 0:
            print(msg, flush=True)

    def _load_model(self):
        """Load the full VisionLanguageModel from checkpoint"""
        base_model_path = self.config.model.model_name_or_path
        mm_vision_hidden_size = self.config.model.mm_vision_hidden_size
        mm_projector_hidden_size = self.config.model.mm_projector_hidden_size
        image_tokens = self.config.model.image_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.print(f"Loading base model: {base_model_path}")

        lm = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        )

        lora_path = os.path.join(self.checkpoint_path, "lora_adapters_dir")
        if os.path.exists(lora_path):
            self.print("Loading LoRA adapters")
            lm = PeftModel.from_pretrained(lm, lora_path)
            lm = lm.merge_and_unload()  # type: ignore
        else:
            self.print("Found no lora adapters.")

        mm_projector = AttentionalPoolProjector(
            embed_dim=mm_vision_hidden_size,
            hidden_dim=mm_projector_hidden_size,
            output_dim=lm.config.hidden_size,
            resample_tokens=image_tokens,
        )

        projector_path = os.path.join(self.checkpoint_path, "mm_projector.pth")
        if os.path.exists(projector_path):
            self.print("Loading projector weights")
            state = torch.load(projector_path, map_location="cpu")
            mm_projector.load_state_dict(state)
        else:
            raise ValueError(f"Projector weights not found at {projector_path}")

        injector = ImageTokenInjector()
        self.model = VisionLanguageModel(
            lm=lm, mm_projector=mm_projector, injector=injector
        )
        self.model.to(dtype=self.dtype, device=self.device)

        self.model.eval()

        self.print("Model loaded successfully")

    @torch.no_grad()
    def generate_report(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        generation_config: dict,
    ) -> str:
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        image_features = image_features.to(device=self.device, dtype=self.dtype)
        model = self.model

        with torch.inference_mode():
            output_ids = model.generate(  # type: ignore
                input_ids=input_ids,
                image_features=[image_features],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_config,
            )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_text.strip()


def get_generation_config(args) -> dict:
    configs = {
        "greedy": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "early_stopping": False,
        },
        "nucleus": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "num_beams": 1,
            "early_stopping": False,
        },
        "beam": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 4,
            "early_stopping": True,
        },
    }
    if args.strategy == "none":
        config = configs["nucleus"]
    else:
        if args.strategy not in configs:
            raise ValueError(
                f"Unknown strategy '{args.strategy}'. Choose from: {list(configs.keys())}"
            )
        config = configs[args.strategy]

    if args.temperature is not None:
        config["temperature"] = args.temperature

    if args.top_p is not None:
        config["top_p"] = args.top_p
        config["do_sample"] = True

    if args.top_k is not None:
        config["top_k"] = args.top_k
        config["do_sample"] = True

    if args.num_beams is not None:
        config["num_beams"] = args.num_beams
        config["early_stopping"] = args.num_beams > 1

    return config


def shard_indices(dataset_len: int, rank: int, world_size: int):
    return list(range(rank, dataset_len, world_size))


def run_inference_worker(
    rank: int,
    world_size: int,
    config: DictConfig,
    checkpoint_path: str,
    eval_conversations_path: str,
    embeddings_path: str,
    output_dir: str,
    generation_config: dict,
):
    torch.cuda.set_device(rank)

    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    generator = ReportGenerator(
        config=config,
        checkpoint_path=checkpoint_path,
        device=f"cuda:{rank}",
        dtype=torch.bfloat16,
        rank=rank,
    )

    dataset = ReportsDataset(
        conversations_path=eval_conversations_path,
        embeddings_path=embeddings_path,
        tokenizer=generator.tokenizer,
        system_prompt=config.data.system_prompt,
        feature_key=config.data.feature_key,
        pooling=config.data.pooling,
    )

    indices = shard_indices(len(dataset), rank, world_size)

    results = []

    if rank == 0:
        pbar = tqdm(total=len(indices), desc="Generating reports")
    else:
        pbar = None

    for idx in indices:
        batch = dataset[idx]

        input_ids = batch["input_ids"]
        series_uid = batch["series_uid"]
        ground_truth = batch["ground_truth"]
        image_features = batch["image_features"]

        try:
            generated_report = generator.generate_report(
                input_ids=input_ids,
                image_features=image_features,
                generation_config=generation_config,
            )
        except Exception as e:
            if rank == 0:
                print(f"Error generating for {series_uid}: {e}", flush=True)
            generated_report = "[ERROR]"

        results.append(
            {
                "series_uid": series_uid,
                "ground_truth": ground_truth,
                "generated": generated_report,
            }
        )

        with open(reports_dir / f"{series_uid}.txt", "w") as f:
            f.write(generated_report)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    gpu_results_file = Path(output_dir) / f"results_gpu_{rank}.json"
    with open(gpu_results_file, "w") as f:
        json.dump(results, f, indent=2)


def merge_results(output_dir: str, world_size: int):
    all_results = []

    for rank in range(world_size):
        gpu_results_file = Path(output_dir) / f"results_gpu_{rank}.json"
        if gpu_results_file.exists():
            with open(gpu_results_file, "r") as f:
                results = json.load(f)
                all_results.extend(results)

    json_file = Path(output_dir) / "all_results.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)

    results_df = pd.DataFrame(all_results)
    csv_file = Path(output_dir) / "all_results.csv"
    results_df.to_csv(csv_file, index=False, sep=";")


def run_inference(
    config: DictConfig,
    checkpoint_path: str,
    eval_conversations_path: str,
    embeddings_path: str,
    output_dir: str,
    generation_config: dict,
    num_gpus: int = 1,
):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    config_save_path = output_dir_path / "generation_config.json"
    with open(config_save_path, "w") as f:
        json.dump(generation_config, f, indent=2)

    print(f"Running multi-GPU inference on {num_gpus} GPUs", flush=True)

    if num_gpus > 1:

        mp.spawn(  # type: ignore
            run_inference_worker,
            args=(
                num_gpus,
                config,
                checkpoint_path,
                eval_conversations_path,
                embeddings_path,
                output_dir,
                generation_config,
            ),
            nprocs=num_gpus,
            join=True,
        )

    else:
        run_inference_worker(
            rank=0,
            world_size=num_gpus,
            config=config,
            checkpoint_path=checkpoint_path,
            eval_conversations_path=eval_conversations_path,
            embeddings_path=embeddings_path,
            output_dir=output_dir,
            generation_config=generation_config,
        )

    print("Merging results from all GPUs...", flush=True)
    merge_results(output_dir, num_gpus)


def main():
    parser = argparse.ArgumentParser(
        description="Generate radiology reports from eval dataset"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to model config (e.g., outputs/config.yaml)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint directory (e.g., outputs/checkpoints/checkpoint-1000)",
    )
    parser.add_argument(
        "--eval_conversations_path",
        type=str,
        required=True,
        help="Path to eval .json file",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to embeddings directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated reports",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for inference",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "nucleus", "beam", "none"],
        help="Generation strategy: greedy (deterministic), nucleus (sampling), beam (beam search), or none (custom)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for sampling (only used with --strategy=none)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p for nucleus sampling (only used with --strategy=none)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k for sampling (only used with --strategy=none)",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams for beam search (only used with --strategy=none)",
    )

    args = parser.parse_args()

    config: DictConfig = OmegaConf.load(file_=args.config_path)  # type: ignore

    generation_config = get_generation_config(args)

    run_inference(
        config=config,
        checkpoint_path=args.checkpoint_path,
        eval_conversations_path=args.eval_conversations_path,
        embeddings_path=args.embeddings_path,
        output_dir=args.output_dir,
        generation_config=generation_config,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
