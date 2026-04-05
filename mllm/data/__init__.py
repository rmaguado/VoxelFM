from typing import Dict
from transformers import PreTrainedTokenizer

from .datasets import VLMDataset
from .processor import preprocess_conversation
from .collator import DataCollatorForVisionLanguage
from mllm.configs import DataArguments


def make_datasets(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
) -> Dict:
    train_dataset = VLMDataset(
        conversations_path=data_args.train_conversations_path,
        embeddings_path=data_args.embeddings_path,
        tokenizer=tokenizer,
        system_prompt=data_args.system_prompt,
        feature_key=data_args.feature_key,
        pooling=data_args.pooling,
        noise_p=data_args.noise_p,
        noise_sigma=data_args.noise_sigma,
    )

    eval_dataset = VLMDataset(
        conversations_path=data_args.eval_conversations_path,
        embeddings_path=data_args.embeddings_path,
        tokenizer=tokenizer,
        system_prompt=data_args.system_prompt,
        feature_key=data_args.feature_key,
        pooling=data_args.pooling,
        noise_p=data_args.noise_p,
        noise_sigma=data_args.noise_sigma,
    )

    data_collator = DataCollatorForVisionLanguage(tokenizer=tokenizer)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
