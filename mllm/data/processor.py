import torch
from transformers import PreTrainedTokenizer

from mllm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from mllm.conversation import Conversation


def tokenize_with_image_token(
    text: str,
    tokenizer: PreTrainedTokenizer,
) -> torch.Tensor:
    """
    Tokenizes text as a whole (special tokens added),
    then injects IMAGE_TOKEN_INDEX where <image> was.
    Only one <image> is allowed per input.
    """
    if text.count("<image>") > 1:
        raise ValueError("Only one <image> token is allowed per input.")

    if "<image>" not in text:
        return torch.tensor(
            tokenizer(text, add_special_tokens=True).input_ids, dtype=torch.long
        )

    before, after = text.split("<image>")

    full_text = before + after
    tokenized = tokenizer(full_text, add_special_tokens=True)
    input_ids = tokenized.input_ids

    before_len = len(tokenizer(before, add_special_tokens=False).input_ids)
    input_ids = input_ids[:before_len] + [IMAGE_TOKEN_INDEX] + input_ids[before_len:]

    return torch.tensor(input_ids, dtype=torch.long)


def preprocess_conversation(
    conversation: Conversation,
    tokenizer: PreTrainedTokenizer,
):
    """
    Converts a Conversation into input_ids + labels
    """
    input_ids = []
    labels = []

    for text, is_target in conversation.get_prompt_chunks():
        ids = tokenize_with_image_token(text, tokenizer)
        input_ids.append(ids)

        if is_target:
            labels.append(ids.clone())
        else:
            labels.append(torch.full_like(ids, IGNORE_INDEX))

    return {
        "input_ids": torch.cat(input_ids),
        "labels": torch.cat(labels),
    }
