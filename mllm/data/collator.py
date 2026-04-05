from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizer
from mllm.constants import IGNORE_INDEX


@dataclass
class DataCollatorForVisionLanguage:
    tokenizer: PreTrainedTokenizer

    def __call__(self, batch):
        input_ids = [x["input_ids"] for x in batch]
        labels = [x["labels"] for x in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,  # type: ignore
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        image_features = [x["image_features"] for x in batch]
        series_uids = [x["series_uid"] for x in batch]

        if all([x is None for x in image_features]):
            image_features = None

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),  # type: ignore
            "image_features": image_features,
            "series_uids": series_uids,
        }
