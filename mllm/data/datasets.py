import os
import json
import copy
import torch
from torch.utils.data import Dataset
from einops import rearrange

from mllm.data.processor import preprocess_conversation
from mllm.conversation import create_conversation


class VLMDataset(Dataset):
    def __init__(
        self,
        conversations_path: str,
        embeddings_path: str,
        tokenizer,
        system_prompt,
        feature_key: str = "cls",
        pooling: str = "none",
        noise_p: float = 0.0,
        noise_sigma: float = 0.1,
    ):
        with open(conversations_path, "r") as f:
            conversations = json.load(f)

        self.embeddings_path = embeddings_path
        embed_series_uids = [
            x.replace(".pth", "")
            for x in os.listdir(embeddings_path)
            if x.endswith(".pth")
        ]

        conv_img = [c for c in conversations if "image" in c]
        conv_img = [c for c in conv_img if c.get("image") in embed_series_uids]
        conv_text = [c for c in conversations if "image" not in c]
        self.conversations = conv_img + conv_text

        dropped = len(conversations) - len(self.conversations)
        if dropped > 0:
            print(
                f"[RadiologyReportDataset] Dropped {dropped} samples with missing embeddings"
            )

        self.tokenizer = tokenizer
        self.conversation = create_conversation(
            tokenizer, system_prompt, do_generation=True
        )
        self.feature_key = feature_key
        self.pooling = pooling
        self.noise_p = noise_p
        self.noise_sigma = noise_sigma

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.conversations:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.conversations)

    def _load_embedding(self, uid):
        x = torch.load(os.path.join(self.embeddings_path, f"{uid}.pth"), mmap=True)

        if self.feature_key != "none":
            x = x[self.feature_key]

        if x.ndim > 1:
            x = rearrange(x, "... d -> (...) d")
            if self.pooling == "avg":
                x = x.mean(dim=0, keepdim=True)

        return x.contiguous()

    def __getitem__(self, idx):
        data = self.conversations[idx]
        series_uid = data.get("image")
        conversation = data["conversations"]

        if series_uid:
            image_features = self._load_embedding(series_uid)
            if self.noise_p > 0 and torch.rand(1).item() < self.noise_p:
                image_features += torch.randn_like(image_features) * self.noise_sigma
        else:
            image_features = None

        conv = copy.deepcopy(self.conversation)
        conv.reset()
        for idx, message in enumerate(conversation):
            role = message["from"]
            content = message["value"]
            conv.append_message(role, content)

        data = preprocess_conversation(conv, self.tokenizer)

        return {
            "input_ids": data["input_ids"],
            "labels": data["labels"],
            "image_features": image_features,
            "series_uid": series_uid,
        }
