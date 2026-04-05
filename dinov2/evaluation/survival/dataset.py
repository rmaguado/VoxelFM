import os
import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler
import torchvision.transforms as transforms
from typing import Literal, List

from einops import rearrange


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        uids: List[str],
        times: List[float],
        events: List[bool],
        embeddings_path: str,
        feature_key: str = "clsn",
        pooling: Literal["avg", "none"] = "none",
        noise_p: float = 0.0,
        noise_sigma: float = 0.1,
    ):
        self.uids = uids
        self.times = times
        self.events = events
        self.embeddings_path = embeddings_path
        self.feature_key = feature_key
        self.pooling = pooling
        self.noise_p = noise_p
        self.noise_sigma = noise_sigma

    def __len__(self) -> int:
        return len(self.uids)

    def _load_embedding(self, uid: str) -> torch.Tensor:
        path = os.path.join(self.embeddings_path, f"{uid}.pth")
        try:
            data = torch.load(path, mmap=True)
        except OSError as e:
            raise Exception(f"Failed to load embedding: {path}. {e}")

        x = data[self.feature_key] if self.feature_key != "none" else data

        if x.ndim > 1:
            x = rearrange(x, "... d -> (...) d")
            if self.pooling == "avg":
                x = x.mean(dim=0, keepdim=True)

        return x.contiguous()

    def __getitem__(self, idx: int):
        x = self._load_embedding(self.uids[idx])
        time = torch.tensor(self.times[idx], dtype=torch.float32)
        event = torch.tensor(self.events[idx], dtype=torch.float32)

        if self.noise_p > 0 and torch.rand(1).item() < self.noise_p:
            x = x + torch.randn_like(x) * self.noise_sigma

        return x, time, event


class StratifiedBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size):
        self.labels = np.asarray(labels, dtype=bool)
        self.batch_size = batch_size
        assert batch_size % 2 == 0
        self.half_bs = batch_size // 2
        self.pos = np.where(self.labels)[0]
        self.neg = np.where(~self.labels)[0]
        self.num_batches = min(len(self.pos), len(self.neg)) // self.half_bs

        if self.num_batches == 0:
            raise ValueError(
                "Not enough samples per class for the requested batch size."
            )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        pos = np.random.permutation(self.pos)
        neg = np.random.permutation(self.neg)
        for i in range(self.num_batches):
            batch = np.concatenate(
                [
                    pos[i * self.half_bs : (i + 1) * self.half_bs],
                    neg[i * self.half_bs : (i + 1) * self.half_bs],
                ]
            )
            np.random.shuffle(batch)
            yield batch.tolist()


def collate_fn_pad(batch):
    patches_list, times, events = zip(*batch)
    times = torch.stack(times).float()
    events = torch.stack(events).float()
    lengths = [p.shape[0] for p in patches_list]
    max_len = max(lengths)
    D = patches_list[0].shape[1]
    batch_size = len(patches_list)

    padded = torch.zeros((batch_size, max_len, D), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, p in enumerate(patches_list):
        L = p.shape[0]
        padded[i, :L, :] = p
        mask[i, :L] = True
    return padded, mask, times, events
