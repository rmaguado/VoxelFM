import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List


class SegmentationEmbeddingDataset(Dataset):
    """
    Each sample:
      x: (Nz, Ny, Nx, D) patch embeddings
      y: (Z, Y, X) voxel mask
    """

    def __init__(
        self,
        uids: List[str],
        embeddings_path: str,
        masks_path: str,
        feature_key: str = "patch",
        noise_p: float = 0.0,
        noise_sigma: float = 0.1,
    ):
        self.uids = uids
        self.embeddings_path = embeddings_path
        self.masks_path = masks_path
        self.feature_key = feature_key
        self.noise_p = noise_p
        self.noise_sigma = noise_sigma

    def __len__(self):
        return len(self.uids)

    def _load_embedding(self, uid):
        data = torch.load(os.path.join(self.embeddings_path, f"{uid}.pth"))
        x = data[self.feature_key]  # (Nz, Ny, Nx, D)

        if self.noise_p > 0 and torch.rand(1).item() < self.noise_p:
            x = x + torch.randn_like(x) * self.noise_sigma

        return x.float()

    def _load_mask(self, uid):
        mask = np.load(os.path.join(self.masks_path, f"{uid}.npy"))
        return torch.from_numpy(mask).long()

    def __getitem__(self, idx):
        uid = self.uids[idx]
        x = self._load_embedding(uid)
        y = self._load_mask(uid)
        return x, y
