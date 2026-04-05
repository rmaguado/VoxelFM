import os
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class EmbeddingLocalizationDataset(Dataset):
    """
    Dataset for precomputed patch/grid embeddings.
    Each sample returns:
        x: (Nz, Ny, Nx, D)
        y: (3,) relative coordinates in [0, 1]
    """

    def __init__(
        self,
        uids: List[str],
        coords: List[Tuple[float, float, float]],
        embeddings_path: str,
        feature_key: str = "patch",
        noise_p: float = 0.0,
        noise_sigma: float = 0.1,
    ):
        self.uids = uids
        self.coords = coords
        self.embeddings_path = embeddings_path
        self.feature_key = feature_key
        self.noise_p = noise_p
        self.noise_sigma = noise_sigma

    def __len__(self):
        return len(self.uids)

    def _load_embedding(self, uid: str) -> torch.Tensor:
        data = torch.load(os.path.join(self.embeddings_path, f"{uid}.pth"))
        x = data[self.feature_key]  # (Nz, Ny, Nx, D)

        if self.noise_p > 0 and torch.rand(1).item() < self.noise_p:
            x = x + torch.randn_like(x) * self.noise_sigma

        return x.float()

    def __getitem__(self, idx):
        x = self._load_embedding(self.uids[idx])
        y = torch.tensor(self.coords[idx], dtype=torch.float32)
        return x, y
