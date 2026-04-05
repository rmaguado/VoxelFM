import torch
import torch.nn as nn


class PositionalAttentionRegressor3D(nn.Module):
    def __init__(self, embed_dim, n_heads=4, hidden_dim=128):
        super().__init__()
        self.pos_linear = nn.Linear(3, embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.spatial_attn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        B, Nz, Ny, Nx, D = x.shape
        N = Nz * Ny * Nx
        x = x.view(B, N, D)

        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, Nz, device=x.device),
            torch.linspace(0, 1, Ny, device=x.device),
            torch.linspace(0, 1, Nx, device=x.device),
            indexing="ij",
        )

        pos = torch.stack([grid_z, grid_y, grid_x], dim=-1)
        pos = pos.view(1, N, 3).expand(B, N, 3)

        x = x + self.pos_linear(pos)

        out, _ = self.attn(x, x, x)
        weights = torch.softmax(self.spatial_attn(out), dim=1)

        coords = (weights * pos).sum(dim=1)
        return coords

