import torch
import torch.nn as nn

from einops import rearrange


class PerceiverResampler(nn.Module):
    def __init__(self, embed_dim, latent_dim, num_queries, num_heads=8, dropout=0.1):
        super().__init__()

        self.num_queries = num_queries

        self.query = nn.Parameter(torch.randn(num_queries, latent_dim))
        self.key = nn.Linear(embed_dim, latent_dim)
        self.value = nn.Linear(embed_dim, latent_dim)

        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tokens of shape (batch_size, seq_len, input_dim).
            mask (torch.Tensor): Attention mask of shape (batch_size, num_queries, seq_len).

        Returns:
            torch.Tensor: Resampled tokens of shape (batch_size, num_queries, latent_dim).
        """
        batch_size, seq_len, _ = x.size()

        q = self.query.unsqueeze(1).repeat(1, batch_size, 1)
        k = self.key(x).transpose(0, 1)
        v = self.value(x).transpose(0, 1)

        attn_output, _ = self.attn(q, k, v, key_padding_mask=mask)

        return attn_output.transpose(0, 1)


class AttentionalPoolProjector(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        output_dim,
        resample_tokens=128,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.resample_tokens = resample_tokens

        self.token_resampler = PerceiverResampler(
            embed_dim=embed_dim,
            latent_dim=hidden_dim,
            num_queries=resample_tokens,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings, mask=None):

        projected_embeddings = []

        for x in embeddings:

            if x is None:
                projected_embeddings.append(None)
                continue

            x = rearrange(x, "t d -> 1 t d")
            x = self.token_resampler(x)
            x = rearrange(x, "1 t d -> t d")
            x = self.norm(x)
            x = self.proj(x)

            projected_embeddings.append(x.squeeze())

        return projected_embeddings
