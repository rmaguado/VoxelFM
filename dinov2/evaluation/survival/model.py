import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, embed_dim)
        return self.mlp(x).squeeze(-1)


class QFormer(nn.Module):
    def __init__(
        self, embed_dim, hidden_dim, num_heads=4, dropout=0.5, attn_dropout=0.1
    ):
        super().__init__()

        self.token_proj = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.queries = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=attn_dropout,
        )

        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, tokens, mask):
        B, M, _ = tokens.shape

        tokens = self.dropout(self.token_proj(tokens))
        Q = self.queries.expand(B, -1, -1)

        key_padding_mask = ~mask
        attn_output, _ = self.attn(Q, tokens, tokens, key_padding_mask=key_padding_mask)
        attn_output = self.dropout(attn_output).squeeze(1)

        logits = self.classifier(attn_output)
        return logits.squeeze(-1)


def build_head(cfg):
    head_cfg = cfg.model.head
    cls = {"mlp": Mlp, "qformer": QFormer}[head_cfg.name]
    return cls(embed_dim=cfg.model.embed_dim, **head_cfg.params)
