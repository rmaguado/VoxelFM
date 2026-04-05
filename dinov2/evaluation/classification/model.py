import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model


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


class BackboneAdapter(nn.Module):
    def forward(self, x) -> torch.Tensor:
        """Return (B, D) pooled features"""
        raise NotImplementedError


class DinoV2Adapter(BackboneAdapter):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        return out["x_norm_clstoken"]


class LoRABackboneWithHead(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()

        assert cfg.model.head.name == "mlp", "Head must be MLP for LoRABackboneWithHead"

        self.backbone = backbone

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        self.head = build_head(cfg)

    def forward(self, x):
        """
        x: volume tensor (B, C, H, W)
        """
        out = self.backbone(x)

        return self.head(out)


def build_head(cfg):
    head_cfg = cfg.model.head
    cls = {"mlp": Mlp, "qformer": QFormer}[head_cfg.name]
    return cls(embed_dim=cfg.model.embed_dim, **head_cfg.params)
