"""Checkpoint-compatible image encoder and projection head from the LeWM repo.

These definitions mirror the user's ``models/encoder.py`` and the
``ProjectionHead`` in ``models/predictor2.py``. Keeping the small inference-time
subset here lets the SO-101 project load LeWM weights without cloning the whole
LeWM training repository onto every training machine.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch: int = 16,
        in_ch: int = 3,
        dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        out_dim: int = 192,
    ):
        super().__init__()
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(
            in_ch, dim, kernel_size=patch, stride=patch
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos
        x = self.transformer(x)
        x = self.norm(x)
        return self.proj_bn(self.proj(x[:, 0]))


class ProjectionHead(nn.Module):
    def __init__(self, dim: int = 192, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = self.net(x.reshape(-1, shape[-1]))
        return x.reshape(*shape[:-1], -1)

