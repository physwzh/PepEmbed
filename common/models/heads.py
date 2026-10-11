"""Projection heads / prediction heads.

ProjectionHead: h_G -> teacher space (chem 768 / esm 1280), for contrastive.
FingerprintHead: h_G -> 4096 logits, BCE multi-label.
PropertyHead: h_G -> 5, MSE regression.
"""
from __future__ import annotations

from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class FingerprintHead(nn.Module):
    """h_G -> 4096 logits."""
    def __init__(self, input_dim: int, n_bits: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, n_bits),
        )

    def forward(self, x):
        return self.net(x)


class PropertyHead(nn.Module):
    """h_G -> 5."""
    def __init__(self, input_dim: int, n_props: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, n_props),
        )

    def forward(self, x):
        return self.net(x)
