"""Attention pooling (plan §5).

  α_i = softmax(W_attn · h_i)
  h_G = Σ_i α_i · h_i

Padding nodes are masked out and excluded from the softmax. Returns (h_G, alpha).
"""
from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, node_repr, node_mask):
        # node_repr: [B, max_frag, hidden], node_mask: [B, max_frag]
        scores = self.attn(node_repr).squeeze(-1)  # [B, max_frag]
        scores = scores.masked_fill(~node_mask, float("-inf"))
        # Handle the edge case of an all-padding row (all -inf) -> set to 0 to avoid nan
        all_pad = (~node_mask).all(dim=1, keepdim=True)
        scores = torch.where(all_pad.expand_as(scores), torch.zeros_like(scores), scores)
        alpha = torch.softmax(scores, dim=-1)  # [B, max_frag]
        alpha = torch.nan_to_num(alpha, nan=0.0)
        h_G = torch.bmm(alpha.unsqueeze(1), node_repr).squeeze(1)  # [B, hidden]
        return h_G, alpha
