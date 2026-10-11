"""MPNN backbone (plan §4).

Message passing:
  m_i^k = Σ_{j∈N(i)} MLP_msg(h_i^k, h_j^k, e_ij)
  h_i^{k+1} = GRU(h_i^k, m_i^k)

Batch handling: collate provides dense [B, max_frag, D] tensors with local edge
indices. This implementation converts each molecule's edges into batch-global
offset indices and does scatter aggregation on one large graph (the padded-node
mask ensures padding does not participate).
"""
from __future__ import annotations

import torch
from torch import nn


def _batch_global_edges(edge_index, edge_mask, max_frag):
    """Convert [B,2,max_edges] local edge indices into a flat edge table of
    global node indices.

    Returns (src_flat, dst_flat, batch_id), length = number of valid edges.
    Global index = batch_id * max_frag + local_idx.
    """
    B = edge_index.shape[0]
    valid = edge_mask  # [B, max_edges]
    # batch offset
    offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * max_frag  # [B,1]
    src = edge_index[:, 0, :] + offsets   # [B, max_edges]
    dst = edge_index[:, 1, :] + offsets
    src_flat = src[valid]
    dst_flat = dst[valid]
    batch_ids = torch.arange(B, device=edge_index.device).unsqueeze(1).expand_as(valid)[valid]
    return src_flat, dst_flat, batch_ids


class MPNNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 edge_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, node_mask, edge_index, edge_attr, edge_mask):
        """node_features: [B, max_frag, input_dim] -> node_repr [B, max_frag, hidden].
        Returns (node_repr, node_mask).
        """
        B, max_frag, _ = node_features.shape
        h = self.node_proj(node_features)  # [B, max_frag, hidden]

        # Flatten into one large graph
        h_flat = h.reshape(B * max_frag, -1)  # [B*max_frag, hidden]
        src_flat, dst_flat, _ = _batch_global_edges(edge_index, edge_mask, max_frag)
        edge_attr_flat = edge_attr[edge_mask]  # [n_valid_edges, edge_dim]

        for layer in self.layers:
            h_flat = layer(h_flat, src_flat, dst_flat, edge_attr_flat)

        h_flat = self.output_norm(h_flat)
        node_repr = h_flat.reshape(B, max_frag, -1)
        return node_repr, node_mask


class MPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h, src, dst, edge_attr):
        # Message: MLP_msg(h_src, h_dst, e_{src->dst})
        msg_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)  # [n_edges, hidden]
        # Aggregate into dst (sum)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        # GRU update
        h = self.gru(agg, h)
        return h
