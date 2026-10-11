"""Collate: pack variable-length fragment graphs into a dense batch.

MPNN message passing aggregates over node pairs, so collate produces:
  node_features: [B, max_frag, 768]
  node_mask:     [B, max_frag]
  edge_index:    [B, 2, max_edges]   (local indices; the model handles the
                                      within-batch offset)
  edge_attr:     [B, max_edges, 12]
  edge_mask:     [B, max_edges]
  each teacher/target tensor (when present)
"""
from __future__ import annotations

from typing import List, Optional

import torch

# Fragment embedding table [V, 768]. Set by build_dataset; DataLoader workers inherit it via fork.
FRAG_TABLE: Optional[torch.Tensor] = None


def set_frag_table(table: torch.Tensor):
    global FRAG_TABLE
    FRAG_TABLE = table


def collate_fn(samples: List[dict]):
    B = len(samples)
    max_frag = max(s["node_ids"].shape[0] for s in samples)
    node_dim = FRAG_TABLE.shape[1]
    max_edges = max((s["edge_attr"].shape[0] for s in samples), default=0)
    edge_dim = samples[0]["edge_attr"].shape[1] if samples[0]["edge_attr"].numel() else 12

    node_features = torch.zeros((B, max_frag, node_dim), dtype=torch.float32)
    node_mask = torch.zeros((B, max_frag), dtype=torch.bool)
    edge_index = torch.zeros((B, 2, max(max_edges, 1)), dtype=torch.long)
    edge_attr = torch.zeros((B, max(max_edges, 1), edge_dim), dtype=torch.float32)
    edge_mask = torch.zeros((B, max(max_edges, 1)), dtype=torch.bool)

    batch = {
        "global_id": [s["global_id"] for s in samples],
        "node_features": node_features,
        "node_mask": node_mask,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_mask": edge_mask,
        "n_fragments": [s["n_fragments"] for s in samples],
    }

    # Teacher features / targets (packed only when present)
    for key in ("teacher_chem", "teacher_esm", "target_fp", "target_prop"):
        vals = [s.get(key) for s in samples]
        if all(v is not None for v in vals):
            batch[key] = torch.stack(vals, dim=0).to(torch.float32)

    for i, s in enumerate(samples):
        n_frag = s["node_ids"].shape[0]
        n_edge = s["edge_attr"].shape[0]
        node_features[i, :n_frag] = FRAG_TABLE[s["node_ids"]]  # [n_frag, 768]
        node_mask[i, :n_frag] = True
        if n_edge > 0:
            edge_index[i, :, :n_edge] = s["edge_index"]
            edge_attr[i, :n_edge] = s["edge_attr"]
            edge_mask[i, :n_edge] = True

    return batch
