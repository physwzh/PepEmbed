"""FragmentModel: combines backbone + attention pooling + optional heads.

One class covers all three stages; which heads are attached depends on the
constructor arguments:
  - chem head (proj_chem) / esm head (proj_esm) : for contrastive training
  - fp head / prop head : for Stage 2 prediction

forward returns a dict containing node_repr / pooled(h_G) / each head output /
alpha (for interpretability).
"""
from __future__ import annotations

from typing import Optional

from torch import nn

from .mpnn import MPNNBackbone
from .attention_pooling import AttentionPooling
from .heads import ProjectionHead, FingerprintHead, PropertyHead


class FragmentModel(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 768,
        hidden_dim: int = 256,
        num_mpnn_layers: int = 5,
        edge_dim: int = 12,
        dropout: float = 0.1,
        proj_chem_dim: Optional[int] = None,   # 768
        proj_esm_dim: Optional[int] = None,    # 1280
        fp_bits: Optional[int] = 4096,
        n_props: Optional[int] = 5,
    ):
        super().__init__()
        self.backbone = MPNNBackbone(node_input_dim, hidden_dim, num_mpnn_layers, edge_dim, dropout)
        self.pooling = AttentionPooling(hidden_dim)

        self.proj_chem = ProjectionHead(hidden_dim, proj_chem_dim, dropout=dropout) if proj_chem_dim else None
        self.proj_esm = ProjectionHead(hidden_dim, proj_esm_dim, dropout=dropout) if proj_esm_dim else None
        self.fp_head = FingerprintHead(hidden_dim, fp_bits, dropout) if fp_bits else None
        self.prop_head = PropertyHead(hidden_dim, n_props, dropout) if n_props else None

    def forward(self, batch):
        node_repr, node_mask = self.backbone(
            batch["node_features"], batch["node_mask"],
            batch["edge_index"], batch["edge_attr"], batch["edge_mask"],
        )
        h_G, alpha = self.pooling(node_repr, node_mask)

        out = {"node_repr": node_repr, "h_G": h_G, "alpha": alpha}
        if self.proj_chem is not None:
            out["proj_chem"] = self.proj_chem(h_G)
        if self.proj_esm is not None:
            out["proj_esm"] = self.proj_esm(h_G)
        if self.fp_head is not None:
            out["fp_logits"] = self.fp_head(h_G)
        if self.prop_head is not None:
            out["prop_pred"] = self.prop_head(h_G)
        return out

    def load_backbone_state(self, ckpt_path, strict=True):
        """Load backbone + pooling weights from a checkpoint (1b uses 1a's, 2 uses 1b's).

        Two formats are supported:
          - full .pt (with optimizer/scheduler etc.): take the ["model"] key
          - weights-only .pth: the state_dict itself
        """
        import torch
        is_pth = str(ckpt_path).endswith(".pth")
        if is_pth:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model_state = state
            epoch = None
            metrics = None
        else:
            state = torch.load(ckpt_path, map_location="cpu")
            model_state = state["model"]
            epoch = state.get("epoch")
            metrics = state.get("metrics")
        # Keep only the backbone + pooling weights
        prefix_keys = {k: v for k, v in model_state.items()
                       if k.startswith("backbone.") or k.startswith("pooling.")}
        missing, unexpected = self.load_state_dict(prefix_keys, strict=False)
        if strict and (missing or unexpected):
            # Only care about backbone/pooling-related missing keys
            rel_missing = [m for m in missing if m.startswith("backbone.") or m.startswith("pooling.")]
            if rel_missing:
                raise RuntimeError(f"backbone weights missing: {rel_missing[:5]}")
        return epoch, metrics
