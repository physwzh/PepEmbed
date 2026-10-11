"""MoLeR fragment-encoding backend (512 latent + chirality c(8) + residue kind t(6) = 526-dim).

Unified interface (loaded by get_embedding.py via importlib):
    load(model_dir, device=None, workers=8, **opts) -> encoder (MolToMoler526)
    encode(encoder, capped_forms, **opts) -> Tensor[n, 526]

MoLeR's TF inference can only run on CPU; the encoder masks the GPU internally via
`moler_cpu_session()`, restores the env vars on exit, and torch can still use CUDA
afterwards.
"""
from __future__ import annotations

from typing import List

import torch

from moler_526 import (  # noqa: E402
    DEFAULT_NUM_WORKERS,
    FEATURE_DIM,
    MolToMoler526,
    load_moler_encoder,
)


def load(model_dir: str, device: str = "cpu", workers: int = DEFAULT_NUM_WORKERS, **opts):
    """Construct the MoLeR 526-dim fragment encoder (CPU)."""
    print(f"[INFO] loading MoLeR model from {model_dir} (workers={workers})")
    enc = load_moler_encoder(model_dir=model_dir, num_workers=workers)
    print(f"  latent={enc.latent_size}, full_dim={enc.full_size}")
    return enc


def encode(encoder: MolToMoler526, capped_forms: List[str], **opts) -> torch.Tensor:
    """Run MoLeR over a batch of capped-form SMILES, returning [n, 526] [z(512);c(8);t(6)]."""
    n = len(capped_forms)
    if n == 0:
        return torch.empty((0, FEATURE_DIM), dtype=torch.float32)
    features = encoder.encode(list(capped_forms))
    if features.shape != (n, FEATURE_DIM):
        raise RuntimeError(
            f"MoLeR encoder returned {tuple(features.shape)}, expected {(n, FEATURE_DIM)}"
        )
    return features.to(torch.float32)
