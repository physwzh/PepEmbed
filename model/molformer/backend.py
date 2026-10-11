"""MoLFormer fragment-encoding backend.

Unified interface (loaded by get_embedding.py via importlib):
    load(model_dir, device, **opts) -> encoder
    encode(encoder, capped_forms, device="cpu", batch_size=64, **opts) -> Tensor[n, 768]

Fragment vector = MoLFormer last-layer [CLS] representation (768-dim).
"""
from __future__ import annotations

from typing import List

import torch

import transformers.masking_utils as _masking_utils
from transformers import AutoModel, AutoTokenizer

MOLFORMER_MAX_LENGTH = 202


def _patch_masking_utils():
    """Inject create_bidirectional_mask into transformers.masking_utils.

    transformers >= 4.57 removed this function; MoLFormer's custom implementation
    still imports it. The replacement converts a [B, seq_len] attention_mask into
    a bias tensor [B,1,1,seq_len] (pad positions -> -1e4).
    """
    if hasattr(_masking_utils, "create_bidirectional_mask"):
        return

    def create_bidirectional_mask(config, inputs_embeds=None, attention_mask=None, **kwargs):
        if attention_mask is None:
            return None
        mask = attention_mask.to(torch.float32)
        ext = (1.0 - mask) * -1e4
        return ext[:, None, None, :]

    _masking_utils.create_bidirectional_mask = create_bidirectional_mask


def _get_hidden_size(config):
    for attr in ("hidden_size", "d_model", "dim", "n_embd"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise AttributeError("Cannot determine hidden_size from model config")


def load(model_dir: str, device: str = "cpu", **opts):
    """Load MoLFormer tokenizer + model and apply the compatibility patch."""
    _patch_masking_utils()
    print(f"[INFO] loading MoLFormer tokenizer/model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    model.eval()
    hidden = _get_hidden_size(model.config)
    max_pos = getattr(model.config, "max_position_embeddings", None)
    print(f"  device={device}, hidden_dim={hidden}, max_position_embeddings={max_pos}")
    return {"tokenizer": tokenizer, "model": model, "hidden": hidden}


@torch.inference_mode()
def encode(encoder, capped_forms: List[str], device: str = "cpu",
           batch_size: int = 64, **opts) -> torch.Tensor:
    """Run MoLFormer over a batch of capped-form SMILES, returning [n, 768] [CLS] reps."""
    n = len(capped_forms)
    hidden = encoder["hidden"]
    if n == 0:
        return torch.empty((0, hidden), dtype=torch.float32)
    features = torch.empty((n, hidden), dtype=torch.float32)
    tokenizer, model = encoder["tokenizer"], encoder["model"]

    for start in range(0, n, batch_size):
        batch = capped_forms[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MOLFORMER_MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        inputs.pop("token_type_ids", None)  # MoLFormer does not accept this field

        # MoLFormer (trust_remote_code custom implementation) has random ops inside
        # forward that are not controlled by eval(): the same input yields different
        # results each time -> h_G jitter. Fix the seed before every forward to make
        # inference fully reproducible.
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        outputs = model(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        end = start + len(batch)
        features[start:end] = pooled.detach().cpu().to(torch.float32)

    return features
