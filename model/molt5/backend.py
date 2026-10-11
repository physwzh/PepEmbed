"""MolT5 fragment-encoding backend.

Unified interface (loaded by get_embedding.py via importlib):
    load(model_dir, device, **opts) -> encoder
    encode(encoder, capped_forms, device="cpu", batch_size=128, **opts) -> Tensor[n, 768]

Fragment vector = mask-mean pooling of the T5 encoder output (768-dim); T5 has no
[CLS] token, so the mean is taken over non-pad positions.
"""
from __future__ import annotations

from typing import List

import torch

MOLT5_MAX_LENGTH = 512


def load(model_dir: str, device: str = "cpu", **opts):
    """Load the MolT5 tokenizer + T5 encoder."""
    from transformers import T5EncoderModel, T5Tokenizer

    print(f"[INFO] loading MolT5 tokenizer/encoder from {model_dir}")
    tokenizer = T5Tokenizer.from_pretrained(model_dir, model_max_length=MOLT5_MAX_LENGTH)
    model = T5EncoderModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    hidden = getattr(model.config, "d_model", None) or getattr(model.config, "hidden_size", None)
    n_layers = getattr(model.config, "num_layers", None)
    print(f"  device={device}, d_model={hidden}, num_layers={n_layers}")
    return {"tokenizer": tokenizer, "model": model, "hidden": hidden}


@torch.inference_mode()
def encode(encoder, capped_forms: List[str], device: str = "cpu",
           batch_size: int = 128, **opts) -> torch.Tensor:
    """Run the T5 encoder over a batch of capped-form SMILES, returning [n, 768] mask-mean reps."""
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
            max_length=MOLT5_MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        # T5 is a standard HF implementation, deterministic under eval() +
        # inference_mode(); still fix the seed to match the MoLFormer variant's
        # style and guarantee fully reproducible inference.
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        hidden_states = outputs.last_hidden_state                     # [B, L, 768]
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        end = start + len(batch)
        features[start:end] = pooled.detach().cpu().to(torch.float32)

    return features
