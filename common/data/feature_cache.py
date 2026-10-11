"""Unified reader for the precomputed feature caches (.pt).

Cache formats (produced by the data_preprocessing/ scripts):
  fragment_embedding_cache.pt : {capped_form: List[str], feature: [N_frag, 768]}
  chemberta_smiles_embeddings.pt : {global_id, canonical_smiles, feature:[N,768]}
  esm2_sequence_embeddings.pt : {global_id, original_sequence, esm2_feature:[N,1280]}
  fingerprints_backbone_excluded.pt : {global_id, fingerprint:[N,4096] uint8}
  properties.pt : {global_id, property_names, property:[N,5], property_mean, property_std}
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Option B: process-level memo. (path, feature_key) -> {gid: row-view}.
# In one process, train and val call build_dataloaders twice, which calls
# load_id_feature_cache twice; from the second call on we reuse the first
# dict directly (read-only, the Dataset never mutates it), avoiding loading
# the large table into memory a second time.
_ID_CACHE: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}


def load_fragment_embedding_cache(path) -> Dict[str, torch.Tensor]:
    """Return {capped_form: feature_vector}."""
    data = torch.load(path, map_location="cpu")
    capped_forms = data["capped_form"]
    features = data["feature"]
    return {cf: features[i].to(torch.float32) for i, cf in enumerate(capped_forms)}


def load_id_feature_cache(path, feature_key="feature") -> Dict[str, torch.Tensor]:
    """Read the {global_id, feature_key} format and return {global_id: feature}.

    Option A: returned rows keep the cache's original dtype (e.g. fp uint8),
    instead of calling .to(torch.float32) per row. Each row is only a view
    into feats' shared storage, so there is no per-row [N,4096]*4B copy; the
    actual float32 conversion is done once per batch by collate (uint8 0/1 ->
    fp32 is numerically identical).
    """
    key = (str(path), feature_key)
    if key in _ID_CACHE:
        return _ID_CACHE[key]
    data = torch.load(path, map_location="cpu")
    gids = data["global_id"]
    feats = data[feature_key]
    out = {gid: feats[i] for i, gid in enumerate(gids)}
    _ID_CACHE[key] = out
    return out


def load_fingerprint_cache(path) -> Dict[str, torch.Tensor]:
    """Return {global_id: fingerprint[4096]} float (BCE target)."""
    data = torch.load(path, map_location="cpu")
    gids = data["global_id"]
    fp = data["fingerprint"].to(torch.float32)
    return {gid: fp[i] for i, gid in enumerate(gids)}


def load_property_cache(path):
    """Return ( {gid: prop[5] normalized}, mean, std, names )."""
    data = torch.load(path, map_location="cpu")
    gids = data["global_id"]
    prop = data["property"].to(torch.float32)
    mean = data["property_mean"].to(torch.float32)
    std = data["property_std"].to(torch.float32)
    names = data["property_names"]
    return {gid: prop[i] for i, gid in enumerate(gids)}, mean, std, names


def load_splits(path) -> dict:
    """Read dataset_splits.json and return {'esm2_split':..., 'chem_split':...}."""
    import json
    with open(path) as f:
        return json.load(f)
