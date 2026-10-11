"""FragmentDataset: build one sample per split entry; collate packs them into a batch.

A sample = fragment graph + node indices (collate looks up fragment_embedding_cache
           to get node features) + edge features (12 dims)
           + optional teacher features (chem/esm) + optional targets (fp/prop)

To support the full 5.2M dataset:
  - A sample stores only node indices node_ids (int); the embeddings are gathered
    per batch inside collate, avoiding each sample holding an [n_frag, 768] float
    matrix (4.68M samples x 29 fragments x 768 x 4B ~= 410GB, which would surely OOM).
  - gids are sorted by shard at build time so _load_graph's single-shard cache hits
    (otherwise each sample would re-parse a 20MB shard JSON, which for 4.68M samples
    would take tens of days).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .feature_cache import load_fragment_embedding_cache, load_id_feature_cache


_SHARD_SIZE = 10000
_shard_cache: dict = {"path": None, "data": None}


@dataclass
class FragmentSample:
    global_id: str
    n_fragments: int
    node_ids: torch.Tensor             # [n_frag] long, indexes into collate.FRAG_TABLE
    edge_index: torch.Tensor           # [2, n_edges]
    edge_attr: torch.Tensor            # [n_edges, 12]
    teacher_chem: Optional[torch.Tensor]  # [768] or None
    teacher_esm: Optional[torch.Tensor]   # [1280] or None
    target_fp: Optional[torch.Tensor]     # [4096] or None
    target_prop: Optional[torch.Tensor]   # [5] or None


def _load_graph(graph_dir, gid) -> dict:
    """Load one molecule's graph record from a shard JSON (with single-shard cache)."""
    global _shard_cache
    num = int(gid.split("_")[1])
    shard_idx = num // _SHARD_SIZE
    shard_path = Path(graph_dir) / f"shard_{shard_idx:05d}.json"
    if _shard_cache["path"] != shard_path:
        with open(shard_path) as f:
            _shard_cache["data"] = json.load(f)
        _shard_cache["path"] = shard_path
    return _shard_cache["data"][gid]


class FragmentDataset(Dataset):
    def __init__(self, samples: List[FragmentSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "global_id": s.global_id,
            "node_ids": s.node_ids,
            "edge_index": s.edge_index,
            "edge_attr": s.edge_attr,
            "n_fragments": s.n_fragments,
            "teacher_chem": s.teacher_chem,
            "teacher_esm": s.teacher_esm,
            "target_fp": s.target_fp,
            "target_prop": s.target_prop,
        }


def build_edge_index(graph_record: dict) -> torch.Tensor:
    """Build a [2, n_edges] bidirectional edge index from graph_record edges."""
    edges = graph_record["edges"]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    pairs = []
    for e in edges:
        pairs.append([e["src"], e["dst"]])
        pairs.append([e["dst"], e["src"]])
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def expand_edge_attr_for_bidirectional(edge_attr_single: torch.Tensor) -> torch.Tensor:
    """Edge features one-directional -> bidirectional (duplicate each edge)."""
    if edge_attr_single.shape[0] == 0:
        return edge_attr_single
    return torch.cat([edge_attr_single, edge_attr_single], dim=0)


def build_dataset(
    split_gids: List[str],
    graph_dir: str,
    fragment_embedding_cache_path: str,
    chem_cache_path: Optional[str] = None,
    esm_cache_path: Optional[str] = None,
    fp_cache_path: Optional[str] = None,
    maccs_cache_path: Optional[str] = None,
    prop_cache_path: Optional[str] = None,
    drop_missing_graph: bool = True,
):
    """Build a FragmentDataset from the split's global_id list.

    Stage 2 multi-target:
      - fp_cache (full Morgan 4096) + maccs_cache (MACCS 167) -> concat -> target_fp [4263]
      - prop_cache (PeptideCLM 99 descriptors, fixed version) -> target_prop [99]

    The following samples are filtered out:
      - graph JSON does not exist
      - fragment embedding not found (capped_form not in the fragment library)
      - a teacher/target cache was provided but this global_id is not in the cache
        (chemical/sequence/fingerprint/property)
    """
    graph_dir = Path(graph_dir)

    # Sort by shard: gids of the same shard become adjacent so _load_graph's
    # single-shard cache hits. (train DataLoader uses shuffle=True, so sample
    # order does not affect training.)
    split_gids = sorted(split_gids, key=lambda g: int(g.split("_")[1]) // _SHARD_SIZE)

    frag_emb = load_fragment_embedding_cache(fragment_embedding_cache_path)
    frag_forms = sorted(frag_emb.keys())
    form2idx = {cf: i for i, cf in enumerate(frag_forms)}
    frag_table = torch.stack([frag_emb[cf] for cf in frag_forms])  # [V, 768]

    from .collate import set_frag_table
    set_frag_table(frag_table)

    chem_cache = load_id_feature_cache(chem_cache_path, "feature") if chem_cache_path else None
    esm_cache = load_id_feature_cache(esm_cache_path, "esm2_feature") if esm_cache_path else None
    fp_cache = load_id_feature_cache(fp_cache_path, "fingerprint") if fp_cache_path else None
    maccs_cache = load_id_feature_cache(maccs_cache_path, "fingerprint") if maccs_cache_path else None
    prop_cache = load_id_feature_cache(prop_cache_path, "descriptor") if prop_cache_path else None

    samples = []
    stats = {"requested": len(split_gids), "missing_graph": 0, "missing_frag_emb": 0,
             "missing_chem": 0, "missing_esm": 0, "missing_fp": 0, "missing_maccs": 0,
             "missing_prop": 0, "kept": 0}

    from .edge_features import build_edge_feature_matrix

    for gid in split_gids:
        try:
            graph = _load_graph(graph_dir, gid)
        except (FileNotFoundError, KeyError):
            stats["missing_graph"] += 1
            continue

        # Node indices: store only capped_form -> embedding-table index, avoiding
        # each sample holding an [n_frag, 768] matrix
        node_ids = []
        ok = True
        for frag in graph["fragments"]:
            cf = frag.get("capped_smiles", "")
            idx = form2idx.get(cf)
            if idx is None:
                ok = False
                break
            node_ids.append(idx)
        if not ok:
            stats["missing_frag_emb"] += 1
            continue
        node_ids_tensor = torch.tensor(node_ids, dtype=torch.long)  # [n_frag]

        # Edges: index is bidirectional + attr is computed one-directional then duplicated
        edge_index = build_edge_index(graph)
        edge_attr_single = torch.tensor(build_edge_feature_matrix(graph), dtype=torch.float32)
        edge_attr = expand_edge_attr_for_bidirectional(edge_attr_single)

        # Teacher features: required when the cache is provided; drop the sample if missing
        teacher_chem = chem_cache.get(gid) if chem_cache else None
        teacher_esm = esm_cache.get(gid) if esm_cache else None
        target_fp = fp_cache.get(gid) if fp_cache else None
        target_maccs = maccs_cache.get(gid) if maccs_cache else None
        target_prop = prop_cache.get(gid) if prop_cache else None

        if teacher_chem is None and chem_cache:
            stats["missing_chem"] += 1
            continue
        if teacher_esm is None and esm_cache:
            stats["missing_esm"] += 1
            continue
        if target_fp is None and fp_cache:
            stats["missing_fp"] = stats.get("missing_fp", 0) + 1
            continue
        if target_maccs is None and maccs_cache:
            stats["missing_maccs"] = stats.get("missing_maccs", 0) + 1
            continue
        if target_prop is None and prop_cache:
            stats["missing_prop"] = stats.get("missing_prop", 0) + 1
            continue

        # Concatenate fp (4096) + maccs (167) -> target_fp [4263]
        if target_fp is not None and target_maccs is not None:
            target_fp = torch.cat([target_fp, target_maccs], dim=-1)
        elif target_fp is None and target_maccs is not None:
            target_fp = target_maccs

        samples.append(FragmentSample(
            global_id=gid,
            n_fragments=graph["n_fragments"],
            node_ids=node_ids_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            teacher_chem=teacher_chem,
            teacher_esm=teacher_esm,
            target_fp=target_fp,
            target_prop=target_prop,
        ))
        stats["kept"] += 1

        if stats["kept"] % 200000 == 0:
            print(f"  [build] kept={stats['kept']}/{len(split_gids)}", flush=True)

    return FragmentDataset(samples), stats
