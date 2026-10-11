"""Lazy-loading FragmentDataset (option A): no sample list is built; samples are
read from LMDB one at a time during training.

Compared with dataset.py's build_dataset (which builds 4.68M sample objects up front):
  - Startup only needs to load the caches + filter the teacher set (~1-2s), then
    training starts immediately;
  - Samples are read on demand inside DataLoader workers (LMDB get ~2us +
    pickle ~50us); for a 256-sample batch spread over 8 workers that is ~10ms,
    fully hidden behind the GPU forward/backward;
  - Memory holds only the caches (chem 16GB etc.), no longer the ~5GB sample list.

Contract kept identical to dataset.py:
  - The dict keys returned by __getitem__ are exactly the same as FragmentDataset,
    so collate_fn needs no change;
  - Reuses collate.set_frag_table to register the fragment embedding table;
  - build_lazy_dataset returns (Dataset, stats) with the same stats structure as
    build_dataset.

Filtering strategy (effectively no leakage at the 5.2M scale):
  - Missing teacher/target cache: filtered in one pass via set membership
    (O(n), ~2s), matching build_dataset's per-sample filtering;
  - Missing graph / capped_form not found: Step1 ok=5,196,996/5,197,000, leakage
    ~0, cannot be counted lazily, so __getitem__ raises KeyError directly on a
    miss (prefer failing loudly).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import lmdb
import torch
from torch.utils.data import Dataset

from .collate import set_frag_table
from .dataset import build_edge_index, expand_edge_attr_for_bidirectional
from .edge_features import build_edge_feature_matrix
from .feature_cache import load_fragment_embedding_cache, load_id_feature_cache

# A single path may only be opened once per process (train/val are two Datasets),
# so the handle is cached at module level
_LMDB_ENVS: Dict[str, "lmdb.Environment"] = {}


def _open_lmdb(lmdb_path: str) -> "lmdb.Environment":
    if lmdb_path not in _LMDB_ENVS:
        _LMDB_ENVS[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=2048)
    return _LMDB_ENVS[lmdb_path]


def _forms_hash(form2idx: Dict[str, int]) -> str:
    return hashlib.md5(
        json.dumps(list(form2idx.keys()), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _valid_gids(lmdb_path: str, form2idx: Dict[str, int]) -> Set[str]:
    """Scan the whole LMDB and return the set of gids whose fragments all have a
    capped_smiles that resolves in the lookup table.

    Equivalent to the missing_frag_emb filter of the JSON build_dataset (including
    records with an empty capped_smiles). The result is cached to
    <lmdb_path>.valid_gids.json; subsequent runs load in seconds as long as the
    fragment embedding cache is unchanged.
    """
    cache_path = Path(str(lmdb_path) + ".valid_gids.json")
    h = _forms_hash(form2idx)

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("forms_hash") == h and isinstance(cached.get("gids"), list):
            print(f"  [valid_gids] loaded {len(cached['gids'])} gids from cache", flush=True)
            return set(cached["gids"])

    env = _open_lmdb(lmdb_path)
    valid: Set[str] = set()
    invalid = 0
    last_mark = 0
    t0 = time.time()
    with env.begin() as txn:
        cur = txn.cursor()
        for key, value in cur:
            rec = pickle.loads(value)
            if all(f.get("capped_smiles", "") in form2idx for f in rec["fragments"]):
                valid.add(key.decode("utf-8"))
            else:
                invalid += 1
            if len(valid) - last_mark >= 1_000_000:
                last_mark = len(valid)
                print(f"  [valid_gids] scanned {len(valid):,} valid, {invalid:,} invalid, "
                      f"{time.time()-t0:.0f}s", flush=True)
    cache_path.write_text(json.dumps({"forms_hash": h, "gids": sorted(valid)}))
    print(f"  [valid_gids] done: {len(valid):,} valid, {invalid:,} invalid in "
          f"{time.time()-t0:.0f}s (cached to {cache_path})", flush=True)
    return valid


class LazyFragmentDataset(Dataset):
    def __init__(
        self,
        gids: List[str],
        lmdb_path: str,
        form2idx: Dict[str, int],
        teacher_chem: Optional[Dict[str, torch.Tensor]] = None,
        teacher_esm: Optional[Dict[str, torch.Tensor]] = None,
        target_fp: Optional[Dict[str, torch.Tensor]] = None,
        target_maccs: Optional[Dict[str, torch.Tensor]] = None,
        target_prop: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.gids = gids
        self.form2idx = form2idx
        self.teacher_chem = teacher_chem
        self.teacher_esm = teacher_esm
        self.target_fp = target_fp
        self.target_maccs = target_maccs
        self.target_prop = target_prop
        # read-only, lock-free; each DataLoader worker holds its own handle via fork
        self._env = _open_lmdb(str(lmdb_path))
        self._txn = None  # lazily open one read transaction per worker

    def __len__(self):
        return len(self.gids)

    def _get_txn(self):
        if self._txn is None:
            self._txn = self._env.begin()
        return self._txn

    def __getitem__(self, idx):
        gid = self.gids[idx]
        raw = self._get_txn().get(gid.encode("utf-8"))
        if raw is None:
            raise KeyError(f"gid {gid} not found in LMDB")
        rec = pickle.loads(raw)

        node_ids = torch.tensor(
            [self.form2idx[f["capped_smiles"]] for f in rec["fragments"]], dtype=torch.long
        )
        edge_index = build_edge_index(rec)
        edge_attr = expand_edge_attr_for_bidirectional(
            torch.tensor(build_edge_feature_matrix(rec), dtype=torch.float32)
        )

        # Same as build_dataset: fp (4096) + maccs (167) -> target_fp [4263]
        target_fp = self.target_fp.get(gid) if self.target_fp else None
        target_maccs = self.target_maccs.get(gid) if self.target_maccs else None
        if target_fp is not None and target_maccs is not None:
            target_fp = torch.cat([target_fp, target_maccs], dim=-1)
        elif target_fp is None and target_maccs is not None:
            target_fp = target_maccs

        sample = {
            "global_id": gid,
            "node_ids": node_ids,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "n_fragments": rec["n_fragments"],
            "teacher_chem": self.teacher_chem.get(gid) if self.teacher_chem else None,
            "teacher_esm": self.teacher_esm.get(gid) if self.teacher_esm else None,
            "target_fp": target_fp,
            "target_prop": self.target_prop.get(gid) if self.target_prop else None,
        }
        return sample


def build_lazy_dataset(
    split_gids: List[str],
    graph_dir: str,
    fragment_embedding_cache_path: str,
    chem_cache_path: Optional[str] = None,
    esm_cache_path: Optional[str] = None,
    fp_cache_path: Optional[str] = None,
    maccs_cache_path: Optional[str] = None,
    prop_cache_path: Optional[str] = None,
    drop_missing_graph: bool = True,
    lmdb_path: Optional[str] = None,
):
    """Lazy version of build_dataset; lmdb_path points at convert_shards_to_lmdb.py output."""
    if lmdb_path is None:
        raise ValueError("build_lazy_dataset requires lmdb_path")
    if not Path(lmdb_path).exists():
        raise FileNotFoundError(f"LMDB not found: {lmdb_path}")

    frag_emb = load_fragment_embedding_cache(fragment_embedding_cache_path)
    frag_forms = sorted(frag_emb.keys())
    form2idx = {cf: i for i, cf in enumerate(frag_forms)}
    frag_table = torch.stack([frag_emb[cf] for cf in frag_forms])
    set_frag_table(frag_table)

    chem_cache = load_id_feature_cache(chem_cache_path, "feature") if chem_cache_path else None
    esm_cache = load_id_feature_cache(esm_cache_path, "esm2_feature") if esm_cache_path else None
    fp_cache = load_id_feature_cache(fp_cache_path, "fingerprint") if fp_cache_path else None
    maccs_cache = load_id_feature_cache(maccs_cache_path, "fingerprint") if maccs_cache_path else None
    prop_cache = load_id_feature_cache(prop_cache_path, "descriptor") if prop_cache_path else None

    # One-pass set filtering (equivalent to build_dataset's per-sample filtering)
    stats = {"requested": len(split_gids), "missing_graph": 0, "missing_frag_emb": 0,
             "missing_chem": 0, "missing_esm": 0, "missing_fp": 0, "missing_maccs": 0,
             "missing_prop": 0, "kept": 0}
    gids = list(split_gids)

    # capped_smiles lookup filtering (one full scan, disk-cached)
    valid = _valid_gids(lmdb_path, form2idx)
    before = len(gids)
    gids = [g for g in gids if g in valid]
    stats["missing_frag_emb"] = before - len(gids)

    for name, cache in (("chem", chem_cache), ("esm", esm_cache), ("fp", fp_cache),
                        ("maccs", maccs_cache), ("prop", prop_cache)):
        if cache is None:
            continue
        keys = set(cache.keys())
        before = len(gids)
        gids = [g for g in gids if g in keys]
        stats[f"missing_{name}"] = before - len(gids)
    stats["kept"] = len(gids)

    ds = LazyFragmentDataset(
        gids=gids,
        lmdb_path=lmdb_path,
        form2idx=form2idx,
        teacher_chem=chem_cache,
        teacher_esm=esm_cache,
        target_fp=fp_cache,
        target_maccs=maccs_cache,
        target_prop=prop_cache,
    )
    return ds, stats
