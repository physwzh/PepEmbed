#!/usr/bin/env python3
"""Unified fragment/graph-level embedding entry (SMILES -> h_G and other representations).

The three pretrained projects (full_split_molformer / full_split_MolT5 /
full_split_MoLeR) share an identical pretraining pipeline; the only difference is
the **fragment initial encoding** layer. This script switches between the three
via `--method`, while all other logic (SMILES canonicalisation -> fragment
decomposition -> graph tensor assembly -> GNN forward) is shared. Each method's
encoding backend lives in `model/<method>/backend.py`; directories / weights /
hyperparameters are defined in `config.yaml`.

Usage:
    # Single SMILES (default method taken from config.default_method)
    python get_embedding.py "C(C(=O)NCC(=O)O)N"
    python get_embedding.py "C(C(=O)NCC(=O)O)N" --method moler

    # CSV input, save a subset of outputs
    python get_embedding.py --csv input.csv --method molt5 \
        --output h_G,alpha,fragment_meta --save embeddings.pt

    # Show all available outputs for a single SMILES
    python get_embedding.py "C(C(=O)NCC(=O)O)N" --output all
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ---- path setup ----
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import split_fragment as sf
from common.models import FragmentModel
from common.data.edge_features import build_edge_feature_matrix
from common.data.dataset import build_edge_index, expand_edge_attr_for_bidirectional

DEFAULT_CONFIG = str(PROJECT_ROOT / "config.yaml")


# ======================================================================
#  Config / backend loading
# ======================================================================

def load_config(config_path: str) -> dict:
    """Read config.yaml and resolve relative paths into absolute paths under the project root."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = PROJECT_ROOT
    if "common_sets" in cfg:
        cfg["_common_sets_abs"] = str((base / cfg["common_sets"]).resolve())
    for method, m in cfg.get("methods", {}).items():
        for key in ("backend", "model_dir", "ckpt"):
            if key in m:
                m[f"_{key}_abs"] = str((base / m[key]).resolve())
    return cfg


def _load_backend(backend_path: str):
    """Load the encoding backend module by file path and add its directory to
    sys.path (a backend may import a sibling self-contained module, e.g.
    model/moler/moler_526.py)."""
    backend_dir = str(Path(backend_path).resolve().parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    spec = importlib.util.spec_from_file_location(
        f"_embed_backend_{Path(backend_path).stem}", backend_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backend_opts(m: dict) -> dict:
    """Return the hyperparameters to forward to backend.load / backend.encode."""
    reserved = {"desc", "backend", "model_dir", "ckpt", "node_input_dim",
                "_backend_abs", "_model_dir_abs", "_ckpt_abs"}
    return {k: v for k, v in m.items() if k not in reserved}


# ======================================================================
#  FragmentModel loader
# ======================================================================

def load_fragment_model(ckpt_path: str, device: str, node_input_dim: int,
                        hidden_dim: int = 1024, num_mpnn_layers: int = 5,
                        edge_dim: int = 12, dropout: float = 0.1) -> FragmentModel:
    """Load a trained FragmentModel from a weights-only .pth checkpoint."""
    model = FragmentModel(
        node_input_dim=node_input_dim,
        hidden_dim=hidden_dim,
        num_mpnn_layers=num_mpnn_layers,
        edge_dim=edge_dim,
        dropout=dropout,
        proj_chem_dim=None,
        proj_esm_dim=None,
        fp_bits=None,
        n_props=None,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    n_loaded = len(state_dict)
    n_matched = sum(1 for k in state_dict if k in model.state_dict())
    print(f"  matched {n_matched}/{n_loaded} checkpoint keys (heads skipped as expected)")
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] loaded FragmentModel from {Path(ckpt_path).name}")
    print(f"  node_input_dim={node_input_dim}, hidden_dim={hidden_dim}, num_params={n_params:,}")
    return model


# ======================================================================
#  Common sets (for fragment decomposition Tier-2 / Tier-3)
# ======================================================================

def load_common_sets(json_path: str):
    """Load common_residues / common_linkers sets from common_sets.json."""
    with open(json_path) as f:
        data = json.load(f)
    common_residues = set(data["common_residues"])
    common_linkers = set(data["common_linkers"])
    print(f"[INFO] common_sets: {len(common_residues)} residues, {len(common_linkers)} linkers")
    return common_residues, common_linkers


# ======================================================================
#  SMILES canonicalisation
# ======================================================================

def canonicalize_smiles(smiles: str) -> str:
    """RDKit canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


# ======================================================================
#  Core: encode fragments via the selected backend
# ======================================================================

def _encode_fragments(backend, encoder, capped_forms: List[str], device: str,
                      encode_opts: dict) -> torch.Tensor:
    """Call the selected backend's encode; return [n_frag, node_input_dim] fragment features."""
    return backend.encode(encoder, capped_forms, device=device, **encode_opts)


# ======================================================================
#  Core: build graph tensors from a decomposed molecule
# ======================================================================

def _build_sample_tensors(
    fragments: List[sf.Fragment],
    edges: list,
    frag_embeddings: torch.Tensor,
    original_smiles: str,
) -> dict:
    """Build per-sample tensors (NO batch dim) for collate_fn.

    Returns keys: node_features, node_mask, edge_index, edge_attr, edge_mask, n_fragments.
      node_features: [n_frag, node_input_dim]
      node_mask:     [n_frag]  (all True)
      edge_index:    [2, n_edges*2] or [2, 0]
      edge_attr:     [n_edges*2, 12] or [0, 12]
      edge_mask:     [n_edges*2]  (all True) or None (no edges)
    """
    graph_record = sf.to_graph_record(fragments, edges, original_smiles)
    n_frag = graph_record["n_fragments"]

    node_features = frag_embeddings                     # [n_frag, node_input_dim]
    node_mask = torch.ones(n_frag, dtype=torch.bool)

    edge_index = build_edge_index(graph_record)          # [2, n_edges*2] or [2, 0]
    edge_attr_single = torch.tensor(
        build_edge_feature_matrix(graph_record), dtype=torch.float32,
    )  # [n_edges, 12] or [] for edgeless
    if edge_attr_single.dim() == 1:
        edge_attr_single = edge_attr_single.reshape(0, 12)  # [0, 12]
    edge_attr = expand_edge_attr_for_bidirectional(edge_attr_single)  # [n_edges*2, 12]

    n_edges = edge_index.shape[1]
    if n_edges == 0:
        edge_mask = torch.empty(0, dtype=torch.bool)
    else:
        edge_mask = torch.ones(n_edges, dtype=torch.bool)

    return {
        "node_features": node_features,
        "node_mask": node_mask,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_mask": edge_mask,
        "n_fragments": n_frag,
    }


# ======================================================================
#  Core: assemble output dict from model outputs
# ======================================================================

def _assemble_output(
    h_G: torch.Tensor,
    node_repr: torch.Tensor,
    alpha: torch.Tensor,
    fragments: List[sf.Fragment],
    output_keys: tuple,
) -> Dict:
    """Build the result dictionary for one molecule."""
    result = {}
    for key in output_keys:
        if key == "h_G":
            result["h_G"] = h_G
        elif key == "node_repr":
            result["node_repr"] = node_repr
        elif key == "alpha":
            result["alpha"] = alpha
        elif key == "weighted_node_repr":
            result["weighted_node_repr"] = node_repr * alpha.unsqueeze(-1)
        elif key == "fragment_meta":
            result["fragment_meta"] = [
                {"capped_form": f.capped_form, "category": f.category, "role": f.role}
                for f in fragments
            ]
    return result


# ======================================================================
#  Core public function: SMILES → embeddings  (single molecule)
# ======================================================================

@torch.inference_mode()
def get_embeddings(
    smiles: str,
    model: FragmentModel,
    backend,
    encoder,
    common_residues: set,
    common_linkers: set,
    device: str = "cpu",
    encode_opts: Optional[dict] = None,
    output: tuple = ("h_G",),
) -> Dict:
    """Process one SMILES and return the requested representations.

    Parameters
    ----------
    smiles:
        Input SMILES string (will be RDKit-canonicalized internally).
    model:
        Loaded FragmentModel (eval mode).
    backend:
        Loaded fragment-encoding backend module (model/<method>/backend.py).
    encoder:
        Backend-specific encoder object returned by ``backend.load``.
    common_residues, common_linkers:
        Sets of common fragment forms for decomposition control.
    device:
        Torch device string.
    encode_opts:
        Extra kwargs forwarded to ``backend.encode`` (e.g. batch_size / workers).
    output:
        Tuple of requested output keys.  Options:
        - ``"h_G"``               : graph-level embedding ``[hidden_dim]`` (default).
        - ``"node_repr"``         : per-fragment MPNN output ``[n_frag, hidden_dim]``.
        - ``"alpha"``             : attention weights ``[n_frag]``.
        - ``"weighted_node_repr"``: ``node_repr * alpha.unsqueeze(-1)``.
        - ``"fragment_meta"``     : list of dicts with capped_form, category, role.

    Returns
    -------
    dict
        Keys are the requested ``output`` entries.
    """
    encode_opts = encode_opts or {}
    canon = canonicalize_smiles(smiles)
    fragments, edges = sf.decompose(
        canon,
        common_residues=common_residues,
        common_linkers=common_linkers,
    )

    capped_forms = [f.capped_form for f in fragments]
    frag_embeddings = _encode_fragments(
        backend, encoder, capped_forms, device, encode_opts,
    )  # [n_frag, node_input_dim]

    batch = _build_sample_tensors(fragments, edges, frag_embeddings, canon)

    # Add batch dimension for single-sample forward
    model_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            model_batch[k] = v.unsqueeze(0).to(device)
        elif k == "n_fragments":
            model_batch[k] = [v]
        else:
            model_batch[k] = v

    # For edgeless graphs, edge_mask is [1, 0] after unsqueeze; the model
    # expects [1, max(1, n_edges)].  collate_fn handles this natively, but
    # for single-sample we must ensure at least dim 1.
    n_edges = model_batch["edge_mask"].shape[1]
    if n_edges == 0:
        model_batch["edge_mask"] = torch.zeros((1, 1), dtype=torch.bool, device=device)
        model_batch["edge_index"] = model_batch["edge_index"][:, :, :0]  # keep [1,2,0]
        model_batch["edge_attr"] = model_batch["edge_attr"][:, :0, :]    # keep [1,0,12]

    out = model(model_batch)
    h_G = out["h_G"][0]               # [hidden_dim]
    node_repr = out["node_repr"][0]    # [n_frag, hidden_dim]
    alpha = out["alpha"][0]            # [n_frag]

    return _assemble_output(h_G, node_repr, alpha, fragments, output)


# ======================================================================
#  Inference collate (node features are already embeddings; no training-side
#  node_ids/FRAG_TABLE needed)
# ======================================================================

def _collate_inference(samples: List[dict]) -> dict:
    """Collate variable-length fragment graphs into a dense batch.

    Same padding logic as ``common/data/collate.py::collate_fn`` but
    uses precomputed ``node_features`` directly (inference path does not build
    a fragment embedding table).
    """
    B = len(samples)
    max_frag = max(s["node_features"].shape[0] for s in samples)
    node_dim = samples[0]["node_features"].shape[1]
    max_edges = max((s["edge_attr"].shape[0] for s in samples), default=0)
    edge_dim = samples[0]["edge_attr"].shape[1] if samples[0]["edge_attr"].numel() else 12

    node_features = torch.zeros((B, max_frag, node_dim), dtype=torch.float32)
    node_mask = torch.zeros((B, max_frag), dtype=torch.bool)
    edge_index = torch.zeros((B, 2, max(max_edges, 1)), dtype=torch.long)
    edge_attr = torch.zeros((B, max(max_edges, 1), edge_dim), dtype=torch.float32)
    edge_mask = torch.zeros((B, max(max_edges, 1)), dtype=torch.bool)

    for i, s in enumerate(samples):
        n_frag = s["node_features"].shape[0]
        n_edge = s["edge_attr"].shape[0]
        node_features[i, :n_frag] = s["node_features"]
        node_mask[i, :n_frag] = True
        if n_edge > 0:
            edge_index[i, :, :n_edge] = s["edge_index"]
            edge_attr[i, :n_edge] = s["edge_attr"]
            edge_mask[i, :n_edge] = True

    return {
        "global_id": [s["global_id"] for s in samples],
        "node_features": node_features,
        "node_mask": node_mask,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_mask": edge_mask,
        "n_fragments": [s["n_fragments"] for s in samples],
    }


# ======================================================================
#  Batch variant for CSV processing
# ======================================================================

@torch.inference_mode()
def _get_embeddings_batch(
    smiles_list: List[str],
    ids: Optional[List[str]],
    model: FragmentModel,
    backend,
    encoder,
    common_residues: set,
    common_linkers: set,
    device: str = "cpu",
    encode_opts: Optional[dict] = None,
    gnn_batch_size: int = 256,
    output: tuple = ("h_G",),
) -> Dict[str, Dict]:
    """Process multiple SMILES with batched fragment encoding + batched GNN inference.

    Returns
    -------
    dict
        {id: {output_key: tensor or list, ...}}  (ids fall back to 0-based index).
    """
    encode_opts = encode_opts or {}

    # ---- Pass 1: canonicalise + decompose, collect capped_forms ----
    all_fragments = []
    all_edges = []
    all_capped_forms = []  # list of lists
    all_canons = []
    skipped = []

    for idx, smi in enumerate(smiles_list):
        try:
            canon = canonicalize_smiles(smi)
            frags, edges = sf.decompose(
                canon,
                common_residues=common_residues,
                common_linkers=common_linkers,
            )
            all_fragments.append(frags)
            all_edges.append(edges)
            all_capped_forms.append([f.capped_form for f in frags])
            all_canons.append(canon)
        except Exception as e:
            skipped.append((idx, smi[:60], str(e)[:60]))
            all_fragments.append(None)
            all_edges.append(None)
            all_capped_forms.append([])
            all_canons.append(None)

    if skipped:
        print(f"[WARN] skipped {len(skipped)} molecules during decomposition:")
        for idx, smi, err in skipped[:5]:
            print(f"  [{idx}] {smi}  |  {err}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    # ---- Pass 2: filter out failed molecules ----
    valid_indices = [i for i, f in enumerate(all_fragments) if f is not None]
    n_valid = len(valid_indices)
    n_total = len(smiles_list)
    if n_valid < n_total:
        print(f"[INFO] processing {n_valid}/{n_total} molecules ({n_total - n_valid} skipped)")

    if n_valid == 0:
        print("[ERROR] no valid molecules to process")
        return {}

    valid_fragments = [all_fragments[i] for i in valid_indices]
    valid_edges = [all_edges[i] for i in valid_indices]
    valid_capped = [all_capped_forms[i] for i in valid_indices]
    valid_canons = [all_canons[i] for i in valid_indices]
    valid_ids = [(ids[i] if ids else str(i)) for i in valid_indices]

    # ---- Pass 3: unique capped_forms → backend encoder (single batched call) ----
    unique_set: Dict[str, int] = {}
    for cf_list in valid_capped:
        for cf in cf_list:
            if cf not in unique_set:
                unique_set[cf] = len(unique_set)

    unique_list = [""] * len(unique_set)
    for cf, idx in unique_set.items():
        unique_list[idx] = cf

    print(f"[INFO] unique fragments to encode: {len(unique_list)}")
    t0 = time.time()
    unique_embeddings = _encode_fragments(
        backend, encoder, unique_list, device, encode_opts,
    )
    print(f"  done in {time.time() - t0:.1f}s")
    unique_dict = {cf: unique_embeddings[idx] for cf, idx in unique_set.items()}

    # ---- Pass 4: build graph tensors per molecule, collate, GNN forward ----
    class _TempDataset(torch.utils.data.Dataset):
        def __init__(self, samples_list):
            self.samples_list = samples_list
        def __len__(self):
            return len(self.samples_list)
        def __getitem__(self, idx):
            return self.samples_list[idx]

    samples_list = []
    for i in range(n_valid):
        frag_emb = torch.stack([unique_dict[cf] for cf in valid_capped[i]], dim=0)
        tensors = _build_sample_tensors(
            valid_fragments[i], valid_edges[i], frag_emb, valid_canons[i],
        )
        tensors["global_id"] = valid_ids[i]
        samples_list.append(tensors)

    loader = torch.utils.data.DataLoader(
        _TempDataset(samples_list),
        batch_size=gnn_batch_size,
        collate_fn=_collate_inference,
        shuffle=False,
    )

    print(f"[INFO] GNN batch inference: {n_valid} molecules, batch_size={gnn_batch_size}")
    t0 = time.time()
    results: Dict[str, Dict] = {}
    gid_to_vidx = {gid: i for i, gid in enumerate(valid_ids)}
    for batch in loader:
        gids = batch.pop("global_id")
        batch.pop("n_fragments", None)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        out = model(batch)
        for b_idx, gid in enumerate(gids):
            v_idx = gid_to_vidx[gid]
            frags = valid_fragments[v_idx]
            n_frag = len(frags)
            h_G_i = out["h_G"][b_idx]
            node_repr_i = out["node_repr"][b_idx, :n_frag]
            alpha_i = out["alpha"][b_idx, :n_frag]
            results[gid] = _assemble_output(h_G_i, node_repr_i, alpha_i, frags, output)

    print(f"  done in {time.time() - t0:.1f}s")
    return results


# ======================================================================
#  CLI
# ======================================================================

def _parse_output(output_str: str) -> tuple:
    """Parse --output argument.  'all' expands to every available key."""
    choices = {"h_G", "node_repr", "alpha", "weighted_node_repr", "fragment_meta"}
    if output_str == "all":
        return tuple(choices)
    keys = [k.strip() for k in output_str.split(",")]
    for k in keys:
        if k not in choices:
            raise argparse.ArgumentTypeError(
                f"Unknown output key '{k}'.  Choose from: {', '.join(sorted(choices))}, or 'all'."
            )
    return tuple(keys)


def _tensor_summary(t: torch.Tensor, max_elts: int = 6) -> str:
    """Short string representation: shape + first few elements."""
    flat = t.flatten()
    n = flat.numel()
    if n == 0:
        return f"shape={tuple(t.shape)}, []"
    prefix = ", ".join(f"{flat[i]:.6g}" for i in range(min(max_elts, n)))
    suffix = ", ..." if n > max_elts else ""
    return f"shape={tuple(t.shape)}, [{prefix}{suffix}]"


def main():
    # Peek at --config / --method early so defaults can come from config.
    parser = argparse.ArgumentParser(
        description="Compute embeddings from SMILES (unified fragment-encoding entry).",
    )
    parser.add_argument("smiles", nargs="?", default=None,
                        help="Single SMILES string (omit when using --csv)")
    parser.add_argument("--csv", default=None,
                        help="CSV file with columns: id, smiles")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to config.yaml (default: config.yaml in the project root)")
    parser.add_argument("--method", default=None,
                        help="Fragment-encoding method: molformer | molt5 | moler "
                             "(default: config.default_method)")
    parser.add_argument("--output", default="h_G",
                        help="Comma-separated output keys, or 'all'.  "
                             "Default: h_G.  Options: h_G, node_repr, alpha, "
                             "weighted_node_repr, fragment_meta")
    parser.add_argument("--save", default=None,
                        help="Path to save .pt results (required for --csv)")
    parser.add_argument("--model-dir", default=None,
                        help="Override base-model directory (default: config)")
    parser.add_argument("--ckpt", default=None,
                        help="Override FragmentModel checkpoint (default: config)")
    parser.add_argument("--common-sets", default=None,
                        help="Override common_sets.json path (default: config)")
    parser.add_argument("--device", default=None,
                        help="cpu / cuda (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override fragment-encoding batch size (default: config)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override backend worker count (MoLeR; default: config)")
    parser.add_argument("--gnn-batch-size", type=int, default=256,
                        help="Batch size for GNN forward (CSV mode)")
    args = parser.parse_args()

    if args.smiles is None and args.csv is None:
        parser.error("Either provide a SMILES string or --csv")
    if args.csv and args.save is None:
        parser.error("--save is required when using --csv")

    # ---- Resolve config + method ----
    cfg = load_config(args.config)
    method = args.method or cfg.get("default_method")
    if method not in cfg["methods"]:
        parser.error(f"Unknown method '{method}'.  Available: {', '.join(cfg['methods'])}")
    m = cfg["methods"][method]

    backend_path = m["_backend_abs"]
    model_dir = args.model_dir or m["_model_dir_abs"]
    ckpt = args.ckpt or m["_ckpt_abs"]
    common_sets = args.common_sets or cfg["_common_sets_abs"]
    node_input_dim = m["node_input_dim"]

    encode_opts = _backend_opts(m)
    if args.batch_size is not None:
        encode_opts["batch_size"] = args.batch_size
    if args.workers is not None:
        encode_opts["workers"] = args.workers

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_keys = _parse_output(args.output)

    print("=" * 55)
    print(f"  method: {method}  ({m.get('desc', '')})")
    print(f"  device: {device}, node_input_dim={node_input_dim}")
    print("=" * 55)

    # ---- Load models ----
    print("  Loading models ...")
    t0 = time.time()
    backend = _load_backend(backend_path)
    encoder = backend.load(model_dir, device=device, **encode_opts)
    fragment_model = load_fragment_model(ckpt, device, node_input_dim=node_input_dim)
    common_residues, common_linkers = load_common_sets(common_sets)
    print(f"  Total load time: {time.time() - t0:.1f}s\n")

    # ---- Single SMILES mode ----
    if args.smiles is not None:
        canon = canonicalize_smiles(args.smiles)
        print(f"SMILES: {args.smiles}")
        print(f"Canonical: {canon}")
        print(f"Output keys: {output_keys}")
        print("-" * 40)

        t0 = time.time()
        result = get_embeddings(
            args.smiles, fragment_model, backend, encoder,
            common_residues, common_linkers, device,
            encode_opts=encode_opts,
            output=output_keys,
        )
        elapsed = time.time() - t0

        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {_tensor_summary(val)}")
            elif isinstance(val, list):
                print(f"  {key}: [{len(val)} items]")
                if val and isinstance(val[0], dict):
                    for i, meta in enumerate(val):
                        print(f"    [{i}] {meta}")
            else:
                print(f"  {key}: {val}")

        print(f"Time: {elapsed:.2f}s")

        if args.save:
            torch.save(result, args.save)
            print(f"Saved to {args.save}")

    # ---- CSV mode ----
    else:
        print(f"Reading CSV: {args.csv}")
        ids = []
        smiles_list = []
        with open(args.csv, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                # skip header row if present
                if row[0].strip().lower() in ("id", "global_id", "smiles") and \
                   row[1].strip().lower() in ("smiles", "smile", "canonical_smiles"):
                    continue
                ids.append(row[0].strip())
                smiles_list.append(row[1].strip())

        print(f"  molecules: {len(smiles_list)}")
        print(f"Output keys: {output_keys}")
        print(f"Save path:   {args.save}")
        print()

        results = _get_embeddings_batch(
            smiles_list, ids, fragment_model, backend, encoder,
            common_residues, common_linkers, device,
            encode_opts=encode_opts,
            gnn_batch_size=args.gnn_batch_size,
            output=output_keys,
        )

        # Convert tensors to cpu for saving
        save_dict = {}
        for gid, out_dict in results.items():
            save_dict[gid] = {}
            for k, v in out_dict.items():
                if isinstance(v, torch.Tensor):
                    save_dict[gid][k] = v.cpu()
                else:
                    save_dict[gid][k] = v

        torch.save(save_dict, args.save)
        print(f"\n[OK] saved {len(save_dict)} entries to {args.save}")

        # Print summary
        for gid in list(save_dict.keys())[:3]:
            print(f"\n  {gid}:")
            for k, v in save_dict[gid].items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {_tensor_summary(v)}")
                elif isinstance(v, list):
                    print(f"    {k}: [{len(v)} items]")
        if len(save_dict) > 3:
            print(f"  ... ({len(save_dict) - 3} more)")


if __name__ == "__main__":
    main()
