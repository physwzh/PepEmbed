"""MoLeR + chirality channel + residue-kind channel -- a self-contained 526-dim fragment encoder (zero training).

    full = [ z_moler (512) ; c (k=8) ; t (m=6) ]   -> 526-dim float32

This file is the **single source of truth** for `full_split_MoLeR_1024_5.3M`. All the
algorithmic logic of the `c` / `t` channels is vendored from the `add_dimen_chirality`
module:
        chirality_code.py   -> c channel (fixed k-dim R/S CIP slots)
        residue_kind.py     -> t channel (residue kind)
        augmented_moler.py  -> encoding half (encode/encode_one/full_size/mirror_code)
It does **not import, nor sys.path-inject** that directory -- this directory is
self-contained and independently reproducible. The only external dependency,
`molecule_generation`, is a package already installed in editable mode in the
`peptide` conda environment, so it is a normal import, not a path injection.

Semantics aligned verbatim with upstream
----------------------------------------
* What is fed to MoLeR is the **stereo-stripped** SMILES (`stereo_free`) -- MoLeR
  would call `Chem.RemoveStereochemistry` internally anyway; stripping it explicitly
  keeps it consistent with the slot anchors of `c` (stereo-free canonical atom order);
* `c` is computed from the **original stereo-bearing** SMILES: per-slot R/S CIP in
  {-1, 0, +1};
* `t` is computed from the same SMILES: `std_aa`/`mod_aa`/`non_aa` one-hot +
  multi-residue + backbone L/D.
Neither participates in MoLeR's encoding/decoding; they are simply concatenated after
the latent => no training required.

Consistency evidence: for 200 real fragments, this module's `c`/`t` are element-wise
identical (0 difference) to the upstream original implementation; `z[:512]` matches
the bare `VaeWrapper.encode` to float32 precision. See `MOLER_SWAP.md`.

Reproducibility contract for the latent (important)
---------------------------------------------------
`MoLeRInferenceServer.encode`'s chunk formula depends on the input length N:
`chunk_size = min(500//beam_size, N//num_workers + 1)`. Different chunk sizes ->
different reduction order on the TF side -> a ~**7e-7** difference on float32
(measured). Therefore:
* **Within a single call**: row order maps strictly one-to-one to the input; a
  repeated identical molecule is bit-identical (diff = 0);
* **Across calls**: the same molecule at different chunk scales jitters by <=1e-6,
  which is **not** a misalignment.
For controlled comparisons, make `num_workers` and N identical on both sides, or use
a tolerance of <=1e-5.

CPU / GPU timing
----------------
MoLeR can only run on CPU: this machine's TF 2.9.3 has no kernel for Blackwell sm_120,
and PTX JIT makes the GPU 105x slower than the CPU. Meanwhile the online encoding path
(`get_embedding.py` etc.) also needs torch to use the GPU. Wrap the encoding section in
`moler_cpu_session()`:

    with moler_cpu_session():
        with MolToMoler526(model_dir, num_workers=44) as enc:
            z = enc.encode(smiles)
    # CUDA_VISIBLE_DEVICES is restored on exit -> only the first torch.cuda call
    # afterwards sees the GPU

The workers are separate forked processes; the parent's `tf.config.set_visible_devices`
is **not** propagated, so the GPU can only be masked via `CUDA_VISIBLE_DEVICES=""`, and
it must be set before importing tf -- which is exactly what `moler_cpu_session()` does.

Parallelism
-----------
`MoLeRInferenceServer.encode`'s chunk formula (`moler_inference_server.py:300`):
    chunk_size = min(500 // beam_size, len(smiles_list) // num_workers + 1)
`beam_size=1` => cap 500. 5356 fragments x num_workers=44 => chunk=122 => 44 chunks,
44 workers each handling ~122 molecules, fully loaded. Measured 1.85 ms/mol
(542 mol/s), peak anon 18 GiB.

Alignment defence (R9)
----------------------
`moler_inference_server._encode_from_smiles` (:49-56) `continue`s on molecules it cannot
parse, **silently skipping** them, so the returned list may be **shorter** than the input
list; upstream `augmented_moler` indexes it positionally with `zs[i]` => once a fragment
in the middle is skipped, **all subsequent fragments are silently misaligned** (no error!).
This module asserts equal length after every call; if they differ it **binary-searches**
for the failing SMILES, records it in `self.dropped`, and uses an **all-zero 526 vector**
as a placeholder, guaranteeing that `feature` rows align one-to-one with `capped_form` rows.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
FEATURE_DIM = 526
MOLER_LATENT_SIZE = 512
DEFAULT_K = 8
DEFAULT_M = 6
DEFAULT_NUM_WORKERS = 44        # offline whole-library encoding (CPU-exclusive, saturate the 48-core quota)
ONLINE_MOLER_WORKERS = 8        # online encoding (same process as the GPU job; worker count must be small)
# Default MoLeR base model, resolved relative to this file (model/moler/base_model);
# config.yaml points at the same directory and normally passes it explicitly.
DEFAULT_MOLER_MODEL_DIR = str(Path(__file__).resolve().parent / "base_model")


# =========================================================================== #
# vendored: add_dimen_chirality/chirality_code.py
# =========================================================================== #
try:  # reuse MoLeR's built-in implementation to stay same-source with the pipeline (molecule_generation is an installed package)
    from molecule_generation.chem.rdkit_helpers import compute_canonical_atom_order
except Exception:  # pragma: no cover - degrade to an equivalent implementation when run standalone

    def compute_canonical_atom_order(mol: Chem.Mol) -> List[int]:
        """Canonical atom order: returns order, where order[p] is the atom index for canonical position p."""
        _ = Chem.MolToSmiles(mol)
        return list(
            mol.GetPropsAsDict(includePrivate=True, includeComputed=True)[
                "_smilesAtomOutputOrder"
            ]
        )


K_DEFAULT = 8

R_ = 1.0
S_ = -1.0
UNSPEC_ = 0.0


def _as_mol(smi: Union[str, Chem.Mol]) -> Chem.Mol:
    if isinstance(smi, Chem.Mol):
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smi!r}")
    return mol


def stereo_free(smi: Union[str, Chem.Mol]) -> str:
    """Canonical SMILES after removing stereochemistry (invariant to enantiomers)."""
    mol = Chem.Mol(_as_mol(smi))
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _flat_and_order(smi: Union[str, Chem.Mol]) -> Tuple[Chem.Mol, List[int], Dict[int, int]]:
    """Return (stereo-free molecule, order, pos). `flat` shares atom indices with the input `mol`."""
    mol = _as_mol(smi)
    flat = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat)
    order = compute_canonical_atom_order(flat)
    pos = {atom_idx: p for p, atom_idx in enumerate(order)}
    return flat, order, pos


def stereo_slots(flat_smi: Union[str, Chem.Mol], k: Optional[int] = None) -> List[int]:
    """The "potential chirality slots" of a stereo-free molecule -- a list of canonical positions in ascending order."""
    mol = Chem.Mol(_as_mol(flat_smi))
    order = compute_canonical_atom_order(mol)
    pos = {atom_idx: p for p, atom_idx in enumerate(order)}
    cand = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    slots = sorted(pos[i] for i, _ in cand)
    return slots[:k] if k is not None else slots


def _assigned_cips(mol: Chem.Mol) -> Dict[int, str]:
    """Atom index -> CIP ('R'/'S'), including only centres with an assigned configuration."""
    return {
        i: c
        for i, c in Chem.FindMolChiralCenters(
            mol, includeUnassigned=False, useLegacyImplementation=False
        )
    }


def chirality_code(
    smiles: Union[str, Chem.Mol], k: int = K_DEFAULT
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Encode a molecule's stereo information into a length-k vector `c`.

    Returns `(c, meta)`: `c[j] ∈ {-1, 0, +1}` (S / unspecified / R);
    `meta` contains `n_slots`, `n_assigned`, `overflow` (`n_slots > k`).
    """
    mol = _as_mol(smiles)
    flat, order, pos = _flat_and_order(mol)
    assigned = _assigned_cips(mol)
    slots = stereo_slots(flat)

    c = np.zeros(k, dtype=np.float32)
    for j, p in enumerate(slots[:k]):
        atom_idx = order[p]
        if atom_idx in assigned:
            c[j] = R_ if assigned[atom_idx] == "R" else S_
    meta = {
        "n_slots": len(slots),
        "n_assigned": len(assigned),
        "overflow": len(slots) > k,
    }
    return c, meta


def invert(c: np.ndarray) -> np.ndarray:
    """Enantiomer: c -> -c (each slot R<->S)."""
    return -np.asarray(c)


def _flip_tag(mol: Chem.Mol, atom_idx: int) -> None:
    a = mol.GetAtomWithIdx(atom_idx)
    if a.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    else:
        a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)


def reapply_stereo(
    flat_smi: Union[str, Chem.Mol],
    c: np.ndarray,
    slots: Optional[List[int]] = None,
    max_passes: int = 12,
    max_bruteforce: int = 16,
) -> Tuple[str, int]:
    """Write the chirality vector `c` back onto a (stereo-free) molecule.

    Returns `(isomeric_smiles, n_unresolved)`. The decoding half is not needed in this
    directory (encoding only), but this function is kept to self-test the semantics of
    `c` (`reapply_stereo(stereo_free(s), c)` is an identity round-trip).
    """
    mol = Chem.Mol(_as_mol(flat_smi))
    Chem.RemoveStereochemistry(mol)
    order = compute_canonical_atom_order(mol)
    if slots is None:
        slots = stereo_slots(mol)

    targets: Dict[int, str] = {}
    for j, p in enumerate(slots[: len(c)]):
        v = float(c[j])
        if v > 0:
            targets[p] = "R"
        elif v < 0:
            targets[p] = "S"
    if not targets:
        return Chem.MolToSmiles(mol), 0

    # Initial guess: R -> CW, S -> CCW.
    for p, tgt in targets.items():
        atom = mol.GetAtomWithIdx(order[p])
        atom.SetChiralTag(
            Chem.ChiralType.CHI_TETRAHEDRAL_CW
            if tgt == "R"
            else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        )

    pos = {atom_idx: p for p, atom_idx in enumerate(order)}

    def current_mismatches() -> List[int]:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        got = {pos[i]: cip for i, cip in _assigned_cips(mol).items() if i in pos}
        return [p for p, tgt in targets.items() if got.get(p) != tgt]

    # 1) Greedy: flip the tag of every mismatching slot once per round.
    unresolved = current_mismatches()
    for _ in range(max_passes):
        if not unresolved:
            break
        for p in unresolved:
            _flip_tag(mol, order[p])
        unresolved = current_mismatches()

    # 2) Fallback: CIP priorities couple between adjacent centres (like/unlike rules), so greedy may oscillate.
    if unresolved and len(unresolved) <= max_bruteforce:
        import itertools

        stubborn = list(unresolved)
        best = None
        for combo in itertools.product(
            (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
            repeat=len(stubborn),
        ):
            for p, tag in zip(stubborn, combo):
                mol.GetAtomWithIdx(order[p]).SetChiralTag(tag)
            mism = current_mismatches()
            if not mism:
                unresolved = []
                break
            if best is None or len(mism) < best[0]:
                best = (len(mism), list(mism))
        else:
            unresolved = best[1] if best else unresolved

    return Chem.MolToSmiles(mol), len(unresolved)


# =========================================================================== #
# vendored: add_dimen_chirality/residue_kind.py
# =========================================================================== #
BACKBONE_SMARTS = "[NX3][CX4][CX3]=O"
_BACKBONE = Chem.MolFromSmarts(BACKBONE_SMARTS)

STD_RESIDUE_SMILES: Dict[str, str] = {
    "Gly": "NCC(=O)O",
    "Ala": "C[C@@H](N)C(=O)O",
    "Ser": "OC[C@@H](N)C(=O)O",
    "Cys": "SC[C@@H](N)C(=O)O",
    "Thr": "C[C@@H](O)[C@@H](N)C(=O)O",
    "Val": "C[C@@H](C)[C@@H](N)C(=O)O",
    "Leu": "CC(C)C[C@@H](N)C(=O)O",
    "Ile": "CC[C@H](C)[C@H](N)C(=O)O",
    "Met": "CSCC[C@@H](N)C(=O)O",
    "Pro": "OC(=O)[C@@H]1CCCN1",
    "Phe": "OC(=O)[C@@H](N)Cc1ccccc1",
    "Tyr": "OC(=O)[C@@H](N)Cc1ccc(O)cc1",
    "Trp": "OC(=O)[C@@H](N)Cc1c[nH]c2ccccc12",
    "His": "OC(=O)[C@@H](N)Cc1c[nH]cn1",
    "Asp": "OC(=O)C[C@@H](N)C(=O)O",
    "Asn": "NC(=O)C[C@@H](N)C(=O)O",
    "Glu": "OC(=O)CC[C@@H](N)C(=O)O",
    "Gln": "NC(=O)CC[C@@H](N)C(=O)O",
    "Lys": "NCCCC[C@@H](N)C(=O)O",
    "Arg": "N=C(N)NCCC[C@@H](N)C(=O)O",
}

# Literature values: CIP of the alpha carbon in the L configuration. All are S except Cys.
EXPECTED_L_CIP: Dict[str, str] = {
    name: ("R" if name == "Cys" else "S") for name in STD_RESIDUE_SMILES
}

# Literature values: CIP of **side-chain** chiral centres in the L configuration (only Thr/Ile have one among the 20 standard residues).
EXPECTED_L_SIDECHAIN_CIP: Dict[str, str] = {"Thr": "R", "Ile": "S"}

# `t` channel layout
KIND_NAMES = ("std_aa", "mod_aa", "non_aa")
T_SIZE = 6
# t[0:3] one-hot {std_aa, mod_aa, non_aa}
# t[3]   n_residues > 1 (multi-residue/peptide)
# t[4]   backbone L
# t[5]   backbone D
T_LAYOUT = {
    0: "kind=std_aa",
    1: "kind=mod_aa",
    2: "kind=non_aa",
    3: "n_residues>1",
    4: "backbone=L",
    5: "backbone=D",
}


def backbone_matches(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    """Return [(n_idx, alpha_idx, c_idx, o_idx), ...]."""
    return [tuple(m) for m in mol.GetSubstructMatches(_BACKBONE)]  # type: ignore[misc]


def alpha_cip(mol: Chem.Mol, alpha_idx: int) -> Optional[str]:
    """CIP of the alpha carbon ('R'/'S'), or None if unspecified or not a centre."""
    m = Chem.Mol(mol)
    try:
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    except Exception:
        return None
    for i, c in Chem.FindMolChiralCenters(
        m, includeUnassigned=True, useLegacyImplementation=False
    ):
        if i == alpha_idx:
            return c
    return None


def _cip_map(mol: Chem.Mol) -> Dict[int, str]:
    """{atom index: CIP letter}, including unspecified centres (value '?')."""
    m = Chem.Mol(mol)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return dict(Chem.FindMolChiralCenters(
        m, includeUnassigned=True, useLegacyImplementation=False))


def _flip_alpha(mol: Chem.Mol, alpha_idx: int) -> Chem.Mol:
    m = Chem.Mol(mol)
    a = m.GetAtomWithIdx(alpha_idx)
    t = a.GetChiralTag()
    if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return m


def _mirror(mol: Chem.Mol) -> Chem.Mol:
    """Flip the tag of **all** chiral centres to obtain the true enantiomer.

    beta-branched residues (Thr/Ile) have a side-chain chiral centre: flipping only the
    alpha yields an **epimer** (D-allo-Thr / D-allo-Ile), not the enantiomer. The
    enantiomer must be mirrored as a whole.
    """
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        t = a.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return m


def residue_key(
    mol: Chem.Mol,
    n_idx: int,
    a_idx: int,
    c_idx: int,
    drop_alpha_stereo: bool = False,
) -> Optional[str]:
    """Residue fingerprint: cap-independent; stereo-bearing (alpha-carbon parity encoded in `@`/`@@`)."""
    try:
        keep = {a_idx}
        stack = [a_idx]
        while stack:
            for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                j = nb.GetIdx()
                if j in (n_idx, c_idx) or j in keep:
                    continue
                keep.add(j)
                stack.append(j)
        # Pro type: N part of a ring -> extend along ring bonds from N (excluding return to Calpha / carboxyl C)
        stack = [n_idx]
        seen_n = {n_idx}
        while stack:
            cur = stack.pop()
            for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                j = nb.GetIdx()
                if j in (a_idx, c_idx) or j in seen_n:
                    continue
                b = mol.GetBondBetweenAtoms(cur, j)
                if b is None or not b.IsInRing():
                    continue
                seen_n.add(j)
                keep.add(j)
                stack.append(j)

        em = Chem.RWMol(mol)
        for idx, lab in ((n_idx, 1), (c_idx, 2)):
            at = em.GetAtomWithIdx(idx)
            at.SetAtomicNum(0)
            at.SetIsotope(lab)
            at.SetNoImplicit(True)
            at.SetFormalCharge(0)
            at.SetNumExplicitHs(0)
        drop = sorted(
            set(range(em.GetNumAtoms())) - keep - {n_idx, c_idx}, reverse=True
        )
        for idx in drop:
            em.RemoveAtom(idx)
        m2 = em.GetMol()
        Chem.SanitizeMol(m2)
        root = a_idx - sum(1 for d in drop if d < a_idx)
        if drop_alpha_stereo:
            m2.GetAtomWithIdx(root).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        return Chem.MolToSmiles(m2, rootedAtAtom=root, canonical=True)
    except Exception:
        return None


def _build_reference_tables():
    l_keys: Dict[str, str] = {}
    d_keys: Dict[str, str] = {}
    id_keys: Dict[str, str] = {}   # "identity" fingerprint with the alpha-carbon configuration erased
    l_smiles: Dict[str, str] = {}
    corrected: Dict[str, bool] = {}
    for name, smi in STD_RESIDUE_SMILES.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise ValueError(f"bad reference SMILES for {name!r}: {smi!r}")
        matches = backbone_matches(m)
        if len(matches) != 1:
            raise ValueError(
                f"reference {name!r} must have exactly 1 backbone match, "
                f"got {len(matches)}"
            )
        n, a, c, _o = matches[0]
        got = alpha_cip(m, a)
        if name != "Gly" and got != EXPECTED_L_CIP[name]:
            m = _flip_alpha(m, a)
            got = alpha_cip(m, a)
            corrected[name] = True
        if name != "Gly" and got != EXPECTED_L_CIP[name]:
            raise ValueError(
                f"reference {name!r}: cannot reach L CIP "
                f"{EXPECTED_L_CIP[name]} (have {got})"
            )
        # The **side-chain** chiral centre of beta-branched residues must be verified
        # separately: the alpha correction only flips the alpha, so if the input is the
        # allo form, flipping the alpha makes it "look right" and the error is silently
        # absorbed.
        if name in EXPECTED_L_SIDECHAIN_CIP:
            cip = _cip_map(m)
            side = [(i, v) for i, v in cip.items() if i != a]
            want = EXPECTED_L_SIDECHAIN_CIP[name]
            if len(side) != 1 or side[0][1] != want:
                raise ValueError(
                    f"reference {name!r}: side-chain chiral centre must be CIP "
                    f"{want!r}, got {side!r} — the alpha-only correction cannot "
                    f"fix a wrong side-chain centre (allo vs standard)"
                )
        n, a, c, _o = backbone_matches(m)[0]
        l_keys[name] = residue_key(m, n, a, c)
        id_keys[name] = residue_key(m, n, a, c, drop_alpha_stereo=True)
        l_smiles[name] = Chem.MolToSmiles(m)
        # The D reference must be the **true enantiomer** of the L reference (flip all centres), not just the alpha.
        md = _mirror(m)
        n2, a2, c2, _o2 = backbone_matches(md)[0]
        d_keys[name] = residue_key(md, n2, a2, c2)
    if len(set(id_keys.values())) != len(id_keys):
        raise ValueError(
            "identity keys (alpha-stereo-stripped) must be pairwise distinct, "
            "else side-chain epimers would be conflated"
        )
    return l_keys, d_keys, id_keys, l_smiles, corrected


L_KEYS, D_KEYS, ID_KEYS, L_REF_SMILES, _CORRECTED = _build_reference_tables()
KEY_TO_L: Dict[str, str] = {v: k for k, v in L_KEYS.items()}
KEY_TO_D: Dict[str, str] = {v: k for k, v in D_KEYS.items()}
ID_KEY_TO_NAME: Dict[str, str] = {v: k for k, v in ID_KEYS.items()}
_ACHIRAL_KEYS = {k for k in L_KEYS.values() if k in set(D_KEYS.values())}  # Gly


def reference_report() -> Dict[str, object]:
    """Self-check report: whether the reference tables are sound (20 distinct L fingerprints / identity fingerprints, stereo corrected by CIP)."""
    return {
        "n_residues": len(STD_RESIDUE_SMILES),
        "n_distinct_L_keys": len(set(L_KEYS.values())),
        "n_distinct_identity_keys": len(set(ID_KEYS.values())),
        "achiral_keys": sorted(_ACHIRAL_KEYS),
        "cip_corrected": sorted(_CORRECTED),
        "expected_L_cip": dict(EXPECTED_L_CIP),
        "expected_L_sidechain_cip": dict(EXPECTED_L_SIDECHAIN_CIP),
    }


def classify(smiles: Union[str, Chem.Mol]) -> Tuple[str, Dict[str, object]]:
    """Return `(kind, info)`, `kind ∈ {'std_aa', 'mod_aa', 'non_aa'}`."""
    mol = smiles if isinstance(smiles, Chem.Mol) else Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    matches = backbone_matches(mol)
    if not matches:
        return "non_aa", {"n_residues": 0, "residues": [], "backbone_ld": None}

    residues: List[Dict[str, object]] = []
    n_std = 0
    cip_map = _cip_map(mol)
    for (n, a, c, _o) in matches:
        k = residue_key(mol, n, a, c)
        name = KEY_TO_L.get(k) or KEY_TO_D.get(k)
        if name is None:
            # The full-stereo fingerprint missed: this may be a "standard residue whose
            # alpha-carbon configuration is **unspecified**". If the identity fingerprint
            # (alpha configuration erased, side-chain stereo kept) hits, the residue name
            # is known and L/D is unspecified. But we must first confirm the alpha is
            # really unspecified: the identity fingerprint discards the alpha
            # configuration, so a side-chain epimer with "alpha assigned but a non-standard
            # configuration" (e.g. D-allo-Thr) would also hit and be misnamed as a
            # standard residue.
            if cip_map.get(a) not in ("R", "S"):
                name = ID_KEY_TO_NAME.get(
                    residue_key(mol, n, a, c, drop_alpha_stereo=True))
            ld = None
        elif k in KEY_TO_D and k in KEY_TO_L:
            # Gly (no chiral centre): L/D are isomorphous and cannot be distinguished
            ld = None
        else:
            ld = "L" if k in KEY_TO_L else "D"
        if name is not None:
            n_std += 1
        residues.append(
            {"name": name, "ld": ld, "alpha_cip": alpha_cip(mol, a), "key": k}
        )

    kind = "std_aa" if n_std == len(matches) else "mod_aa"
    lds = [r["ld"] for r in residues]
    if lds and all(x == "L" for x in lds):
        backbone_ld = "L"
    elif lds and all(x == "D" for x in lds):
        backbone_ld = "D"
    else:
        backbone_ld = None  # mixed / unnamed / no stereo
    return kind, {
        "n_residues": len(matches),
        "residues": residues,
        "backbone_ld": backbone_ld,
    }


def residue_kind_code(
    smiles: Union[str, Chem.Mol], m: int = T_SIZE
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return `(t, meta)`, where `t` is a float32 vector of length m=T_SIZE (see `T_LAYOUT`)."""
    kind, info = classify(smiles)
    t = np.zeros(m, dtype=np.float32)
    t[KIND_NAMES.index(kind)] = 1.0
    if m > 3:
        t[3] = 1.0 if info["n_residues"] > 1 else 0.0
    if m > 5:
        t[4] = 1.0 if info["backbone_ld"] == "L" else 0.0
        t[5] = 1.0 if info["backbone_ld"] == "D" else 0.0
    meta = {
        "kind": kind,
        "n_residues": info["n_residues"],
        "backbone_ld": info["backbone_ld"],
        "residue_names": [r["name"] for r in info["residues"]],
        "residues": info["residues"],
        "is_non_aa": kind == "non_aa",
        "is_non_standard_aa": kind == "mod_aa",
    }
    return t, meta


# =========================================================================== #
# CPU timing / thread pinning
# =========================================================================== #
# Each worker must be single-threaded, otherwise 44 workers x TF's default
# thread count would blow past the 48-CPU quota.
# These env vars must be set **before the fork** so the workers inherit them
# (workers are separate processes; calling tf.config.threading.* in the parent
# afterwards has no effect on already-forked workers).
_CPU_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
}


def pin_cpu_threads() -> Dict[str, Optional[str]]:
    """Set the single-thread env vars (inherited by the forked workers) and return the old values for restoration."""
    old: Dict[str, Optional[str]] = {}
    for k, v in _CPU_THREAD_ENV.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    return old


def restore_env(old: Dict[str, Optional[str]]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@contextlib.contextmanager
def moler_cpu_session(pin_threads: bool = True):
    """Timing guarantee so MoLeR runs on CPU while torch can still use the GPU afterwards.

    * On entry `CUDA_VISIBLE_DEVICES=""` -- this must happen before importing tf /
      constructing MoLeR, because the workers are separate forked processes and
      `tf.config.set_visible_devices` does not propagate to them.
    * At the same time set the single-thread env vars so the subsequently forked
      workers inherit them.
    * On exit **restore** `CUDA_VISIBLE_DEVICES` (and the thread env vars) -- torch
      reads that variable lazily, so the first `torch.cuda` call afterwards sees the GPU.

    Usage::

        with moler_cpu_session():
            with MolToMoler526(model_dir, num_workers=44) as enc:
                z = enc.encode(smiles)
        # only after this point is torch.cuda.is_available() True
    """
    old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    old_env = pin_cpu_threads() if pin_threads else {}
    try:
        yield
    finally:
        if old_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
        if old_env:
            restore_env(old_env)


# =========================================================================== #
# Encoder
# =========================================================================== #
class MolToMoler526:
    """526-dim fragment encoder = MoLeR(512) + c(k=8) + t(m=6) (training-free).

    Public interface::

        with MolToMoler526(num_workers=44) as enc:
            z = enc.encode(["C[C@H](N)C(=O)O", ...])   # torch.FloatTensor[N,526]
        enc.dropped          # SMILES silently skipped by MoLeR (zero-vector placeholders)
        enc.stats            # cumulative stats: n_dropped / n_overflow / kind distribution ...
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MOLER_MODEL_DIR,
        k: int = DEFAULT_K,
        m: int = DEFAULT_M,
        num_workers: int = DEFAULT_NUM_WORKERS,
        use_kind: bool = True,
        fill_dropped_with_zeros: bool = True,
    ):
        from molecule_generation import load_model_from_directory

        self.k = int(k)
        self.m = int(m) if use_kind else 0
        self.num_workers = int(num_workers)
        self.model_dir = model_dir
        self.fill_dropped_with_zeros = bool(fill_dropped_with_zeros)

        # Workers are forked, so TF's intra/inter-op thread counts must be locked
        # down before the fork.
        # Importing tensorflow here is safe: workers inherit the already-initialized
        # TF directly from the parent process.
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        self._ctx = load_model_from_directory(model_dir, num_workers=self.num_workers)
        self.model = self._ctx.__enter__()
        # The workers are non-daemon processes; on interpreter exit,
        # multiprocessing.util._exit_function joins them with no timeout -- they are
        # blocked on the request queue and never return => the process hangs. This
        # atexit handler is registered later than multiprocessing's own, so it runs
        # **before** it, ensuring callers that forget to call close() still exit cleanly.
        import atexit

        atexit.register(self.close)
        self.latent_size = int(self.model._latent_size)  # 512
        if self.latent_size != MOLER_LATENT_SIZE:
            raise ValueError(
                f"unexpected MoLeR latent size {self.latent_size} "
                f"(expected {MOLER_LATENT_SIZE})"
            )

        # Cumulative stats (for Phase-1 self-check / MOLER_SWAP.md evidence)
        self.dropped: List[str] = []
        self.stats: Dict[str, object] = {
            "n_encoded": 0,
            "n_dropped": 0,
            "n_overflow": 0,
            "max_n_slots": 0,
            "kind_counts": {name: 0 for name in KIND_NAMES},
            "n_multi_residue": 0,
            "n_backbone_L": 0,
            "n_backbone_D": 0,
        }

    # -- Lifecycle ----------------------------------------------------------
    def close(self) -> None:
        try:
            self._ctx.__exit__(None, None, None)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def full_size(self) -> int:
        return self.latent_size + self.k + self.m

    # -- Encoding -----------------------------------------------------------
    def _encode_latents_checked(
        self, flats: Sequence[str]
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """Call MoLeR and guarantee the returned length == input length; if not equal, binary-search the culprit and use an all-zero placeholder (R9).

        Returns `(latents, dropped)`, both of length equal to the input. `dropped[i]`
        being True means molecule i was silently skipped by MoLeR, and `latents[i]`
        is an **all-zero placeholder vector**.
        """
        n = len(flats)
        if n == 0:
            return [], []
        raw = [np.asarray(v, dtype=np.float32) for v in self.model.encode(list(flats))]
        if len(raw) == n:
            return raw, [False] * n

        # Length mismatch -- some molecule was silently skipped by
        # `_encode_from_smiles` (its :49-56 continue).
        # Binary-search: halve the interval each round until it narrows to a single molecule.
        out: List[Optional[np.ndarray]] = [None] * n
        dropped = [False] * n
        stack: List[Tuple[int, int]] = [(0, n)]
        while stack:
            lo, hi = stack.pop()
            sub = list(flats[lo:hi])
            if not sub:
                continue
            try:
                got = [np.asarray(v, dtype=np.float32) for v in self.model.encode(sub)]
            except Exception as e:  # a worker error also counts as "this sub-interval is bad"
                print(f"[moler_526] sub-encode raised on [{lo},{hi}): {type(e).__name__}: {e}")
                got = []
            if len(got) == len(sub):
                for i, v in enumerate(got):
                    out[lo + i] = v
                continue
            if len(sub) == 1:
                dropped[lo] = True
                self.dropped.append(sub[0])
                out[lo] = np.zeros(self.latent_size, dtype=np.float32)
                print(
                    f"[moler_526] WARNING MoLeR dropped molecule #{lo}: "
                    f"{sub[0]!r} -> all-zero placeholder"
                )
                continue
            mid = lo + len(sub) // 2
            stack.append((mid, hi))
            stack.append((lo, mid))

        zero = np.zeros(self.latent_size, dtype=np.float32)
        return [v if v is not None else zero for v in out], dropped

    def encode_meta(
        self, smiles_list: Sequence[str]
    ) -> Tuple[np.ndarray, List[dict]]:
        """Return `(Z[N,526] float32, metas[N])`, with rows strictly one-to-one with the input.

        `metas[i]` = {dropped, n_slots, n_assigned, overflow, kind, n_residues,
                      backbone_ld, residue_names, flat}
        """
        smis = list(smiles_list)
        # 1) Strip-stereo SMILES go to MoLeR (byte-for-byte aligned with upstream augmented_moler)
        flats = [stereo_free(s) for s in smis]
        zs, dropped = self._encode_latents_checked(flats)

        # 2) c / t are computed from the **original stereo-bearing** SMILES (pure RDKit, in the parent process)
        rows: List[np.ndarray] = []
        metas: List[dict] = []
        for i, s in enumerate(smis):
            c, c_meta = chirality_code(s, k=self.k)
            parts = [zs[i], c]
            if self.m > 0:
                t, t_meta = residue_kind_code(s, m=self.m)
                parts.append(t)
            else:
                t_meta = {"kind": None, "n_residues": 0, "backbone_ld": None,
                          "residue_names": []}
            rows.append(np.concatenate(parts).astype(np.float32))
            metas.append({
                "dropped": bool(dropped[i]),
                "n_slots": c_meta["n_slots"],
                "n_assigned": c_meta["n_assigned"],
                "overflow": bool(c_meta["overflow"]),
                "kind": t_meta["kind"],
                "n_residues": t_meta["n_residues"],
                "backbone_ld": t_meta["backbone_ld"],
                "residue_names": t_meta["residue_names"],
                "flat": flats[i],
            })

        # 3) Cumulative stats (for Phase-1 self-check / MOLER_SWAP.md evidence)
        self.stats["n_encoded"] += len(smis)
        for mt in metas:
            self.stats["n_overflow"] += int(mt["overflow"])
            self.stats["max_n_slots"] = max(int(self.stats["max_n_slots"]), int(mt["n_slots"]))
            if mt["kind"] is not None:
                self.stats["kind_counts"][mt["kind"]] = (
                    self.stats["kind_counts"].get(mt["kind"], 0) + 1
                )
            self.stats["n_multi_residue"] += int(mt["n_residues"] > 1)
            self.stats["n_backbone_L"] += int(mt["backbone_ld"] == "L")
            self.stats["n_backbone_D"] += int(mt["backbone_ld"] == "D")
        self.stats["n_dropped"] = len(self.dropped)

        Z = np.stack(rows) if rows else np.zeros((0, self.full_size), dtype=np.float32)
        return Z, metas

    def encode(self, smiles_list: Sequence[str]):
        """Return `torch.FloatTensor[N, 526]` (float32, order-preserving, rows one-to-one with the input)."""
        import torch

        Z, _ = self.encode_meta(smiles_list)
        return torch.from_numpy(Z)

    def encode_numpy(self, smiles_list: Sequence[str]) -> np.ndarray:
        Z, _ = self.encode_meta(smiles_list)
        return Z

    def encode_one(self, smiles: str):
        return self.encode([smiles])[0]

    # -- Convenience --------------------------------------------------------
    def mirror_code(self, c: np.ndarray) -> np.ndarray:
        """Enantiomer chirality code: c -> -c (flips only the chirality channel; leaves z_moler / t untouched)."""
        return -np.asarray(c, dtype=np.float32)

    @property
    def chirality_slice(self) -> slice:
        return slice(self.latent_size, self.latent_size + self.k)

    @property
    def kind_slice(self) -> slice:
        return slice(self.latent_size + self.k, self.full_size)


def load_moler_encoder(
    model_dir: str = DEFAULT_MOLER_MODEL_DIR,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> MolToMoler526:
    """Construct a **long-lived, reusable** 526-dim encoder (the unified entry for the online encoding path).

    It differs from constructing `MolToMoler526` directly in one critical way:

    * MoLeR's `num_workers` workers are `fork`ed inside
      `MoLeRInferenceServer.__init__`; they **inherit the parent's env at that moment**,
      and the parent's `tf.config.set_visible_devices` does not propagate to
      already-forked children. Hence `CUDA_VISIBLE_DEVICES=""` and the single-thread env
      vars **must be set before construction** -- which is exactly what
      `moler_cpu_session()` does.
    * Once the fork is done the env can be restored: from then on TF encoding only
      happens in the worker processes (`MoLeRInferenceServer.encode` dispatches chunks
      to the workers; the parent never runs TF), and torch **reads
      `CUDA_VISIBLE_DEVICES` lazily**, so the first `torch.cuda` call **after this
      function returns** still sees the GPU.

    In short: this function is the timing wrapper for "MoLeR on CPU, torch on GPU in
    the same process". The caller is responsible for `enc.close()` before exiting (or
    using it as a context manager).
    """
    with moler_cpu_session():
        enc = MolToMoler526(model_dir=model_dir, num_workers=num_workers)
    return enc


# =========================================================================== #
# Self-check
# =========================================================================== #
def _selfcheck() -> int:
    print("[selfcheck] reference tables:")
    rep = reference_report()
    print(f"  n_residues={rep['n_residues']} "
          f"n_distinct_L={rep['n_distinct_L_keys']} "
          f"n_distinct_identity={rep['n_distinct_identity_keys']} "
          f"cip_corrected={rep['cip_corrected']}")
    assert rep["n_distinct_L_keys"] == 20
    assert rep["n_distinct_identity_keys"] == 20

    # -- Pure-RDKit part: D/L channel invariants (no MoLeR needed) --
    L, D = "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"
    cL, mL = chirality_code(L, k=8)
    cD, mD = chirality_code(D, k=8)
    print(f"[selfcheck] c(L)={cL[:4].tolist()} c(D)={cD[:4].tolist()}")
    assert np.allclose(cL, -cD), "enantiomer c must be bitwise opposite"
    tL, kL = residue_kind_code(L)
    tD, kD = residue_kind_code(D)
    print(f"[selfcheck] t(L)={tL.tolist()} kind={kL['kind']} ld={kL['backbone_ld']}")
    print(f"[selfcheck] t(D)={tD.tolist()} kind={kD['kind']} ld={kD['backbone_ld']}")
    assert kL["kind"] == "std_aa" and kL["backbone_ld"] == "L"
    assert kD["kind"] == "std_aa" and kD["backbone_ld"] == "D"
    # Semantics of c: re-applying it should be the identity
    back, bad = reapply_stereo(stereo_free(L), cL)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(back)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(L)
    ), "reapply_stereo round-trip mismatch"
    assert bad == 0, f"unresolved={bad}"
    print("[selfcheck] c channel roundtrip ok, t channel ok")

    # -- Part requiring MoLeR --
    nw = int(os.environ.get("SELFCHECK_WORKERS", "8"))
    with moler_cpu_session():
        with MolToMoler526(num_workers=nw) as enc:
            print(f"[selfcheck] latent_size={enc.latent_size} full_size={enc.full_size}")
            assert enc.full_size == FEATURE_DIM, enc.full_size
            zL, zD = enc.encode([L, D])
            zL = zL.numpy(); zD = zD.numpy()
            print(f"[selfcheck] z shape {zL.shape} {zL.dtype}")
            assert zL.shape == (FEATURE_DIM,), zL.shape
            assert zL.dtype == np.float32
            d = np.abs(zL[:512] - zD[:512]).max()
            print(f"[selfcheck] max|zL[:512]-zD[:512]| = {d:.3e} (should be 0: MoLeR has no chirality)")
            assert d == 0.0
            dm = np.abs(zL[512:520] + zD[512:520]).max()
            print(f"[selfcheck] max|zL[512:520]+zD[512:520]| = {dm:.3e} (should be 0)")
            assert dm == 0.0
            assert np.array_equal(zL[520:526], tL)
            # Batch + order preservation
            smis = [L, D, "NCC(=O)O", "O=C(O)[C@@H]1CCCN1"] * 3
            Z = enc.encode(smis)
            print(f"[selfcheck] batch {tuple(Z.shape)} dropped={len(enc.dropped)}")
            assert Z.shape == (len(smis), FEATURE_DIM)
            assert len(enc.dropped) == 0, enc.dropped
            # Order preservation: entry 0 of the batch must equal the single-item zL.
            # Tolerance, not bitwise: MoLeR's chunk_size = min(500, N//num_workers+1)
            # depends on N, so single-item (N=1) and batch (N=12) use different chunks
            # -> different TF reduction order -> ~7e-7 difference in float32. Within a
            # single call, the same molecule appearing repeatedly is bitwise identical (=0).
            d0 = float(np.abs(Z[0].numpy() - zL).max())
            print(f"[selfcheck] order-preservation max|Z[0]-zL| = {d0:.3e} (pass if <1e-5)")
            assert d0 < 1e-5, f"batch not order-preserving: {d0}"
            assert np.array_equal(Z[0].numpy(), Z[4].numpy()), "same molecule in a batch must be bitwise identical"
            print(f"[selfcheck] stats={enc.stats}")
    print("[selfcheck] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
