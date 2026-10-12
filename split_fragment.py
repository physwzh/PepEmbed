#!/usr/bin/env python3
"""Hierarchical fragment splitting for peptides / cyclic peptides (retrosynthesis).

Split a peptide / cyclic-peptide SMILES into residue-level building-block
fragments, each carrying attachment marks ([*:n] dummies), so that the
original molecule can be reconstructed by joining attachments pairwise.

Hierarchical splitting pipeline:
  Tier1   Inter-residue retrosynthetic cuts: cross-residue peptide bonds,
          disulfides, cross-residue thioethers, esters, carbamates, and
          cross-residue ethers.
  Tier2   Blocks that still contain >=2 residue anchors are cut further by
          C-alpha BFS Voronoi, down to <=1 residue per piece; single-anchor
          residues and their internal rings are left intact, and blocks whose
          ring spans multiple C-alphas are skipped whole.
  Tier3   Per-block residue-boundary decision. A single-anchor block is kept
          whole only when it matches a curated modified residue, an exact
          standard template, or the common-residue set; otherwise it falls
          back to residue-core boundary cuts (dropping unrecognized
          decorations, never across a macrocycle bond).
  Tier4   Anchor-less linkers: common linkers are kept whole; composite
          linkers are BRICS-split into units.
  Post    Merge tiny fragments: fragments with <3 heavy atoms are merged back
          into their neighbors.

Usage:
  python split_fragment.py "SMILES"
  python split_fragment.py "SMILES" --json
  python split_fragment.py "SMILES" --no-verify
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import BRICS
from rdkit.Chem import HybridizationType as Hyb

RDLogger.DisableLog("rdApp.*")


# == SMARTS constants =========================================================
PEPTIDE_BOND_SMARTS = "[CX3](=[OX1])[NX3]"
DISULFIDE_SMARTS = "[#6][SX2][SX2][#6]"
ETHER_SMARTS = "[#6][OX2H0][#6]"
THIOETHER_SMARTS = "[#6][SX2][#6]"
ESTER_SMARTS = "[CX3](=[OX1])[OX2H0]"
CARBAMATE_SMARTS = "[OX2][CX3](=[OX1])[NX3]"
BACKBONE_BROAD = "[NX3][CX4][CX3](=[OX1])"
BACKBONE_SMARTS = "[NX3;!$([NX3]C=O)][CH1,CH2][CX3](=[OX1])"


# == Data classes =============================================================
@dataclass
class Attachment:
    att_id: int   # per-fragment index, starting from 1
    att_type: str # amide_n / amide_c / disulfide / thioether / ester / generic
    label: int    # global pairing label (shared by both ends of a cut)


@dataclass
class Fragment:
    canonical_form: str = ""
    attachments: list = field(default_factory=list)
    category: str = "unknown"  # L_std_aa / D_std_aa / std_aa / L_mod_aa / D_mod_aa / mod_aa / non_aa / unknown
    capped_form: str = ""      # SMILES with [*:n] attachment points capped
    chirality: str = ""        # L / D / unknown / none
    role: str = ""             # N_term / C_term / internal / none


# == Atom-level helpers =======================================================
def _is_ca(mol, c_idx: int) -> bool:
    """Residue alpha carbon: tetrahedral carbon adjacent to the backbone N.

    Accepted forms are peptide N-C(alpha)-C(=O) and lactamized N-C(=O)-C(alpha)
    topologies. Do not require sp3 here: an AM1 macrocycle carbonyl can make the
    adjacent alpha carbon appear sp2. The carbon must not itself be a carbonyl
    carbon; this excludes aminoacyl guanidino/ureido groups and side-chain amides.
    """
    c = mol.GetAtomWithIdx(c_idx)
    if c.GetSymbol() != "C":
        return False
    if any(x.GetSymbol() == "O" and x.GetTotalDegree() == 1
           for x in c.GetNeighbors()):
        return False

    has_adjacent_n = any(nb.GetSymbol() == "N" for nb in c.GetNeighbors())
    has_direct_carbonyl = any(
        nb.GetSymbol() == "C" and any(o.GetSymbol() == "O" for o in nb.GetNeighbors())
        for nb in c.GetNeighbors()
        if nb.GetIdx() != c_idx
    )
    if has_adjacent_n and has_direct_carbonyl:
        return True

    # Ring/lactamized residues: C(alpha)-N-C(=O), as in pyroglutamate.
    for n_nb in c.GetNeighbors():
        if n_nb.GetSymbol() != "N" or not n_nb.IsInRing():
            continue
        for next_atom in n_nb.GetNeighbors():
            if next_atom.GetIdx() == c_idx:
                continue
            if (next_atom.GetSymbol() == "C"
                    and any(o.GetSymbol() == "O" and o.GetTotalDegree() == 1
                            for o in next_atom.GetNeighbors())):
                return True
    return False

def _side_has_alpha(mol, start_idx: int, blocked_idx: int) -> bool:
    """Search from an atom, without crossing blocked_idx, for an alpha carbon."""
    seen = {start_idx, blocked_idx}
    queue = [start_idx]
    while queue:
        cur = queue.pop()
        if cur not in (start_idx, blocked_idx) and _is_ca(mol, cur):
            return True
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in seen:
                continue
            seen.add(nb_idx)
            queue.append(nb_idx)
    return False


def _is_carbonyl_c(mol, c_idx: int) -> bool:
    """Carbonyl carbon: sp2 C with degree 3 and a terminal O neighbor."""
    c = mol.GetAtomWithIdx(c_idx)
    if c.GetHybridization() != Hyb.SP2 or c.GetTotalDegree() != 3:
        return False
    return any(nb.GetSymbol() == "O" and nb.GetTotalDegree() == 1
               for nb in c.GetNeighbors())


def _is_cbeta(mol, c_idx: int) -> bool:
    """Side-chain beta carbon: a non-carbonyl carbon directly attached to Calpha."""
    c = mol.GetAtomWithIdx(c_idx)
    if c.GetSymbol() != "C":
        return False
    if any(x.GetSymbol() == "O" and x.GetTotalDegree() == 1
           for x in c.GetNeighbors()):
        return False
    return any(_is_ca(mol, nb.GetIdx()) for nb in c.GetNeighbors())


def _get_component(mol, start_idx: int, exclude_bond=None) -> set:
    """BFS connected component from start_idx, optionally skipping one bond."""
    visited = {start_idx}
    queue = [start_idx]
    while queue:
        cur = queue.pop(0)
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in visited:
                continue
            if exclude_bond is not None:
                bond = mol.GetBondBetweenAtoms(cur, nb_idx)
                if bond and bond.GetIdx() == exclude_bond:
                    continue
            visited.add(nb_idx)
            queue.append(nb_idx)
    return visited


def _fragment_has_backbone(mol, atom_indices: set) -> bool:
    """Check if a set of atoms contains a peptide backbone (N-Calpha-C'=O)."""
    if not atom_indices:
        return False
    atom_list = sorted(atom_indices)
    emol = Chem.RWMol()
    idx_map = {}
    for ai in atom_list:
        orig = mol.GetAtomWithIdx(ai)
        na = Chem.Atom(orig.GetSymbol())
        na.SetFormalCharge(orig.GetFormalCharge())
        idx_map[ai] = emol.AddAtom(na)
    for ai in atom_list:
        for nb in mol.GetAtomWithIdx(ai).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in idx_map and idx_map[nb_idx] > idx_map[ai]:
                ob = mol.GetBondBetweenAtoms(ai, nb_idx)
                emol.AddBond(idx_map[ai], idx_map[nb_idx], ob.GetBondType())
    m = emol.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return False
    return m.HasSubstructMatch(Chem.MolFromSmarts(BACKBONE_BROAD))


# == Tier1: inter-residue retrosynthetic cuts =================================
def _is_intraresidue_amide(mol, c_idx: int, n_idx: int) -> bool:
    """Keep amide bonds whose carbonyl and amide N belong to one residue.

    A peptide carbonyl is owned either by a directly adjacent Calpha or, in a
    lactamized residue, by the Calpha attached to its peptide N. Cut only when
    the amide nitrogen anchors a different residue. This preserves side-chain
    acylation, formyl groups, pyroglutamate and ring-closure C(alpha)-C(=O)
    bonds, while still cutting ordinary and head-tail peptide bonds.
    """

    def alpha_neighbors(atom_idx: int, excluded_idx: int) -> set:
        return {
            nb.GetIdx()
            for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if nb.GetIdx() != excluded_idx and _is_ca(mol, nb.GetIdx())
        }

    n_alphas = alpha_neighbors(n_idx, c_idx)
    if not n_alphas:
        return True
    return False

def _identify_amide_cuts(mol, skip: set) -> list:
    """Cut peptide C(=O)-N bonds, preserving intraresidue acylated nitrogens.

    A bond is cut when the nitrogen has >=1 non-carbonyl heavy neighbor and the
    non-carbonyl side contains a different alpha carbon. This preserves primary
    amides, aminoacylated Lys/Arg side chains, N-formyl-modified amino groups
    and pyroglutamate-like intraring lactams, while still cutting normal
    peptide-chain bonds.
    """
    out = []
    patt = Chem.MolFromSmarts(PEPTIDE_BOND_SMARTS)
    for match in mol.GetSubstructMatches(patt):
        c_idx, n_idx = match[0], match[2]
        n = mol.GetAtomWithIdx(n_idx)
        heavy_outside = sum(1 for nb in n.GetNeighbors()
                            if nb.GetIdx() != c_idx and nb.GetSymbol() != "H")
        if heavy_outside == 0:
            continue
        if _is_intraresidue_amide(mol, c_idx, n_idx):
            continue
        bond = mol.GetBondBetweenAtoms(c_idx, n_idx)
        if bond is None or bond.GetIdx() in skip:
            continue
        out.append((bond.GetIdx(), c_idx, n_idx))
        skip.add(bond.GetIdx())
    return out


def _identify_thioether_cuts(mol, skip: set) -> list:
    """Cut only cross-residue C-S-C bridges, not amino-acid side chains.

    Cys S-modifications and Met-derived side-chain chemistry contain an alpha
    carbon on only one side and are kept as intact residues. Lanthionine-like
    bridges have alpha branches on both sides and remain legitimate
    inter-residue cuts.
    """
    out = []
    patt = Chem.MolFromSmarts(THIOETHER_SMARTS)
    for match in mol.GetSubstructMatches(patt):
        c1, s_idx, c2 = match[0], match[1], match[2]
        if _side_has_alpha(mol, c1, s_idx) and _side_has_alpha(mol, c2, s_idx):
            bond = mol.GetBondBetweenAtoms(c1, s_idx)
            if bond and bond.GetIdx() not in skip:
                out.append((bond.GetIdx(), c1, s_idx))
                skip.add(bond.GetIdx())
    return out


def _identify_ether_cuts(mol, skip: set) -> list:
    """Cut inter-residue ether bonds (C-O-C), excluding ester/carbamate proximity."""
    candidates = []  # (bond_idx, non-backbone-side atom set, bridging O idx)
    patt = Chem.MolFromSmarts(ETHER_SMARTS)
    for match in mol.GetSubstructMatches(patt):
        c1, o, c2 = match[0], match[1], match[2]
        if _is_carbonyl_c(mol, c1) or _is_carbonyl_c(mol, c2):
            continue
        for c_idx in (c1, c2):
            bond = mol.GetBondBetweenAtoms(c_idx, o)
            if bond is None or bond.GetIdx() in skip:
                continue
            comp_c = _get_component(mol, c_idx, exclude_bond=bond.GetIdx())
            comp_o = _get_component(mol, o, exclude_bond=bond.GetIdx())
            bb_c = _fragment_has_backbone(mol, comp_c)
            bb_o = _fragment_has_backbone(mol, comp_o)
            if bb_c != bb_o:
                nb_side = comp_c if not bb_c else comp_o
                candidates.append((bond.GetIdx(), frozenset(nb_side), o))
            elif bb_c and bb_o:
                candidates.append((bond.GetIdx(), frozenset(comp_c), o))
    final = []
    for bi, nb, o_idx in candidates:
        is_subset = any(bi != obi and nb < onb for obi, onb, _ in candidates)
        if not is_subset:
            final.append((bi, nb, o_idx))
    out = []
    for bi, nb, o_idx in final:
        n_c = sum(1 for ai in nb if mol.GetAtomWithIdx(ai).GetSymbol() == "C")
        if n_c <= 2:
            continue
        out.append((bi, list(nb)[0], o_idx))
        skip.add(bi)
    return out


def _identify_generic_bonds(mol, skip: set) -> list:
    """Cut carbamate and ester bonds (for macrocyclization, etc.)."""
    out = []
    patt = Chem.MolFromSmarts(CARBAMATE_SMARTS)
    for match in mol.GetSubstructMatches(patt):
        o, c = match[0], match[1]
        bond = mol.GetBondBetweenAtoms(o, c)
        if bond and bond.GetIdx() not in skip:
            out.append((bond.GetIdx(), o, c))
            skip.add(bond.GetIdx())
    patt = Chem.MolFromSmarts(ESTER_SMARTS)
    for match in mol.GetSubstructMatches(patt):
        c, o = match[0], match[2]
        bond = mol.GetBondBetweenAtoms(c, o)
        if bond and bond.GetIdx() not in skip:
            out.append((bond.GetIdx(), c, o))
            skip.add(bond.GetIdx())
    return out


def _is_known_modified_piece(mol, piece) -> bool:
    """Match a curated capped-form modified residue, ignoring stereo variants.

    Source databases represent the same modified residue with different or
    absent CIP labels. Curated chemistry is what stays whole; stereo is still
    retained by the untouched molecule itself.
    """
    sub, _ = _build_submol(mol, piece, set())
    try:
        capped = cap_fragment(Chem.MolToSmiles(sub))
    except Exception:
        return False
    return capped is not None and _canon_key(capped) in KNOWN_MODIFIED_CAPPED_FORMS


def _tier1_cuts(mol) -> list:
    """Identify all inter-residue retrosynthetic cut bonds."""
    whole = set(range(mol.GetNumAtoms()))
    whole_is_known_modified = _is_known_modified_piece(mol, whole)
    skip = set()
    cuts = []
    for bi, c, n in _identify_amide_cuts(mol, skip):
        if whole_is_known_modified:
            continue
        cuts.append({"bond_idx": bi, "type": "amide", "a": c, "b": n})
    patt = Chem.MolFromSmarts(DISULFIDE_SMARTS)
    for m in mol.GetSubstructMatches(patt):
        s1, s2 = m[1], m[2]
        bond = mol.GetBondBetweenAtoms(s1, s2)
        if whole_is_known_modified or not bond or bond.GetIdx() in skip:
            continue
        cuts.append({"bond_idx": bond.GetIdx(), "type": "disulfide", "a": s1, "b": s2})
        skip.add(bond.GetIdx())
    for bi, a, s in _identify_thioether_cuts(mol, skip):
        if not whole_is_known_modified:
            cuts.append({"bond_idx": bi, "type": "thioether", "a": a, "b": s})
    for bi, a, o in _identify_ether_cuts(mol, skip):
        if not whole_is_known_modified:
            cuts.append({"bond_idx": bi, "type": "generic", "a": a, "b": o})
    for bi, c, o in _identify_generic_bonds(mol, skip):
        if not whole_is_known_modified:
            cuts.append({"bond_idx": bi, "type": "generic", "a": c, "b": o})
    return cuts


# == Tier1.5: recursive ring opening =========================================
def _segment(mol, cut_idxs: set) -> list:
    """Find connected components after removing cut bonds."""
    n = mol.GetNumAtoms()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for b in mol.GetBonds():
        if b.GetIdx() in cut_idxs:
            continue
        union(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    comps = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)
    return [set(v) for v in comps.values()]


def _voronoi(mol, piece, cas) -> dict:
    """Assign each atom to the nearest Calpha (BFS Voronoi partition)."""
    assign = {}
    q = deque()
    for c in cas:
        assign[c] = c
        q.append(c)
    piece = set(piece)
    while q:
        cur = q.popleft()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            nbi = nb.GetIdx()
            if nbi not in piece or nbi in assign:
                continue
            assign[nbi] = assign[cur]
            q.append(nbi)
    return assign


def _tier15(mol, cuts: list) -> list:
    """Break non-cyclic multi-residue blocks by Calpha Voronoi.

    A connected piece containing a ring with multiple alpha anchors is a cyclic
    peptide or crosslink topology. Arbitrary C-C cuts there create context-bound
    vocabulary; such cases must be resolved by explicit inter-residue chemistry.
    Ordinary one-alpha rings (Pro, pGlu, Hyp, aromatic side chains) are unaffected.
    """
    ring_info = mol.GetRingInfo()
    multi_ca_rings = [
        set(ring) for ring in ring_info.AtomRings()
        if sum(1 for a in ring if _is_ca(mol, a)) >= 2
    ]
    cut_idxs = set(c["bond_idx"] for c in cuts)
    while True:
        pieces = _segment(mol, cut_idxs)
        added = False
        for piece in pieces:
            cas = [a for a in piece if _is_ca(mol, a)]
            if len(cas) < 2:
                # One-alpha-carbon components are already intact residues.
                # Tier1.5 must not open their ring lactams or side-chain cycles.
                continue
            if any(ring <= set(piece) for ring in multi_ca_rings):
                continue
            assign = _voronoi(mol, piece, cas)
            for a in piece:
                for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                    b = nb.GetIdx()
                    if b not in piece:
                        continue
                    if assign.get(a) != assign.get(b):
                        bond = mol.GetBondBetweenAtoms(a, b)
                        bi = bond.GetIdx()
                        # Guard: never cut double bonds or aromatic ring bonds
                        # (aromatic order 1.5 would pass a plain <2.0 check and
                        # break rings -> kekulize failure / irreversibility).
                        if (bi not in cut_idxs
                                and bond.GetBondTypeAsDouble() < 2.0
                                and not bond.GetIsAromatic()
                                and bond.GetBondTypeAsDouble() != 1.5):
                            cuts.append({"bond_idx": bi, "type": "generic", "a": a, "b": b})
                            cut_idxs.add(bi)
                            added = True
        if not added:
            break
    return cuts


# == Standard 20 amino-acid SMARTS templates ==================================
def _build_std_smarts_templates() -> dict:
    """Peptidic residue SMARTS templates, ordered by descending atom count.

    These match both terminal (free) and peptide-internal residues because:
      - [NX3] matches N with valence 3 (NH2 at N-term or NH-peptide internally)
      - [CX3](=[OX1]) matches carbonyl C regardless of 3rd substituent

    His: the imidazole ring is written with alternative-atom lists at both
    ring nitrogens ([nX2,nX3]), so a single template matches the Ndelta-H and
    Neps-H neutral tautomers as well as the protonated (imidazolium) form:
    [nH]/[nH+] are X3, pyridine-type [n] is X2.
    """
    smarts_map = {
        "W": "[NX3][CX4]([CX3](=[OX1]))[CX4]c1c[nH]c2ccccc12",
        "Y": "[NX3][CX4]([CX3](=[OX1]))[CX4]c1ccc([OX2])cc1",
        "R": "[NX3][CX4]([CX3](=[OX1]))[CX4][CX4][CX4][NX3][CX3](=[NX2])[NX3]",
        "F": "[NX3][CX4]([CX3](=[OX1]))[CX4]c1ccccc1",
        "H": "[NX3][CX4]([CX3](=[OX1]))[CX4]c1c[nX2,nX3]c[nX2,nX3]1",
        "K": "[NX3][CX4]([CX3](=[OX1]))[CX4][CX4][CX4][CX4][NX3]",
        "E": "[NX3][CX4]([CX3](=[OX1]))[CX4][CX4][CX3](=[OX1])[OX2]",
        "Q": "[NX3][CX4]([CX3](=[OX1]))[CX4][CX4][CX3](=[OX1])[NX3]",
        "D": "[NX3][CX4]([CX3](=[OX1]))[CX4][CX3](=[OX1])[OX2]",
        "N": "[NX3][CX4]([CX3](=[OX1]))[CX4][CX3](=[OX1])[NX3]",
        "M": "[NX3][CX4]([CX3](=[OX1]))[CX4][CX4][SX2][CX4]",
        "L": "[NX3][CX4]([CX3](=[OX1]))[CX4][CX4]([CX4])[CX4]",
        "I": "[NX3][CX4]([CX3](=[OX1]))[CX4]([CX4])[CX4][CX4]",
        "T": "[NX3][CX4]([CX3](=[OX1]))[CX4]([OX2])[CX4]",
        "P": "[NX3]1[CX4]([CX3](=[OX1]))[CX4][CX4][CX4]1",
        "V": "[NX3][CX4]([CX3](=[OX1]))[CX4]([CX4])[CX4]",
        "C": "[NX3][CX4]([CX3](=[OX1]))[CX4][SX2]",
        "S": "[NX3][CX4]([CX3](=[OX1]))[CX4][OX2]",
        "A": "[NX3][CX4]([CX3](=[OX1]))[CX4]",
        "G": "[NX3][CX4]([CX3](=[OX1]))",
    }
    t = {}
    for code, smarts_str in smarts_map.items():
        m = Chem.MolFromSmarts(smarts_str)
        if m is not None:
            t[code] = m
    return t


STD_TEMPLATES = _build_std_smarts_templates()

def _build_modified_smarts_templates() -> dict:
    """Explicit anchors for common modified residues.
    A component with one alpha carbon is preserved by the Tier2 priority rule.
    These templates document phospho, sulfo, acyl, oxidized and ring-modified
    forms and support regression tests without adding residue-internal cuts.
    """
    smarts_map = {
        "pSer": "[NX3][CX4]([CX3]=[OX1])[CX4][OX2][PX4](=[OX1])(=[OX1])[OX2]",
        "pThr": "[NX3][CX4]([CX3]=[OX1])[CX4]([OX2][PX4](=[OX1])(=[OX1])[OX2])[CX4]",
        "sTyr": "[NX3][CX4]([CX3]=[OX1])[CX4]c1ccc([OX2][SX4](=[OX1])(=[OX1]))cc1",
        "pTyr": "[NX3][CX4]([CX3]=[OX1])[CX4]c1ccc([OX2][PX4](=[OX1])(=[OX1])[OX2])cc1",
        "AcK": "[NX3][CX4]([CX3]=[OX1])[CX4][CX4][CX4][CX4][NX3][CX3]=[OX1][CX4]",
        "FoMet": "[NX3][CX4]([CX3]=[OX1])[CX4][CX4][SX2][CX4][CX3]=[OX1]",
        "MetO": "[NX3][CX4]([CX3]=[OX1])[CX4][CX4][SX4](=[OX1])[CX4]",
        "Cit": "[NX3][CX4]([CX3]=[OX1])[CX4][CX4][CX4][NX3][CX3]=[OX1][NX3]",
        "pGlu": "[NX3][CX4]([CX3]=[OX1])[CX4][CX4][CX4][NX3]1C(=O)CC1=O",
        "Hyp": "[NX3][CX4]([CX3]=[OX1])[CX4]1[OX2][CX4][CX4][CX4]1",
    }
    # Full capped molecules are the authoritative curated forms. The compact
    # SMARTS above remain useful documentation, but RDKit rejects several of
    # them because [CX3] is used where a query needs a carbonyl-carbon atom
    # list; they are deliberately not used for splitting decisions.
    capped_forms = [
        "N[C@@H](COP(=O)(O)O)C(=O)O",
        "C[C@H](OP(=O)(O)O)[C@H](N)C(=O)O",
        "N[C@@H](Cc1ccc(S(=O)O)cc1)C(=O)O",
        "N[C@@H](Cc1ccc(OP(=O)(O)O)cc1)C(=O)O",
        "CC(=O)NCCCC[C@H](N)C(=O)O",
        "CSCC[C@H](NC=O)C(=O)O",
        "C[S+](C)CC[C@H](N)C(=O)[O-]",
        "N=C(N)CCC[C@H](N)C(=O)O",
        "O=C1CCC(C(=O)O)N1",
        "N[C@@H]1CC[C@H](O)C1",
    ]
    templates = {}
    for smiles in capped_forms:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            templates[smiles] = mol
    for code, smarts_str in smarts_map.items():
        template = Chem.MolFromSmarts(smarts_str)
        if template is not None:
            templates[code] = template
    return templates
MODIFIED_TEMPLATES = _build_modified_smarts_templates()


# == Tier2/Tier3: modified residues & intra-linker splitting ==================
def _build_submol(mol, atom_set, cut_idxs):
    """Build a subgraph molecule (skipping cut bonds). Returns (submol, idx_map)."""
    atom_set = set(atom_set)
    idx_map = {}
    emol = Chem.RWMol()
    for ai in atom_set:
        orig = mol.GetAtomWithIdx(ai)
        na = Chem.Atom(orig.GetSymbol())
        na.SetFormalCharge(orig.GetFormalCharge())
        idx_map[ai] = emol.AddAtom(na)
    for b in mol.GetBonds():
        if b.GetIdx() in cut_idxs:
            continue
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in atom_set and v in atom_set:
            emol.AddBond(idx_map[u], idx_map[v], b.GetBondType())
    sub = emol.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        pass
    for a in sub.GetAtoms():
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return sub, idx_map


def _norm_key(smi: str) -> str:
    """Normalize dummy atom labels [*:n] to [*] for fragment equivalence."""
    if not smi:
        return smi
    return re.sub(r"\[\*:\d+\]", "[*]", smi)


def _canon_key(smi: str) -> str:
    """Structure-level canonical key for fragment equivalence.

    Parses the SMILES and re-canonicalizes with stereochemistry, isotopes and
    dummy-atom map numbers stripped, so that keys produced by different
    pipelines (pass1 fragment canonical_form with numbered dummies and stereo,
    vs _piece_whole_canonical lookup keys) collapse to the same string when
    the underlying fragment graphs are isomorphic. Falls back to plain label
    stripping when the SMILES cannot be parsed.
    """
    if not smi:
        return smi
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return _norm_key(smi)
    em = Chem.RWMol(m)
    for a in em.GetAtoms():
        if a.GetAtomMapNum():
            a.SetAtomMapNum(0)
        if a.GetIsotope():
            a.SetIsotope(0)
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in em.GetBonds():
        if b.GetStereo() != Chem.BondStereo.STEREONONE:
            b.SetStereo(Chem.BondStereo.STEREONONE)
    out = em.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(out)
    except Exception:
        return _norm_key(smi)


# Cache for normalized whitelist lookups, keyed by container identity. The
# container object itself is held strongly so its id() cannot be recycled
# while cached; on a hit the identity is re-verified. Callers must not mutate
# the collection after passing it to decompose().
_LOOKUP_SET_CACHE = {}


def _as_lookup_set(keys):
    """Normalize a common-residues/common-linkers collection into lookup-key
    form (bare-dummy structural canonical, stereo-blind) once per container.

    common_sets.json stores pass1 whole-fragment canonical_form keys (numbered
    dummies [*:n], stereo present, pass1 atom ordering). Comparing those
    verbatim against _piece_whole_canonical lookup keys never matches; both
    sides are therefore normalized through _canon_key. The result is a
    frozenset, which also turns the whitelist membership test from O(n) scan
    into O(1) hash lookup.
    """
    if keys is None:
        return None
    ent = _LOOKUP_SET_CACHE.get(id(keys))
    if ent is not None and ent[0] is keys:
        return ent[1]
    out = frozenset(_canon_key(k) for k in keys if k)
    _LOOKUP_SET_CACHE[id(keys)] = (keys, out)
    return out


def _match_residue_in_mol(mol, piece):
    """Match standard 20 residues restricted to piece atoms.

    Returns (code, matched_atom_set, exact). exact=True means the template
    covers the entire piece; exact=False means standard residue + extra atoms.
    """
    piece = set(piece)
    for code, tmpl in STD_TEMPLATES.items():
        for m in mol.GetSubstructMatches(tmpl):
            ms = set(m)
            if ms <= piece and len(ms) == len(piece):
                return code, ms, True
    for code, tmpl in STD_TEMPLATES.items():
        for m in mol.GetSubstructMatches(tmpl):
            ms = set(m)
            if ms <= piece:
                return code, ms, False
    return None, None, False


def _residue_core_atoms(mol, piece, ca) -> set:
    """Determine residue core atoms: backbone + full standard sidechain.

    Tries matching standard SMARTS templates anchored at the given Calpha.
    If matched, the full standard residue atom set becomes the core (including
    complete sidechain, e.g. Cys Cbeta-Sgamma). Falls back to manual backbone
    + Cbeta heteroatom safeguard for non-standard residues.
    """
    piece = set(piece)
    for code, tmpl in STD_TEMPLATES.items():
        for m in mol.GetSubstructMatches(tmpl):
            ms = set(m)
            if ca in ms and ms <= piece:
                core = set(ms)
                ri = mol.GetRingInfo()
                for ring in ri.AtomRings():
                    rs = set(ring)
                    if rs & core:
                        core.update(rs)
                return core
    ca_a = mol.GetAtomWithIdx(ca)
    n_atom = c_prime = cb = None
    carbonyl_neighbors = []
    for nb in ca_a.GetNeighbors():
        if nb.GetSymbol() == "N":
            n_atom = nb.GetIdx()
        if (nb.GetSymbol() == "C"
                and any(x.GetSymbol() == "O" and x.GetTotalDegree() == 1
                        for x in nb.GetNeighbors())):
            carbonyl_neighbors.append(nb.GetIdx())
    # Some cyclic-SMILES parsers represent the closure carbonyl as AM1; any
    # such carbon adjacent to a peptide Calpha is its C-prime, not a sidechain.
    c_prime = carbonyl_neighbors[0] if carbonyl_neighbors else None
    for nb in ca_a.GetNeighbors():
        if nb.GetSymbol() == "C" and nb.GetIdx() != c_prime:
            cb = nb.GetIdx()
            break
    core = {ca}
    if n_atom is not None:
        core.add(n_atom)
    if c_prime is not None:
        core.add(c_prime)
        core.update(n for n in mol.GetAtomWithIdx(c_prime).GetNeighbors()
                    if n.GetSymbol() == "O")
        for nb in mol.GetAtomWithIdx(c_prime).GetNeighbors():
            if nb.GetSymbol() == "O":
                core.add(nb.GetIdx())
    if cb is not None:
        core.add(cb)
        for nb in mol.GetAtomWithIdx(cb).GetNeighbors():
            if nb.GetIdx() != ca and nb.GetSymbol() in ("S", "O", "N"):
                core.add(nb.GetIdx())
    return core


def _bond_cut_type(mol, u, v) -> str:
    a, b = mol.GetAtomWithIdx(u), mol.GetAtomWithIdx(v)

    def carbonyl(c):
        return c.GetSymbol() == "C" and any(x.GetSymbol() == "O" and x.GetTotalDegree() == 1
                                            for x in c.GetNeighbors())
    if a.GetSymbol() == "C" and b.GetSymbol() == "N" and carbonyl(a):
        return "amide"
    if a.GetSymbol() == "N" and b.GetSymbol() == "C" and carbonyl(b):
        return "amide"
    if a.GetSymbol() == "C" and b.GetSymbol() == "O" and carbonyl(a):
        return "ester"
    if a.GetSymbol() == "O" and b.GetSymbol() == "C" and carbonyl(b):
        return "ester"
    if (a.GetSymbol() == "C" and b.GetSymbol() == "S") or (a.GetSymbol() == "S" and b.GetSymbol() == "C"):
        return "thioether" if (_is_cbeta(mol, u) or _is_cbeta(mol, v)) else "generic"
    return "generic"


def _add_cut(cuts, cut_idxs, bond, ctype):
    bi = bond.GetIdx()
    if bi in cut_idxs:
        return
    if bond.GetIsAromatic():
        return
    if (bond.GetBondTypeAsDouble() == 1.5
            and not (bond.GetBeginAtom().GetIsAromatic()
                     or bond.GetEndAtom().GetIsAromatic())):
        bond.SetBondType(Chem.BondType.SINGLE)
    cuts.append({"bond_idx": bi, "type": ctype,
                 "a": bond.GetBeginAtomIdx(), "b": bond.GetEndAtomIdx()})
    cut_idxs.add(bi)


def _piece_whole_canonical(mol, piece_atoms, cut_idxs) -> str:
    """Canonical lookup key of a fragment (with boundary dummies).

    Dummies are added without map numbers and the submol carries no stereo
    (stripped in _build_submol), so the key is directly comparable to the
    _canon_key-normalized whitelist entries.
    """
    sub, amap = _build_submol(mol, piece_atoms, cut_idxs)
    rw = Chem.RWMol(sub)
    for bi in cut_idxs:
        b = mol.GetBondWithIdx(bi)
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (u in piece_atoms) == (v in piece_atoms):
            continue
        p_in = u if u in piece_atoms else v
        if p_in not in amap:
            continue
        didx = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(amap[p_in], didx, Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(out)
    except Exception:
        return ""


def _linker_brics_cuts(mol, piece) -> list:
    """Find BRICS cut points within a linker piece, mapped to original mol bonds."""
    sub, amap = _build_submol(mol, piece, set())
    inv = {v: k for k, v in amap.items()}
    try:
        cuts = BRICS.FindBRICSBonds(sub)
    except Exception:
        return []
    out = []
    for cut in cuts:
        (u, v) = cut[0]
        if u in inv and v in inv:
            b = mol.GetBondBetweenAtoms(inv[u], inv[v])
            if b is not None:
                out.append(b.GetIdx())
    return out


def _should_split_linker(mol, piece) -> bool:
    """Decide whether a linker piece should be further split by BRICS."""
    sub, _ = _build_submol(mol, piece, set())
    try:
        cuts = BRICS.FindBRICSBonds(sub)
    except Exception:
        return False
    if not cuts:
        return False
    bond_idxs = []
    for cut in cuts:
        (u, v) = cut[0]
        b = sub.GetBondBetweenAtoms(u, v)
        if b is not None:
            bond_idxs.append(b.GetIdx())
    if not bond_idxs:
        return False
    fm = Chem.FragmentOnBonds(sub, tuple(bond_idxs), addDummies=True)
    nfrags = len(Chem.GetMolFrags(fm, asMols=True, sanitizeFrags=True))
    heavy = sub.GetNumHeavyAtoms()
    return nfrags >= 3 or (nfrags >= 2 and heavy > 10)


KNOWN_MODIFIED_CAPPED_FORMS = {
    _canon_key(Chem.MolToSmiles(mol))
    for smi in MODIFIED_TEMPLATES
    if (mol := Chem.MolFromSmiles(smi)) is not None
}


def _matches_known_modified(mol, piece, cut_idxs) -> bool:
    """Recognize curated modified residues by capped whole-molecule forms."""
    sub, _ = _build_submol(mol, piece, cut_idxs)
    try:
        Chem.SanitizeMol(sub)
        capped = cap_fragment(Chem.MolToSmiles(sub))
    except Exception:
        return False
    return capped is not None and capped in KNOWN_MODIFIED_CAPPED_FORMS


def _decide_and_add_cuts(mol, cuts, common_residues, common_linkers,
                         linker_split, keep_whole) -> list:
    """Resolve residue blocks: curated whole forms, then old core-boundary cuts."""
    if keep_whole:
        return cuts
    cuts = [dict(c) for c in cuts]
    cut_idxs = set(c["bond_idx"] for c in cuts)
    pieces = _segment(mol, cut_idxs)
    for piece in pieces:
        cas = [a for a in piece if _is_ca(mol, a)]
        if len(cas) == 1:
            ca = cas[0]
            if _matches_known_modified(mol, piece, cut_idxs):
                continue

            code, matched, exact = _match_residue_in_mol(mol, piece)
            if code is not None and exact:
                continue

            whole = _piece_whole_canonical(mol, piece, cut_idxs)
            if (common_residues and whole
                    and _canon_key(whole) in common_residues):
                continue

            # An unrecognized single-alpha component is not a license to absorb
            # arbitrary decorations. Restore the old residue-core boundary rule.
            if code is not None and not exact:
                matched = set(matched)
                for b in mol.GetBonds():
                    if b.GetIdx() in cut_idxs:
                        continue
                    u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    if ((u in matched and v in piece and v not in matched)
                            or (v in matched and u in piece and u not in matched)):
                        _add_cut(cuts, cut_idxs, b, "generic")
                continue

            core = _residue_core_atoms(mol, piece, ca)
            # Keep old core expansion for ordinary sidechain rings, but never
            # traverse a macrocycle bond that Tier1 has already identified as a
            # residue-residue boundary; otherwise one core absorbs the cycle.
            protected_bonds = {
                c["bond_idx"] for c in cuts
                if c["type"] in ("amide", "ester", "generic")
            }
            ri = mol.GetRingInfo()
            for ring in ri.AtomRings():
                rs = set(ring)
                if rs & core and not any(
                    mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetIdx()
                    in protected_bonds
                    for i in range(len(ring))
                ):
                    core.update(rs)
            for b in mol.GetBonds():
                if b.GetIdx() in cut_idxs:
                    continue
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if ((u in core and v in piece and v not in core)
                        or (v in core and u in piece and u not in core)):
                    _add_cut(cuts, cut_idxs, b, _bond_cut_type(mol, u, v))
        elif len(cas) == 0:
            if not linker_split:
                continue
            whole = _piece_whole_canonical(mol, piece, cut_idxs)
            if common_linkers and whole in common_linkers:
                continue
            if _should_split_linker(mol, piece):
                for bi in _linker_brics_cuts(mol, piece):
                    if bi not in cut_idxs:
                        _add_cut(cuts, cut_idxs, mol.GetBondWithIdx(bi), "generic")
    return cuts

# == Post-processing: merge tiny fragments ===================================
def _merge_tiny_fragments(mol, cuts, min_heavy=3):
    """Merge sub-minimum fragments back into neighbors.

    A fragment is skipped when removing a separating cut would delete a
    residue anchor from a molecule that still has another anchor: that would
    re-create a multi-residue block in cyclic peptides.
    """
    while True:
        cut_idxs = set(c["bond_idx"] for c in cuts)
        pieces = _segment(mol, cut_idxs)
        merged = False
        for piece in pieces:
            n_heavy = sum(1 for a in piece
                          if mol.GetAtomWithIdx(a).GetSymbol() != "H")
            if n_heavy >= min_heavy:
                continue
            separating_cuts = []
            for c in cuts:
                b = mol.GetBondWithIdx(c["bond_idx"])
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if (u in piece) != (v in piece):
                    separating_cuts.append(c)
            if not separating_cuts:
                continue
            all_cas = {a for a in range(mol.GetNumAtoms()) if _is_ca(mol, a)}
            anchored = len(all_cas) > 1
            for c in separating_cuts:
                other_cut_idxs = {
                    x["bond_idx"] for x in cuts
                    if x["bond_idx"] != c["bond_idx"]
                }
                # Compute anchors of the post-merge component containing this
                # piece; if another anchored component remains, merging is safe.
                component = _segment(mol, other_cut_idxs)
                target = next(g for g in component if piece <= g)
                in_target_cas = sum(1 for a in target if a in all_cas)
                if anchored and in_target_cas == len(all_cas):
                    continue
                cuts.remove(c)
                merged = True
                break
            if merged:
                break
        if not merged:
            break
    # Tiny-fragment merging can make an earlier boundary cut internal again:
    # for example, an aromatic-ring generic cut is consumed into a residue when
    # a 1-2 atom decoration is rejoined. Stale cuts then break aromatic rings
    # during FragmentOnBonds, so retain only true inter-component boundaries.
    while True:
        cut_idxs = {c["bond_idx"] for c in cuts}
        pieces = _segment(mol, cut_idxs)
        owner = {}
        for i, piece in enumerate(pieces):
            owner.update((a, i) for a in piece)
        stale = [
            c for c in cuts
            if owner[mol.GetBondWithIdx(c["bond_idx"]).GetBeginAtomIdx()]
            == owner[mol.GetBondWithIdx(c["bond_idx"]).GetEndAtomIdx()]
        ]
        if not stale:
            return cuts
        cuts = [c for c in cuts if c not in stale]


def _cut_and_normalize(mol, cut_records: list):
    """Fragment the molecule at cut bonds and normalize each fragment's SMILES."""
    if not cut_records:
        smi = Chem.MolToSmiles(mol)
        return [Fragment(canonical_form=smi, attachments=[])], []
    bond_idxs = tuple(c["bond_idx"] for c in cut_records)
    dummy_labels = [(2 * i + 1, 2 * i + 2) for i in range(len(cut_records))]
    label_to_cut = {}
    for i, rec in enumerate(cut_records):
        label_to_cut[2 * i + 1] = rec["type"]
        label_to_cut[2 * i + 2] = rec["type"]
    frag_mol = Chem.FragmentOnBonds(mol, bond_idxs, addDummies=True,
                                     dummyLabels=dummy_labels)
    frag_mols = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    label_location = {}
    fragments = []
    for fi, fm in enumerate(frag_mols):
        atts = []
        att_id = 1
        for atom in fm.GetAtoms():
            if atom.GetAtomicNum() != 0:
                continue
            label = atom.GetIsotope()
            if label == 0:
                continue
            ct = label_to_cut.get(label, "generic")
            atts.append(Attachment(att_id=att_id, att_type=ct, label=label))
            label_location[label] = (fi, att_id)
            att_id += 1
        fragments.append(Fragment(canonical_form="", attachments=atts))
    edges = []
    for i, rec in enumerate(cut_records):
        la, lb = 2 * i + 1, 2 * i + 2
        if la in label_location and lb in label_location:
            fi, ai = label_location[la]
            fj, aj = label_location[lb]
            edges.append((fi, ai, fj, aj, rec["type"]))
    for fi, frag in enumerate(fragments):
        fm = frag_mols[fi]
        canonical, new_atts = _normalize_frag(fm, frag.attachments)
        frag.canonical_form = canonical
        frag.attachments = new_atts
    return fragments, edges


def _normalize_frag(frag_mol, atts: list):
    """Normalize fragment SMILES: set atom map numbers on dummies, refine amide types."""
    emol = Chem.RWMol(frag_mol)
    new_atts = []
    att_id = 1
    for atom in emol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        nb = atom.GetNeighbors()[0]
        orig_att = next((a for a in atts if a.att_id == att_id), None)
        att_type = orig_att.att_type if orig_att else "generic"
        if att_type == "amide":
            att_type = "amide_n" if nb.GetSymbol() == "N" else "amide_c"
        atom.SetAtomMapNum(att_id)
        atom.SetIsotope(0)
        new_atts.append(Attachment(att_id=att_id, att_type=att_type,
                                    label=orig_att.label if orig_att else 0))
        att_id += 1
    out = emol.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return Chem.MolToSmiles(frag_mol), atts
    return Chem.MolToSmiles(out), new_atts


def _classify_fragment(frag: Fragment) -> None:
    mol = Chem.MolFromSmiles(frag.canonical_form)
    if mol is None:
        frag.category = "unknown"
        return
    backbone = Chem.MolFromSmarts(BACKBONE_SMARTS)
    frag.category = "ncaa" if mol.HasSubstructMatch(backbone) else "non_aa"


def _ca_cip(mol):
    """Find the Calpha of a (capped) residue and return its CIP code plus a
    flag for a Cys-like sidechain (S/Se on the beta branch), which flips the
    L-reference from S to R (e.g. L-Cys is R at Calpha)."""
    ca_idx = None
    for a in mol.GetAtoms():
        if _is_ca(mol, a.GetIdx()):
            ca_idx = a.GetIdx()
            break
    if ca_idx is None:
        return None, False
    ca = mol.GetAtomWithIdx(ca_idx)
    cip = ca.GetProp("_CIPCode") if ca.HasProp("_CIPCode") else None
    cys_like = False
    for nb in ca.GetNeighbors():
        if nb.GetSymbol() == "N":
            continue
        if (nb.GetSymbol() == "C" and
                any(x.GetSymbol() == "O" and x.GetTotalDegree() == 1
                    for x in nb.GetNeighbors())):
            continue
        if nb.GetSymbol() in ("S", "Se"):
            cys_like = True
        else:
            for nb2 in nb.GetNeighbors():
                if nb2.GetIdx() == ca_idx:
                    continue
                if nb2.GetSymbol() in ("S", "Se"):
                    cys_like = True
    return cip, cys_like


def _classify_residue(capped_smiles):
    """Classify a fragment by its capped SMILES into a 5-class scheme.

    Returns (category, chirality):
      category: L_std_aa / D_std_aa / std_aa / L_mod_aa / D_mod_aa /
                mod_aa / non_aa / unknown
      chirality: L / D / unknown / none
    `std_aa` / `mod_aa` (without L/D) are used when the Calpha chirality is
    undefined in the input.

    A standard residue is recognized by a standard-template substructure match
    whose only extra heavy atom (beyond the template, which models the residue
    inside a peptide) is the terminal carboxyl OH introduced by capping.
    """
    if not capped_smiles:
        return "unknown", "unknown"
    mol = Chem.MolFromSmiles(capped_smiles)
    if mol is None:
        return "unknown", "unknown"
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    except Exception:
        pass
    backbone = mol.HasSubstructMatch(Chem.MolFromSmarts(BACKBONE_SMARTS))
    n_atoms = mol.GetNumAtoms()
    code, matched, _ = _match_residue_in_mol(mol, set(range(n_atoms)))
    if matched is not None:
        boundary_dummies = [
            a for a in mol.GetAtoms()
            if a.GetAtomicNum() == 0
            and any(nb.GetIdx() in matched for nb in a.GetNeighbors())
        ]
        backbone = backbone or bool(boundary_dummies)
    # A free amino acid adds one carboxyl OH. A capped residue fragment adds
    # one OH at each backbone boundary (e.g. N-acetyl Ala), so allow one or two
    # boundary oxygens while requiring the standard residue core.
    if matched is None:
        return "non_aa", "none"
    extra_heavy = mol.GetNumHeavyAtoms() - len(matched)
    is_standard = code is not None and extra_heavy in (1, 2)
    cip, cys_like = _ca_cip(mol)
    if cip is None:
        chir = "unknown"
    else:
        ref = "R" if cys_like else "S"
        chir = "L" if cip == ref else "D"
    if is_standard:
        if chir == "L":
            return "L_std_aa", chir
        if chir == "D":
            return "D_std_aa", chir
        return "std_aa", chir
    if chir == "L":
        return "L_mod_aa", chir
    if chir == "D":
        return "D_mod_aa", chir
    return "mod_aa", chir


def _assign_roles(fragments, edges) -> None:
    """Assign each fragment a peptide-chain role using backbone connectivity.

    N_term : N-terminal residue (free or acylated N; its C participates in a
             peptide bond, i.e. it has an amide_c)
    C_term : C-terminal residue (free or esterified C; has an amide_n)
    internal : backbone residue with both termini in the peptide chain
    none   : non-amino-acid fragment (linker / cap / sidechain-only)

    Determined from amide attachments plus the backbone-backbone amide
    adjacency, so an N-acylated terminal residue is still recognized as N_term
    (one amide goes to a non-backbone cap, the other into the chain).
    """
    backbone = {i for i, f in enumerate(fragments)
                if f.category not in ("non_aa", "unknown")}
    bb_att = {i: set() for i in backbone}
    for (fi, ai, fj, aj, etype) in edges:
        if etype != "amide":
            continue
        if fi in backbone and fj in backbone:
            bb_att[fi].add(ai)
            bb_att[fj].add(aj)
    for i, f in enumerate(fragments):
        if i not in backbone:
            f.role = "none"
            continue
        types = {a.att_type for a in f.attachments}
        has_n = "amide_n" in types
        has_c = "amide_c" in types
        if has_n and has_c:
            bdeg = len(bb_att[i])
            if bdeg >= 2:
                f.role = "internal"
            elif bdeg == 1:
                connected = bb_att[i]
                if any(a.att_type == "amide_c" for a in f.attachments
                       if a.att_id in connected):
                    f.role = "N_term"   # C-side in chain, N capped
                else:
                    f.role = "C_term"   # N-side in chain, C capped
            else:
                f.role = "internal"     # both ends capped by non-backbone
        elif has_c:
            f.role = "N_term"
        elif has_n:
            f.role = "C_term"
        else:
            f.role = "none"


def to_graph_record(fragments, edges, original_smiles: str = ""):
    """Build a standardized per-molecule graph record for downstream GNN use.

    Node fields: id, smiles (cut SMILES with [*:n] dummies), capped_smiles,
    category, chirality, role, attachments[{att_id, att_type}].
    Edge fields: src, src_att, dst, dst_att, bond_type, and for amide bonds a
    `direction` (C_to_N / N_to_C) indicating which end contributes the
    carbonyl carbon. The atom-level connection is the dummy [*:att_id] in the
    fragment's `smiles` (src_att on src, dst_att on dst).
    """
    nodes = []
    for i, f in enumerate(fragments):
        nodes.append({
            "id": i,
            "smiles": f.canonical_form,
            "capped_smiles": f.capped_form,
            "category": f.category,
            "chirality": f.chirality,
            "role": f.role,
            "attachments": [
                {"att_id": a.att_id, "att_type": a.att_type}
                for a in f.attachments
            ],
        })
    elist = []
    for (fi, ai, fj, aj, etype) in edges:
        sa = next((a for a in fragments[fi].attachments if a.att_id == ai), None)
        da = next((a for a in fragments[fj].attachments if a.att_id == aj), None)
        rec = {
            "src": fi, "src_att": ai,
            "dst": fj, "dst_att": aj,
            "bond_type": etype,
        }
        if etype == "amide" and sa is not None and da is not None:
            rec["direction"] = "C_to_N" if sa.att_type == "amide_c" else "N_to_C"
        elist.append(rec)
    return {
        "smiles": original_smiles,
        "n_fragments": len(nodes),
        "n_edges": len(elist),
        "fragments": nodes,
        "edges": elist,
    }


def _has_separable_alpha_boundary(mol, cuts):
    """Quick gate: at least one direct C(alpha) bond can isolate a residue anchor."""
    base = {c["bond_idx"] for c in cuts}
    for piece in _segment(mol, base):
        for ca in [a for a in piece if _is_ca(mol, a)]:
            for nb in mol.GetAtomWithIdx(ca).GetNeighbors():
                bond = mol.GetBondBetweenAtoms(ca, nb.GetIdx())
                if (bond is None or bond.GetIdx() in base
                        or bond.GetIsAromatic()
                        or bond.GetBondTypeAsDouble() != 1.0):
                    continue
                sizes = [
                    sum(1 for a in part if _is_ca(mol, a))
                    for part in _segment(mol, base | {bond.GetIdx()})
                ]
                if 1 in sizes and max(sizes) >= 2:
                    return True
    return False


def _explicit_alpha_boundary_fallback(mol, base_cuts, full_cuts):
    """Restart from Tier1/Tier1.5 and cut only local C(alpha) boundaries.

    The temporary-corpus experiment established that a deliberately unfinished
    greedy plan is reversible: an aromatic/alkyl topology can contain a small
    two-Calpha block that has no safe direct boundary.  In that case retain the
    block rather than invent extra cuts.  Final round-trip validation remains
    mandatory in `decompose`.
    """
    base_by_idx = {c["bond_idx"]: dict(c) for c in base_cuts}
    cuts = [dict(c) for c in base_by_idx.values()]
    cut_idxs = set(base_by_idx)

    def alpha_counts_after(bond_idx):
        return [
            sum(1 for a in piece if _is_ca(mol, a))
            for piece in _segment(mol, cut_idxs | {bond_idx})
        ]

    while True:
        targets = []
        for piece in _segment(mol, cut_idxs):
            cas = [a for a in piece if _is_ca(mol, a)]
            if len(cas) >= 2:
                targets.append((piece, cas))
        if not targets:
            return cuts

        chosen = None
        for piece, cas in targets:
            candidates = []
            for ca in cas:
                for nb in mol.GetAtomWithIdx(ca).GetNeighbors():
                    bond = mol.GetBondBetweenAtoms(ca, nb.GetIdx())
                    if (bond is None or bond.GetIdx() in cut_idxs
                            or bond.GetIsAromatic()
                            or bond.GetBondTypeAsDouble() != 1.0):
                        continue
                    counts = alpha_counts_after(bond.GetIdx())
                    # This is intentionally global, matching the validated
                    # experiment: it admits a useful local opening even when
                    # another topology remains as an inseparable two-Calpha
                    # block.  Prefer the cut that removes the smallest part.
                    if 1 in counts and max(counts) >= 2:
                        # Isolate the side of this cut with exactly one C(alpha);
                        # global counts may include one-Calpha components in
                        # unrelated pieces, so do not zip against them.
                        trial_parts = _segment(mol, cut_idxs | {bond.GetIdx()})
                        local_parts = [
                            part for part in trial_parts if set(part) & set(piece)
                        ]
                        isolated = min(
                            (len(part) for part in local_parts
                             if sum(1 for a in part if _is_ca(mol, a)) == 1),
                            default=10 ** 9,
                        )
                        candidates.append((isolated, len(local_parts), bond.GetIdx(),
                                           ca, nb.GetIdx()))
            if candidates:
                chosen = min(candidates)
                break
        if chosen is None:
            return cuts
        _, _, bond_idx, ca, nb_idx = chosen
        if bond_idx in cut_idxs:
            continue
        cuts.append({"bond_idx": bond_idx, "type": "cyclization",
                     "a": ca, "b": nb_idx})
        cut_idxs.add(bond_idx)

def _merge_unmatched_single_alpha(mol, cuts):
    """Rejoin over-split standard cores broken by non-core aromatic bonds.

    A standard-template match may include only the core while the molecule is
    aromatic-perceived through a macrocycle attachment. If removing one generic
    boundary cut makes the capped core exact again, undo that cut so the local
    decoration stays attached instead of becoming a singleton.
    """
    if not cuts:
        return cuts
    cut_idxs = {c["bond_idx"] for c in cuts}
    all_cas = {a for a in range(mol.GetNumAtoms()) if _is_ca(mol, a)}
    for piece in _segment(mol, cut_idxs):
        cas = [a for a in piece if a in all_cas]
        if len(cas) != 1:
            continue
        code, matched, exact = _match_residue_in_mol(mol, piece)
        if code is None or exact:
            continue
        boundary = []
        for c in cuts:
            b = mol.GetBondWithIdx(c["bond_idx"])
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if (u in piece) != (v in piece) and (
                    u in matched or v in matched):
                boundary.append(c)
        if len(boundary) != 1:
            continue
        target = boundary[0]
        trial = [c for c in cuts if c is not target]
        trial_pieces = _segment(mol, {c["bond_idx"] for c in trial})
        ca_piece = next(g for g in trial_pieces if cas[0] in g)
        ca_code, _, ca_exact = _match_residue_in_mol(mol, ca_piece)
        if ca_exact:
            cuts.remove(target)
    return cuts


# == Assembly verification ===================================================
def _find_dummy_by_mapnum(rw, frag_idx, att_id, offsets):
    start = offsets[frag_idx]
    end = offsets[frag_idx + 1] if frag_idx + 1 < len(offsets) else rw.GetNumAtoms()
    for idx in range(start, end):
        atom = rw.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == att_id:
            return idx
    return None


def reassemble(fragments, edges: list):
    """Reassemble fragments back into a single molecule. Returns canonical SMILES or None."""
    if not edges:
        if len(fragments) == 1:
            m = Chem.MolFromSmiles(fragments[0].canonical_form)
            return Chem.MolToSmiles(m) if m else None
        return None

    mol = Chem.MolFromSmiles(fragments[0].canonical_form)
    if mol is None:
        return None
    offsets = [0]
    for frag in fragments[1:]:
        m = Chem.MolFromSmiles(frag.canonical_form)
        if m is None:
            return None
        offsets.append(mol.GetNumAtoms())
        mol = Chem.CombineMols(mol, m)

    rw = Chem.RWMol(mol)
    # Removing a dummy changes the owner atom's neighbor order. Restore the
    # chiral tag after rejoining so the fragment-local stereo parity survives
    # round-trip assembly (notably ring C-alpha atoms with generic cuts).
    stereo_by_idx = {}
    for (fi, att_i, fj, att_j, edge_type) in edges:
        dummy_i = _find_dummy_by_mapnum(rw, fi, att_i, offsets)
        dummy_j = _find_dummy_by_mapnum(rw, fj, att_j, offsets)
        if dummy_i is None or dummy_j is None:
            continue
        atom_i = rw.GetAtomWithIdx(dummy_i).GetNeighbors()[0]
        atom_j = rw.GetAtomWithIdx(dummy_j).GetNeighbors()[0]
        stereo_by_idx[atom_i.GetIdx()] = atom_i.GetChiralTag()
        stereo_by_idx[atom_j.GetIdx()] = atom_j.GetChiralTag()
        rw.AddBond(atom_i.GetIdx(), atom_j.GetIdx(), Chem.BondType.SINGLE)

    retained = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() != 0]
    removed = {a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0}
    new_idx = {old: new for new, old in enumerate(retained)}
    for idx in sorted(removed, reverse=True):
        rw.RemoveAtom(idx)
    for old, tag in stereo_by_idx.items():
        if old in new_idx:
            rw.GetAtomWithIdx(new_idx[old]).SetChiralTag(tag)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return Chem.MolToSmiles(out)


def verify_reassembly(original_smiles: str, reassembled_smiles: str) -> bool:
    """Verify round-trip: fragment then reassemble should recover the original."""
    o = Chem.MolFromSmiles(original_smiles)
    r = Chem.MolFromSmiles(reassembled_smiles)
    if o is None or r is None:
        return False
    if Chem.MolToSmiles(o) == Chem.MolToSmiles(r):
        return True
    # Reattaching ring bonds can reverse the traversal direction of a chiral
    # ring C-alpha. Canonical SMILES then prints CW/CCW oppositely even though
    # fragment/reassembly preserve local stereo. Clear only atom maps and ask
    # for stereo-aware isomorphism in both directions.
    def unmapped_copy(mol):
        cp = Chem.Mol(mol)
        for atom in cp.GetAtoms():
            atom.SetAtomMapNum(0)
        return cp
    uo, ur = unmapped_copy(o), unmapped_copy(r)
    return uo.HasSubstructMatch(ur) and ur.HasSubstructMatch(uo)


# == Main entry ==============================================================
def cap_fragment(smiles: str, attachments=None) -> str | None:
    """Cap [*:n] attachment points to restore each fragment's natural
    building-block molecule (restoration capping).

    The cap atom is chosen from the cut context so that a fragment becomes
    the standalone molecule it corresponds to as a synthetic building block:

      | dummy on                            | cap | restored terminus            |
      |-------------------------------------|-----|------------------------------|
      | carbonyl C (amide_c/ester/acyl cap) | O   | carboxylic acid C(=O)OH      |
      | N (peptide amide_n)                 | H   | free amine NH2               |
      | O (ester O-side, ether O-side)      | H   | free hydroxyl OH             |
      | S / Se (disulfide, thioether S-side)| H   | free thiol SH / selenol      |
      | C with att_type == "thioether"      | S   | Cys Cbeta-SH (free cysteine) |
      | P, or S bearing a double-bond O     | O   | phosphoric / sulfonic acid   |
      | any other atom (generic C-C cuts)   | H   | alkyl terminus               |

    Without `attachments` the thioether Cbeta-side case cannot be detected
    (the fragment alone cannot distinguish a lost S from a plain C-C cut) and
    falls back to an H cap; all other cases are decided from local context.

    Implementation notes:
      - Dummies parsed from SMILES carry noImplicit=True; restored O/S caps
        must reset it (SetNoImplicit(False)) so the cap atom receives its
        implicit H (OH/SH). Otherwise an H-less neutral [O]/[S] is produced.
      - H caps become explicit H atoms and are removed by RemoveHs, letting
        the neighbor atom regain its implicit hydrogens.
      - Chirality is preserved via in-place atom replacement (neighbor order
        and tetrahedral parity unchanged; CIP is re-perceived on the capped
        molecule).

    Returns canonical SMILES of the capped molecule, or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    att_type_by_id = {}
    if attachments:
        for att in attachments:
            att_type_by_id[att.att_id] = att.att_type

    rw = Chem.RWMol(mol)

    def _nb_has_double_bond_o(n):
        for b in n.GetBonds():
            other = rw.GetAtomWithIdx(b.GetOtherAtomIdx(n.GetIdx()))
            if other.GetSymbol() == "O" and b.GetBondTypeAsDouble() >= 2.0:
                return True
        return False

    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = atom.GetNeighbors()
        if not neighbors:
            atom.SetAtomicNum(1)
            continue
        nb = neighbors[0]
        nb_sym = nb.GetSymbol()
        is_carbonyl = (
            nb_sym == "C"
            and any(x.GetSymbol() == "O" and x.GetTotalDegree() == 1
                    for x in nb.GetNeighbors())
        )
        att_type = att_type_by_id.get(atom.GetAtomMapNum(), "")
        if is_carbonyl or nb_sym == "P" or (nb_sym == "S" and _nb_has_double_bond_o(nb)):
            elem, implicit_h = 8, True    # O cap: COOH / phosphoric / sulfonic
        elif nb_sym in ("N", "O", "S", "Se"):
            elem, implicit_h = 1, False   # H cap: NH2 / OH / SH / SeH
        elif nb_sym == "C" and att_type == "thioether":
            elem, implicit_h = 16, True   # S cap: restore Cys Cbeta-SH
        else:
            elem, implicit_h = 1, False   # H cap: alkyl terminus
        atom.SetAtomicNum(elem)
        if implicit_h:
            atom.SetNoImplicit(False)
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(Chem.RemoveHs(out))
    except Exception:
        return None


def decompose(canonical_smiles: str, common_residues=None, common_linkers=None,
              keep_whole: bool = False, linker_split: bool = True):
    """Decompose a peptide/cyclic peptide SMILES into building-block fragments.

    Args:
        canonical_smiles: Input molecule SMILES.
        common_residues: Set of common residue canonical forms (for Tier2).
            Raw keys (e.g. from common_sets.json, with numbered dummies and
            stereo) are normalized once per container via _as_lookup_set;
            matching is structural and stereo-blind.
        common_linkers:  Set of common linker canonical forms (for Tier3),
            normalized the same way.
        keep_whole:      If True, skip Tier2/Tier3 (pass-1 frequency counting).
        linker_split:    If True, enable linker BRICS splitting in Tier3.

    Returns:
        (fragments, edges) where fragments is a list of Fragment and edges is a
        list of (frag_i, att_i, frag_j, att_j, edge_type) tuples.
    """
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {canonical_smiles}")
    # Macrocycle SMILES that close on C(=O) can parse the closure carbon as
    # AM1/aromatic. Restore only amide C(=O)-N carbons: phenolic aromatic
    # C-OH also has a terminal O, but it has no nitrogen neighbor and must
    # remain aromatic.
    changed = False
    emol = Chem.RWMol(mol)
    for atom in emol.GetAtoms():
        if (atom.GetSymbol() == "C" and atom.GetIsAromatic()
                and atom.GetDegree() == 3):
            has_n = any(nb.GetSymbol() == "N" for nb in atom.GetNeighbors())
            if not has_n:
                continue
            for nb in atom.GetNeighbors():
                if (nb.GetSymbol() == "O" and nb.GetTotalNumHs() == 0
                        and emol.GetBondBetweenAtoms(
                            atom.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() < 2.0):
                    emol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx()).SetBondType(
                        Chem.BondType.DOUBLE)
                    changed = True
    if changed:
        try:
            Chem.SanitizeMol(emol)
            mol = emol.GetMol()
        except Exception:
            pass
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {canonical_smiles}")
    common_residues = _as_lookup_set(common_residues)
    common_linkers = _as_lookup_set(common_linkers)
    tier_base = _tier15(mol, _tier1_cuts(mol))
    cuts = _decide_and_add_cuts(mol, tier_base, common_residues, common_linkers,
                                linker_split, keep_whole)
    cuts = _merge_tiny_fragments(mol, cuts)
    cuts = _merge_unmatched_single_alpha(mol, cuts)
    fragments = None
    try:
        fragments, edges = _cut_and_normalize(mol, cuts)
        assembled = reassemble(fragments, edges)
        if assembled is None or not verify_reassembly(canonical_smiles, assembled):
            raise RuntimeError("roundtrip-mismatch")
    except Exception:
        fallback_cuts = _explicit_alpha_boundary_fallback(mol, tier_base, cuts)

        if fallback_cuts is None:
            raise
        fragments, edges = _cut_and_normalize(mol, fallback_cuts)
        assembled = reassemble(fragments, edges)
        if assembled is None or not verify_reassembly(canonical_smiles, assembled):
            raise RuntimeError("fallback roundtrip-mismatch")
    for frag in fragments:
        frag.capped_form = cap_fragment(frag.canonical_form, frag.attachments) or ""
        frag.category, frag.chirality = _classify_residue(frag.capped_form)
    _assign_roles(fragments, edges)
    return fragments, edges


def main():
    parser = argparse.ArgumentParser(
        description="Decompose a peptide/cyclic peptide SMILES into building-block fragments.")
    parser.add_argument("smiles", help="Input molecule SMILES")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip round-trip reassembly verification")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of plain text")
    args = parser.parse_args()

    fragments, edges = decompose(args.smiles)

    if args.json:
        record = to_graph_record(fragments, edges, args.smiles)
        if not args.no_verify:
            reassembled = reassemble(fragments, edges)
            record["reversible"] = (
                reassembled is not None and
                verify_reassembly(args.smiles, reassembled)
            )
            record["reassembled_smiles"] = reassembled
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(f"input:  {args.smiles}")
        print(f"fragments: {len(fragments)}")
        for i, f in enumerate(fragments):
            atts = ", ".join(f"{a.att_id}:{a.att_type}" for a in f.attachments)
            print(f"  [{i}] {f.canonical_form}")
            print(f"       capped: {f.capped_form}")
            print(f"       category: {f.category} | chirality: {f.chirality} | role: {f.role} | [{atts}]")
        if edges:
            print(f"edges: {len(edges)}")
            for e in edges:
                extra = ""
                if e[4] == "amide":
                    sa = next((a for a in fragments[e[0]].attachments
                               if a.att_id == e[1]), None)
                    extra = " (C->N)" if sa and sa.att_type == "amide_c" else " (N->C)"
                print(f"  frag{e[0]}.att{e[1]} --{e[4]}-- frag{e[2]}.att{e[3]}{extra}")
        if not args.no_verify:
            reassembled = reassemble(fragments, edges)
            ok = (reassembled is not None and
                  verify_reassembly(args.smiles, reassembled))
            print(f"reversible: {'yes' if ok else 'no'}")
            if not ok and reassembled:
                print(f"  reassembled: {reassembled}")


if __name__ == "__main__":
    main()
