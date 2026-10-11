"""Edge-feature encoding (12 dims, plan five fragment_train_strategy.md §2.3.3).

Concatenation of four groups:
  bond_type          5  one-hot: amide / disulfide / thioether / ester / generic
  direction          3  one-hot: C_to_N / N_to_C / none
  is_ring_edge       1  binary: whether the edge lies on a cycle of the fragment graph
  connection_category 3  one-hot: bb_bb / bb_branch / branch_branch

bond_type + direction come from split_fragment.py's graph_record;
is_ring_edge is computed on the graph with union-find;
connection_category is determined by the category of the two endpoints
(whether they are amino acids).
"""
from __future__ import annotations

from typing import List

BOND_TYPES = ["amide", "disulfide", "thioether", "ester", "generic"]
DIRECTIONS = ["C_to_N", "N_to_C", "none"]
CONN_CATS = ["bb_bb", "bb_branch", "branch_branch"]

# Amino-acid category set (output of split_fragment.py's _classify_residue)
AA_CATEGORIES = {"L_std_aa", "D_std_aa", "std_aa", "L_mod_aa", "D_mod_aa", "mod_aa"}

EDGE_DIM = len(BOND_TYPES) + len(DIRECTIONS) + 1 + len(CONN_CATS)  # = 12


def _one_hot(idx, size):
    vec = [0] * size
    vec[idx] = 1
    return vec


def _bond_type_id(bond_type: str) -> int:
    return BOND_TYPES.index(bond_type) if bond_type in BOND_TYPES else BOND_TYPES.index("generic")


def _direction_id(direction) -> int:
    if direction in DIRECTIONS:
        return DIRECTIONS.index(direction)
    return DIRECTIONS.index("none")


def compute_ring_edges(n_fragments: int, edges: List[dict]) -> List[bool]:
    """Union-find test of whether each edge creates a cycle (i.e. whether the
    pair is still connected after removing that edge).

    Returns a bool list as long as edges; True = cycle edge.
    """
    parent = list(range(n_fragments))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    is_ring = []
    for e in edges:
        ra, rb = find(e["src"]), find(e["dst"])
        if ra == rb:
            is_ring.append(True)
        else:
            is_ring.append(False)
            parent[ra] = rb
    return is_ring


def conn_category(src_cat: str, dst_cat: str) -> str:
    is_aa_s = src_cat in AA_CATEGORIES
    is_aa_d = dst_cat in AA_CATEGORIES
    if is_aa_s and is_aa_d:
        return "bb_bb"
    if is_aa_s or is_aa_d:
        return "bb_branch"
    return "branch_branch"


def encode_edge(bond_type: str, direction, is_ring: bool, src_cat: str, dst_cat: str) -> List[float]:
    """Encode a single edge as a 12-dim vector."""
    vec = []
    vec += _one_hot(_bond_type_id(bond_type), len(BOND_TYPES))
    vec += _one_hot(_direction_id(direction), len(DIRECTIONS))
    vec.append(1.0 if is_ring else 0.0)
    vec += _one_hot(CONN_CATS.index(conn_category(src_cat, dst_cat)), len(CONN_CATS))
    assert len(vec) == EDGE_DIM
    return vec


def build_edge_feature_matrix(graph_record: dict):
    """For one molecule graph, return the [n_edges, 12] edge-feature matrix
    (in graph_record['edges'] order)."""
    fragments = graph_record["fragments"]
    edges = graph_record["edges"]
    n = graph_record["n_fragments"]
    cats = {f["id"]: f.get("category", "non_aa") for f in fragments}
    is_ring = compute_ring_edges(n, edges)

    matrix = []
    for e, ring in zip(edges, is_ring):
        matrix.append(encode_edge(
            e.get("bond_type", "generic"),
            e.get("direction"),
            ring,
            cats.get(e["src"], "non_aa"),
            cats.get(e["dst"], "non_aa"),
        ))
    return matrix  # list of lists, n_edges x 12
