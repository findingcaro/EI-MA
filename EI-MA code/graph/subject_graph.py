"""Subject interaction graph G_A (PPMI + mutual top-K + distance <= 2)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import math
import numpy as np


def ppmi_weight(n_ij: int, n_i: int, n_j: int, n_v: int) -> float:
    """w_ij = [log(n_ij * |V| / (n_i * n_j))]_+"""
    if n_ij <= 0 or n_i <= 0 or n_j <= 0 or n_v <= 0:
        return 0.0
    val = math.log((n_ij * n_v) / (n_i * n_j))
    return max(0.0, val)


def build_entity_sets(
    subject_facts: Dict[int, Sequence[Tuple]],
) -> Dict[int, Set[int]]:
    """Map subject -> entities appearing in its local memory K_i."""
    out: Dict[int, Set[int]] = {}
    for sid, facts in subject_facts.items():
        ents: Set[int] = set()
        for fact in facts:
            # fact layout: (s, r, o, qualifiers...) — keep all entity ids present
            for x in fact:
                if isinstance(x, int):
                    ents.add(x)
                elif isinstance(x, (list, tuple)):
                    for y in x:
                        if isinstance(y, int):
                            ents.add(y)
        out[sid] = ents
    return out


def build_subject_graph(
    subject_ids: Sequence[int],
    entity_sets: Dict[int, Set[int]],
    n_entities: int,
    K_A: int = 20,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build undirected weighted adjacency for G_A.
    Returns: adj[i] = [(j, w_ij), ...] after mutual top-K_A pruning.
    """
    raw: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    subs = list(subject_ids)
    sizes = {i: len(entity_sets[i]) for i in subs}

    for a_idx, i in enumerate(subs):
        Vi = entity_sets[i]
        for j in subs[a_idx + 1 :]:
            n_ij = len(Vi & entity_sets[j])
            w = ppmi_weight(n_ij, sizes[i], sizes[j], n_entities)
            if w > 0:
                raw[i].append((j, w))
                raw[j].append((i, w))

    # mutual top-K_A
    top: Dict[int, Set[int]] = {}
    for i, nbrs in raw.items():
        nbrs = sorted(nbrs, key=lambda x: x[1], reverse=True)[:K_A]
        top[i] = {j for j, _ in nbrs}

    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for i, nbrs in raw.items():
        for j, w in nbrs:
            if j in top.get(i, set()) and i in top.get(j, set()):
                adj[i].append((j, w))
    return adj


def shortest_path_dist(
    adj: Dict[int, List[Tuple[int, float]]],
    src: int,
    dst: int,
    max_d: int = 2,
) -> int:
    """BFS distance on G_A, capped at max_d+1 (unreachable)."""
    if src == dst:
        return 0
    q = [src]
    dist = {src: 0}
    while q:
        u = q.pop(0)
        if dist[u] >= max_d:
            continue
        for v, _ in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                if v == dst:
                    return dist[v]
                q.append(v)
    return max_d + 1


def eligible_pairs(
    adj: Dict[int, List[Tuple[int, float]]],
    subjects: Iterable[int],
    max_d: int = 2,
) -> List[Tuple[int, int, int]]:
    """Return list of (i, j, d_A) with d_A <= max_d."""
    subs = list(subjects)
    pairs = []
    for a, i in enumerate(subs):
        for j in subs[a + 1 :]:
            d = shortest_path_dist(adj, i, j, max_d=max_d)
            if d <= max_d:
                pairs.append((i, j, d))
    return pairs


def common_neighbors_topk(
    adj: Dict[int, List[Tuple[int, float]]],
    i: int,
    j: int,
    K_nbr: int = 12,
) -> List[int]:
    """N_ij = Top-K_nbr of Nbr(i) ∩ Nbr(j), ranked by w_ik + w_jk."""
    wi = {v: w for v, w in adj.get(i, [])}
    wj = {v: w for v, w in adj.get(j, [])}
    common = set(wi) & set(wj)
    ranked = sorted(common, key=lambda k: wi[k] + wj[k], reverse=True)
    return ranked[:K_nbr]
