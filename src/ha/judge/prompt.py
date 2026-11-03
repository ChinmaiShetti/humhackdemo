"""Judge prompt rendering (spec §9.2, Appendix B.4; prereg §5.2).

v0 = B.4 verbatim (`judge_v0_system`, `judge_v0_user`). Neighbour order for permutation
`perm_idx` is a seeded shuffle with seed = hash(query_id, perm_idx), independent of the split
seed (D-002 note), so identical neighbour sets give identical prompts across seeds.
M1-iso renders the same template with no neighbours: the neighbour block is omitted.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ha.config import load_config
from ha.prompts import render


def perm_seed(query_id: str, perm_idx: int) -> int:
    return int(hashlib.sha256(f"{query_id}|{perm_idx}".encode()).hexdigest()[:16], 16)


def permute(items: list, query_id: str, perm_idx: int) -> list:
    order = np.random.default_rng(perm_seed(query_id, perm_idx)).permutation(len(items))
    return [items[i] for i in order]


def judge_messages(query_id: str, query: str, passage: str, neighbours: list[str] | None,
                   lang_class: str, domain_description: str, perm_idx: int = 0) -> list[dict]:
    """neighbours=None renders M1-iso (no neighbour block)."""
    p = load_config("judge.yaml")["prompt"]
    neigh = [] if neighbours is None else permute(list(neighbours), query_id, perm_idx)
    user = render(p["user"], domain_description=domain_description, passage=passage,
                  neighbours=neigh, query=query, lang_class=lang_class)
    return [{"role": "system", "content": render(p["system"])},
            {"role": "user", "content": user}]
