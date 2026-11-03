"""Ranking agreement between query sets (spec §13.3; prereg §1: scipy kendalltau, variant b)."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.stats import kendalltau


def tau_b(x: dict[str, float] | list[float], y: dict[str, float] | list[float]) -> float:
    """Kendall tau-b between two score vectors over the same systems (nan if undefined)."""
    if isinstance(x, dict):
        keys = sorted(x)
        x, y = [x[k] for k in keys], [y[k] for k in keys]  # type: ignore[index]
    res = kendalltau(x, y, variant="b")
    return float(res.statistic) if not math.isnan(res.statistic) else float("nan")


def set_scores(perquery: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    """Mean per-query metric of each system over the queries at positions `idx`."""
    return {s: float(v[idx].mean()) for s, v in perquery.items()}


def pair_disagreements(a: dict[str, float], b: dict[str, float]) -> dict[tuple[str, str], bool]:
    """For every system pair, whether the two score dicts order it differently."""
    out = {}
    for s, t in combinations(sorted(a), 2):
        out[(s, t)] = np.sign(a[s] - a[t]) != np.sign(b[s] - b[t])
    return out
