"""Judge aggregation (spec §9.3; prereg §4.7-4.8).

- Permutations: per-dimension median over the successful permutations. C is the median of the
  non-null C values (null if all are null).
- Gate: F >= 4 (fixed a priori). Score: w . (H, S, P, C*), where C* = C, or the mean of H, S, P
  when C is null. The score is -inf when the gate fails.
- Weights: NNLS of human overall realism on (H, S, P, C*) over cal-dev; 5-fold CV Spearman rho.
  Fitted separately for M1 and M1-iso. Equal weights is an ablation.
"""

from __future__ import annotations

import math
from statistics import median

import numpy as np

DIMS = ("F", "H", "S", "P", "C")
SCORE_DIMS = ("H", "S", "P", "C")
GATE_MIN_F = 4
CV_FOLDS, CV_SEED = 5, 17   # fold assignment seed = calibration seed (D-049)


def needs_third(a: dict, b: dict, threshold: int = 2) -> bool:
    """True if any dimension differs by >= threshold between two permutation outputs."""
    for d in DIMS:
        x, y = a.get(d), b.get(d)
        if x is not None and y is not None and abs(x - y) >= threshold:
            return True
    return False


def median_dims(outputs: list[dict]) -> dict:
    out = {}
    for d in DIMS:
        vals = [o[d] for o in outputs if o.get(d) is not None]
        out[d] = float(median(vals)) if vals else None
    return out


def c_star(dims: dict) -> float:
    return dims["C"] if dims.get("C") is not None else float(np.mean([dims[d] for d in "HSP"]))


def passes_gate(dims: dict) -> bool:
    return dims.get("F") is not None and dims["F"] >= GATE_MIN_F


def score(dims: dict, weights: dict[str, float]) -> float:
    if not passes_gate(dims):
        return -math.inf
    x = {"H": dims["H"], "S": dims["S"], "P": dims["P"], "C": c_star(dims)}
    return float(sum(weights[d] * x[d] for d in SCORE_DIMS))


def design_matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[r["H"], r["S"], r["P"], c_star(r)] for r in rows], dtype=float)


def fit_nnls(rows: list[dict], y: list[float]) -> dict:
    """NNLS weights on all rows, plus 5-fold CV Spearman rho (prereg §4.7)."""
    from scipy.optimize import nnls
    from scipy.stats import spearmanr

    x, y = design_matrix(rows), np.asarray(y, dtype=float)
    w, _ = nnls(x, y)
    rng = np.random.default_rng(CV_SEED)
    folds = rng.permutation(len(y)) % CV_FOLDS
    pred = np.empty_like(y)
    for k in range(CV_FOLDS):
        tr, te = folds != k, folds == k
        wk, _ = nnls(x[tr], y[tr])
        pred[te] = x[te] @ wk
    rho = spearmanr(pred, y).statistic
    return {"weights": dict(zip(SCORE_DIMS, map(float, w), strict=True)), "cv_spearman": float(rho),
            "n": int(len(y))}


EQUAL_WEIGHTS = {d: 1.0 for d in SCORE_DIMS}
