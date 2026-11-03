import hashlib

import numpy as np
import pandas as pd
import pytest

from ha.eval import explore_extra as X


def test_sparse_qrels_keeps_highest_grade_then_smallest_hash():
    q = pd.DataFrame({"qid": ["a", "a", "a", "b"], "pid": ["p1", "p2", "p3", "p9"],
                      "grade": [1, 2, 2, 1]})
    out = X.sparse_qrels(q).set_index("qid")["pid"]
    h = {p: hashlib.sha256(p.encode()).hexdigest() for p in ("p2", "p3")}
    assert out["a"] == min(h, key=h.get) and out["b"] == "p9" and len(out) == 2


def test_generic_bootstrap_known_case_and_n():
    rng = np.random.default_rng(0)
    base = np.linspace(0.2, 0.8, 7)
    a = base + rng.normal(0, 0.05, (150, 7))
    good, bad = base + rng.normal(0, 0.05, (150, 7)), base[::-1] + rng.normal(0, 0.05, (150, 7))
    si = {"A": a, "sets": [("OVL", 0, good), ("B0", 0, bad), ("B0", 1, bad)]}
    out = X.bootstrap([si, si], [("OVL", "B0")], b_iter=30, seed=1)
    assert out["comparisons"]["OVL-B0"]["delta"] == pytest.approx(2.0)
    assert out["comparisons"]["OVL-B0"]["p_one_sided"] == 0.0 and out["n"] == 150


def test_dims_selection_uses_gate_and_dimension():
    rng = np.random.default_rng(2)
    n_u = 300
    qids = [f"u{i}" for i in range(n_u)]
    u_emb = rng.normal(size=(n_u, 16)).astype(np.float32)
    u_emb /= np.linalg.norm(u_emb, axis=1, keepdims=True)
    r_emb = u_emb[:60].copy()
    m1 = pd.DataFrame({"qid": qids, "ok": [i != 7 for i in range(n_u)],
                       "F": [3.0 if i % 4 == 0 else 5.0 for i in range(n_u)],
                       "H": rng.integers(1, 6, n_u).astype(float)})
    picked = X.dims_selection(0, qids, u_emb, r_emb, m1, "H", 40)
    assert len(picked) == 40 and 7 not in picked
    assert all(i % 4 != 0 for i in picked)                       # gate F >= 4
    assert m1.loc[picked, "H"].mean() > m1.loc[m1["F"] >= 4, "H"].mean()
    assert X.dims_selection(0, qids, u_emb, r_emb, m1, "H", 40) == picked
