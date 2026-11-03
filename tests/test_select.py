import math

import numpy as np
import pandas as pd
import pytest

from ha.select import sets as S


def _unit(rng, n, d=16):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_allocate_largest_remainder_caps_and_redistribution():
    assert S.allocate(10, [0.5, 0.3, 0.2], [100, 100, 100]).tolist() == [5, 3, 2]
    assert S.allocate(10, [1 / 3, 1 / 3, 1 / 3], [99, 99, 99]).tolist() == [4, 3, 3]
    q = S.allocate(10, [0.5, 0.3, 0.2], [2, 100, 100])
    assert q.tolist() == [2, 5, 3] and q.sum() == 10
    assert S.allocate(50, [0.5, 0.5], [3, 4]).tolist() == [3, 4]          # total capacity binds
    assert S.allocate(4, [1.0, 0.0], [1, 9]).tolist() == [1, 3]           # pi = 0 stratum fills


def test_stratified_pick_quota_dedupe_and_refill():
    stratum = np.array([0, 0, 0, 1, 1, 1])
    emb = np.eye(6, dtype=np.float32)
    emb[1] = emb[0]                                    # 1 duplicates 0 (cosine 1.0)
    picked = S.stratified_pick([0, 1, 2, 3, 4, 5], stratum, np.array([0.5, 0.5]), 4, emb, 0.9)
    assert picked == [0, 2, 3, 4]                      # 1 skipped as a near-copy of 0
    short = S.stratified_pick([0, 1, 3, 4, 5], stratum, np.array([0.5, 0.5]), 4, emb, 0.9)
    assert sorted(short) == [0, 3, 4, 5]               # stratum 0 ran dry: 1 slot moved to 1


@pytest.fixture
def seed_inputs():
    rng = np.random.default_rng(1)
    n_u = 120
    u = pd.DataFrame({"qid": [f"u{i:03d}" for i in range(n_u)]})
    u_emb, r_emb = _unit(rng, n_u), _unit(rng, 60)
    near = {"u000", "u001"}

    def judged(fs, hs):
        return pd.DataFrame({"qid": u["qid"], "ok": [i != 5 for i in range(n_u)], "F": fs,
                             "H": hs, "S": 4.0, "P": 4.0, "C": None})

    f_iso = [5.0 if i % 3 else 3.0 for i in range(n_u)]                 # 1/3 fail the gate
    f_m1 = [5.0 if i % 2 else 3.0 for i in range(n_u)]                  # 1/2 fail the gate
    iso = judged(f_iso, [float(1 + i % 5) for i in range(n_u)])
    m1 = judged(f_m1, [float(1 + (i * 7) % 5) for i in range(n_u)])
    m1 = m1[~m1["qid"].isin(near)]                                       # never judged
    return u, u_emb, r_emb, near, iso, m1


def test_select_seed_sizes_gate_near_duplicates_and_greedy(seed_inputs):
    u, u_emb, r_emb, near, iso, m1 = seed_inputs
    sets, meta = S.select_seed(0, u, u_emb, r_emb, near, iso, m1, n_a=500)
    m1_pass = {q for q, f, ok in zip(m1["qid"], m1["F"], m1["ok"], strict=True) if f >= 4 and ok}
    assert meta["N"] == meta["M1_passes"] == len(m1_pass) and meta["binding"] == ["M1"]
    assert meta["k"] == 4 and abs(sum(meta["pi"]) - 1) < 1e-3
    for (method, _draw), g in sets.groupby(["method", "draw"]):
        assert len(g) == g["qid"].nunique() <= meta["N"]
        assert not set(g["qid"]) & near                                   # excluded everywhere
        if method in ("B0", "B1"):
            assert len(g) == meta["N"]
    assert sets.query("method in ['B0', 'B1']").groupby("method")["draw"].nunique().eq(20).all()
    m1_set = set(sets.query("method == 'M1'")["qid"])
    assert m1_set <= m1_pass                                              # gate, never failures
    assert "u005" not in set(sets["qid"][sets["method"].isin(["M1", "M1-iso"])])
    idx = [u["qid"].tolist().index(q) for q in sets.query("method == 'M1-iso'")["qid"]]
    sims = u_emb[idx] @ u_emb[idx].T
    assert (sims[~np.eye(len(idx), dtype=bool)] <= S.DEDUPE_COS).all()     # dedupe holds


def test_select_seed_is_deterministic_and_greedy_by_score(seed_inputs):
    u, u_emb, r_emb, near, iso, m1 = seed_inputs
    a, meta = S.select_seed(3, u, u_emb, r_emb, near, iso, m1, n_a=10)
    b, _ = S.select_seed(3, u, u_emb, r_emb, near, iso, m1, n_a=10)
    pd.testing.assert_frame_equal(a, b)
    assert meta["N"] == 10 and meta["binding"] == ["A"]
    sc = S.judge_scores(iso)
    chosen = a.query("method == 'M1-iso'")["qid"]
    assert sc[chosen].min() >= sc[[q for q in sc.index if q not in near]].quantile(0.5)
    c, _ = S.select_seed(4, u, u_emb, r_emb, near, iso, m1, n_a=10)
    assert not a.query("method == 'B0'").reset_index(drop=True).equals(
        c.query("method == 'B0'").reset_index(drop=True))                 # draws differ by seed


def test_judge_scores_failures_and_gate():
    df = pd.DataFrame({"qid": ["a", "b", "c"], "ok": [True, True, False], "F": [5.0, 3.0, 5.0],
                       "H": [4.0, 5.0, 5.0], "S": [4.0, 5.0, 5.0], "P": [4.0, 5.0, 5.0],
                       "C": [None, None, None]})
    sc = S.judge_scores(df)
    assert sc["a"] == 16.0 and sc["b"] == -math.inf and sc["c"] == -math.inf


def test_a_size_reads_labels_only(monkeypatch, tmp_path):
    from ha.data import public

    pd.DataFrame({"qid": ["q1", "q2", "q3"], "split_s0": ["A", "R", "A"]}).to_parquet(
        tmp_path / "splits.parquet")
    monkeypatch.setattr(public, "processed_dir", lambda ds: tmp_path)
    assert S.a_size("x", 0) == 2
