import numpy as np
import pytest

from ha.select import variants as V


def test_bands_equal_frequency_on_R_and_shared_edges():
    r = np.linspace(0, 1, 101)
    edges = V.band_edges(r)
    assert len(edges) == 4
    counts = np.bincount(V.bands(r, edges), minlength=5)
    assert counts.min() >= 19 and counts.max() <= 22
    assert V.band_edges(np.array([1.0] * 50 + [0.5] * 50)).tolist() == [0.5, 1.0]   # ties merge


def test_select_ovl_matches_R_band_shares_and_is_deterministic():
    rng = np.random.default_rng(0)
    u_ov = rng.beta(5, 2, 2000)                       # synthetic: high overlap
    r_ov = rng.beta(2, 2, 300)                        # real: lower overlap
    eligible = list(range(2000))
    draws = V.select_ovl(1, u_ov, eligible, r_ov, 300)
    assert len(draws) == V.N_DRAWS and all(len(d) == len(set(d)) == 300 for d in draws)
    edges = V.band_edges(r_ov)
    want = np.bincount(V.bands(r_ov, edges), minlength=5)          # quota at n = |R| = 300
    got = np.bincount(V.bands(u_ov[draws[0]], edges), minlength=5)
    avail = np.bincount(V.bands(u_ov, edges), minlength=5)
    for b in range(5):                                # full quota, or the band is exhausted
        assert got[b] >= want[b] or got[b] == avail[b]
    assert got[0] == avail[0] < want[0]               # this data: too few low-overlap queries
    assert abs(u_ov[draws[0]].mean() - r_ov.mean()) < abs(u_ov.mean() - r_ov.mean()) / 3
    assert V.select_ovl(1, u_ov, eligible, r_ov, 300) == draws
    assert V.select_ovl(2, u_ov, eligible, r_ov, 300) != draws


def test_select_nn_prefers_high_scores_within_strata_and_dedupes():
    rng = np.random.default_rng(1)
    u_emb = rng.normal(size=(400, 16)).astype(np.float32)
    u_emb /= np.linalg.norm(u_emb, axis=1, keepdims=True)
    r_emb = u_emb[:60].copy()
    score = rng.random(400)
    picked = V.select_nn(0, score, list(range(5, 400)), u_emb, r_emb, 40)
    assert len(picked) == 40 and not set(picked) & set(range(5))
    assert score[picked].mean() > np.quantile(score, 0.75)
    sims = u_emb[picked] @ u_emb[picked].T
    assert (sims[~np.eye(40, dtype=bool)] <= V.DEDUPE_COS).all()
    assert V.select_nn(0, score, list(range(5, 400)), u_emb, r_emb, 40) == picked


def test_bootstrap_known_case():
    rng = np.random.default_rng(2)
    base = np.linspace(0.2, 0.8, 7)
    a = base + rng.normal(0, 0.05, (300, 7))
    good = base + rng.normal(0, 0.05, (300, 7))
    bad = base[::-1] + rng.normal(0, 0.05, (300, 7))
    si = {"seed": 0, "A": a, "sets": [("M1", 0, bad), ("B0", 0, bad), ("B1", 0, bad),
                                      ("OVL", 0, good), ("OVL", 1, good), ("NN", 0, good)]}
    out = V.bootstrap([si], b_iter=50, seed=5)
    for c in ("OVL-B0", "OVL-B1", "OVL-M1", "NN-B0", "NN-B1", "NN-M1"):
        assert out["comparisons"][c]["delta"] == pytest.approx(2.0)
        assert out["comparisons"][c]["p_one_sided"] == 0.0
    assert out["tau_seed_mean"]["OVL"]["tau_b"] == pytest.approx(1.0)
