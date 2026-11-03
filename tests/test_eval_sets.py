import numpy as np
import pytest
from scipy.stats import kendalltau

from ha.eval import sets as E


def test_tau_b_rows_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    a = rng.random(7)
    x = rng.integers(0, 4, size=(200, 7)).astype(float)          # many ties
    x[0] = a                                                      # identical ranking
    got = E.tau_b_rows(x, a)
    for row, t in zip(x, got, strict=True):
        ref = kendalltau(row, a, variant="b").statistic
        assert (np.isnan(ref) and np.isnan(t)) or t == pytest.approx(ref, abs=1e-12)
    assert got[0] == pytest.approx(1.0)
    assert np.isnan(E.tau_b_rows(np.ones((1, 7)), a)[0])          # constant set: undefined


def test_tau_ap_and_describe():
    a = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert E.tau_ap(a, a) == pytest.approx(1.0)
    assert E.tau_ap(a[::-1].copy(), a) == pytest.approx(-1.0)
    swap_top = a.copy()
    swap_top[[0, 1]] = swap_top[[1, 0]]
    swap_bottom = a.copy()
    swap_bottom[[5, 6]] = swap_bottom[[6, 5]]
    assert E.tau_ap(swap_top, a) < E.tau_ap(swap_bottom, a) < 1   # top errors weigh more
    d = E.describe(swap_top, a)
    assert d["winner_agrees"] == 0.0 and d["top3_overlap"] == 1.0 and d["mae"] > 0
    assert d["signed_bias"] == pytest.approx(0.0)


def _seed_input(seed, good_noise, rng):
    """A: 300 queries whose per-system means are strictly ordered C1 < ... < C7. M1 follows A;
    B0/B1/M1-iso are noisy or reversed."""
    base = np.linspace(0.2, 0.8, 7)
    a = np.clip(base + rng.normal(0, 0.05, (300, 7)), 0, 1)
    m1 = np.clip(base + rng.normal(0, good_noise, (300, 7)), 0, 1)
    rev = np.clip(base[::-1] + rng.normal(0, 0.05, (300, 7)), 0, 1)
    sets = [("M1", 0, m1), ("M1-iso", 0, rev)]
    sets += [("B0", d, rev.copy()) for d in range(3)] + [("B1", d, rev.copy()) for d in range(3)]
    return {"seed": seed, "A": a, "sets": sets}


def test_bootstrap_known_case_is_significant_and_reproducible():
    rng = np.random.default_rng(1)
    inputs = {"fiqa": [_seed_input(s, 0.05, rng) for s in range(2)]}
    out = E.bootstrap(inputs, b_iter=60, seed=7)
    for comp in ("M1-B0", "M1-B1", "M1-M1-iso"):
        c = out["comparisons"][comp]["pooled"]
        assert c["delta"] == pytest.approx(2.0)                   # +1 vs -1
        assert c["p_one_sided"] == 0.0 and c["ci_low"] > 0
    assert out["H2_supported"] and out["H3_supported"]
    again = E.bootstrap(inputs, b_iter=60, seed=7)
    assert again["comparisons"] == out["comparisons"]


def test_bootstrap_null_case_is_not_significant():
    rng = np.random.default_rng(2)
    si = _seed_input(0, 0.05, rng)
    same = si["sets"][0][2]
    si["sets"] = [(m, d, same.copy()) for m, d, _ in si["sets"]]   # every method = M1's set
    out = E.bootstrap({"fiqa": [si]}, b_iter=40, seed=3)
    c = out["comparisons"]["M1-B0"]["pooled"]
    assert c["delta"] == pytest.approx(0.0) and c["p_one_sided"] > 0.05
    assert not out["H2_supported"]


def test_point_estimates_average_draws():
    rng = np.random.default_rng(3)
    pts, vectors = E.point_estimates([_seed_input(0, 0.05, rng)])
    by = pts.set_index("method")
    assert by.loc["M1", "tau_b"] == pytest.approx(1.0) and by.loc["B0", "n_draws"] == 3
    assert by.loc["B0", "tau_b"] == pytest.approx(-1.0)
    assert {v["set"] for v in vectors} == {"A", "M1", "M1-iso", "B0", "B1"}
