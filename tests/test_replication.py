import numpy as np
import pytest

from ha.select import replication as RP
from ha.select import variants as V


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    u_emb = rng.normal(size=(500, 16)).astype(np.float32)
    u_emb /= np.linalg.norm(u_emb, axis=1, keepdims=True)
    r_emb = rng.normal(size=(90, 16)).astype(np.float32)
    r_emb /= np.linalg.norm(r_emb, axis=1, keepdims=True)
    return u_emb, r_emb, rng.beta(5, 2, 500), rng.beta(2, 2, 90)


def test_select_seed_sizes_methods_and_determinism():
    u_emb, r_emb, u_ov, r_ov = _inputs()
    eligible = list(range(10, 500))
    out, n = RP.select_seed(2, eligible, u_emb, r_emb, u_ov, r_ov, n_a=120)
    assert n == 120 and set(out) == {"OVL", "B0", "B1"}
    for m in out:
        assert len(out[m]) == 20
        assert all(len(d) == len(set(d)) == 120 and not set(d) & set(range(10)) for d in out[m])
    assert out["OVL"] == V.select_ovl(2, u_ov, eligible, r_ov, 120)       # the D-068 selector
    assert out["B0"][0] != out["B0"][1]
    again, _ = RP.select_seed(2, eligible, u_emb, r_emb, u_ov, r_ov, n_a=120)
    assert again == out
    _, n_small = RP.select_seed(2, eligible[:50], u_emb, r_emb, u_ov, r_ov, n_a=120)
    assert n_small == 50                                                    # N capped by eligible


def _seed_input(rng, ovl_good=True):
    base = np.linspace(0.2, 0.8, 7)
    a = base + rng.normal(0, 0.05, (200, 7))
    good = base + rng.normal(0, 0.05, (200, 7))
    bad = base[::-1] + rng.normal(0, 0.05, (200, 7))
    ovl = good if ovl_good else bad
    return {"A": a, "sets": [("OVL", 0, ovl), ("OVL", 1, ovl), ("B0", 0, bad), ("B0", 1, bad),
                             ("B1", 0, bad), ("B1", 1, bad)]}


def test_pooled_bootstrap_known_and_null_cases():
    rng = np.random.default_rng(3)
    yes = RP.bootstrap({"d5": [_seed_input(rng) for _ in range(2)],
                        "d6": [_seed_input(rng) for _ in range(2)]}, b_iter=40, seed=1)
    assert yes["R1_supported"]
    assert yes["comparisons"]["OVL-B0"]["pooled"]["delta"] == pytest.approx(2.0)
    assert set(yes["comparisons"]["OVL-B1"]) == {"pooled", "d5", "d6"}
    no = RP.bootstrap({"d5": [_seed_input(rng, ovl_good=False)]}, b_iter=40, seed=1)
    assert not no["R1_supported"]
    assert no["comparisons"]["OVL-B0"]["pooled"]["delta"] == pytest.approx(0.0)
