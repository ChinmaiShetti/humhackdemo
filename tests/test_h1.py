import numpy as np
import pytest

from ha.eval import h1


def test_overlap_and_length():
    assert h1.overlap("How do I file my tax return?", "File your tax return online.") == \
        pytest.approx(3 / 3)                       # file, tax, return; stopwords dropped
    assert h1.overlap("refund status", "nothing here") == 0.0
    assert h1.overlap("the of and", "anything") == 0.0      # only stopwords
    assert h1.n_words("  two   words ") == 2


def test_holm_step_down():
    p = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}
    assert h1.holm(p) == {"a": True, "b": False, "c": False, "d": False}   # 0.02 > 0.05/3
    assert h1.holm({"a": 0.01, "b": 0.02}) == {"a": True, "b": True}


def test_decide_uses_max_p_over_seeds_and_p1_for_missing_datasets():
    seed = {m: {"p": 0.001} for m in h1.MEASURES}
    worse = {**seed, "length": {"p": 0.2}}
    out = h1.decide({"fiqa": [seed, worse]})
    assert out["m"] == 9
    assert out["p"]["fiqa:length"] == 0.2 and out["p"]["fiqa:overlap"] == 0.001
    assert out["p"]["hiaq:topic"] == 1.0 and out["p"]["cqa_android:length"] == 1.0
    assert out["rejected"]["fiqa:overlap"] and not out["rejected"]["fiqa:length"]


def test_js_permutation_and_w1():
    rng = np.random.default_rng(0)
    same = rng.integers(0, 5, 400)
    assert h1.js_test(same, rng.integers(0, 5, 400), 5, rng, n_perm=200)["p"] > 0.01
    skew = np.concatenate([np.zeros(300, int), rng.integers(0, 5, 100)])
    t = h1.js_test(skew, rng.integers(0, 5, 400), 5, rng, n_perm=200)
    assert t["p"] < 0.01 and t["js"] > 0.1
    w = h1.w1_with_ci(np.arange(100.0), np.arange(100.0) + 3, rng, b_iter=200)
    assert w["w1"] == pytest.approx(3.0) and w["ci_low"] <= 3.0 <= w["ci_high"]
