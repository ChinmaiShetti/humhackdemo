import math

import numpy as np
import pytest

from ha.eval.agreement import pair_disagreements, set_scores, tau_b


def test_tau_b_identical_reversed_and_ties():
    x = {"a": 0.1, "b": 0.2, "c": 0.3}
    assert tau_b(x, x) == pytest.approx(1.0)
    assert tau_b(x, {"a": 0.3, "b": 0.2, "c": 0.1}) == pytest.approx(-1.0)
    # one tie in y: variant b corrects for ties (scipy reference value)
    assert tau_b([1, 2, 3, 4], [1, 2, 2, 4]) == pytest.approx(0.9128709, abs=1e-6)
    assert math.isnan(tau_b([1, 2, 3], [5, 5, 5]))


def test_tau_b_uses_same_keys_regardless_of_dict_order():
    a = {"C1": 0.2, "C2": 0.4, "C3": 0.3}
    b = {"C3": 0.35, "C1": 0.1, "C2": 0.5}
    assert tau_b(a, b) == pytest.approx(1.0)


def test_set_scores_and_pair_flips():
    pq = {"s1": np.array([0.0, 1.0, 1.0, 0.0]), "s2": np.array([1.0, 0.0, 0.0, 1.0])}
    a, b = set_scores(pq, np.array([0, 3])), set_scores(pq, np.array([1, 2]))
    assert a == {"s1": 0.0, "s2": 1.0} and b == {"s1": 1.0, "s2": 0.0}
    assert pair_disagreements(a, b) == {("s1", "s2"): True}
