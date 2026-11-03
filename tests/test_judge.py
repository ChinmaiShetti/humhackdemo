import json
import math
from types import SimpleNamespace

import pytest

from ha.judge import aggregate as A
from ha.judge import run as J
from ha.judge.prompt import judge_messages, permute

NEIGH = ["real q one", "real q two", "real q three", "real q four", "real q five"]


def _out(F=5, H=4, S=4, P=4, C=None):
    return json.dumps({"F": F, "H": H, "S": S, "P": P, "C": C, "keep": True, "reason": "ok"})


class Client:
    """Scripted responses per call; records the messages it received."""

    def __init__(self, responses):
        self.responses, self.seen = list(responses), []

    def chat(self, model, messages, format, think, options):
        self.seen.append(messages)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(message=SimpleNamespace(content=r), prompt_eval_count=700,
                               eval_count=40)


def test_permutation_deterministic_and_seed_independent():
    a = permute(NEIGH, "q1", 0)
    assert a == permute(NEIGH, "q1", 0) and sorted(a) == sorted(NEIGH)
    assert a != permute(NEIGH, "q1", 1) or a != permute(NEIGH, "q2", 0)


def test_iso_prompt_has_no_neighbour_block_and_m1_does():
    m1 = judge_messages("q1", "query", "passage", NEIGH, "EN", "finance", 0)[1]["content"]
    iso = judge_messages("q1", "query", "passage", None, "EN", "finance", 0)[1]["content"]
    assert "REAL USER QUERIES" in m1 and all(n in m1 for n in NEIGH)
    assert "REAL USER QUERIES" not in iso and not any(n in iso for n in NEIGH)


def test_retry_once_with_error_then_success_and_then_failure():
    c = Client(["not json", _out()])
    r = J.judge_call([{"role": "user", "content": "x"}], None, c)
    assert r["ok"] and r["attempts"] == 2
    assert "invalid" in c.seen[1][-1]["content"] and c.seen[1][-2]["role"] == "assistant"
    c2 = Client(["bad", "still bad"])
    r2 = J.judge_call([{"role": "user", "content": "x"}], None, c2)
    assert not r2["ok"] and r2["attempts"] == 2 and r2["error"].startswith("validation")


def test_transport_error_is_failure_without_retry():
    c = Client([RuntimeError("exceed_context_size_error")])
    r = J.judge_call([{"role": "user", "content": "x"}], None, c)
    assert not r["ok"] and r["attempts"] == 1 and "exceed_context" in r["error"]


def test_m1_third_permutation_and_median():
    c = Client([_out(F=5, H=2), _out(F=5, H=4), _out(F=4, H=5)])   # H differs by 2 -> 3rd run
    r = J.judge_m1("q1", "query", "passage", NEIGH, "EN", "finance", None, c)
    assert r["ok"] and r["third_run"] and len(r["perms"]) == 3
    assert r["dims"]["H"] == 4 and r["dims"]["F"] == 5 and r["dims"]["C"] is None
    c2 = Client([_out(H=3), _out(H=4)])
    r2 = J.judge_m1("q1", "query", "passage", NEIGH, "EN", "finance", None, c2)
    assert not r2["third_run"] and r2["dims"]["H"] == 3.5


def test_m1_fails_if_any_permutation_fails():
    c = Client([_out(), "bad", "bad"])
    r = J.judge_m1("q1", "query", "passage", NEIGH, "EN", "finance", None, c)
    assert not r["ok"] and r["dims"] is None


def test_cache_hit(tmp_path):
    from ha.cache import KVCache
    cache = KVCache(tmp_path / "j.sqlite")
    c = Client([_out()])
    J.judge_call([{"role": "user", "content": "x"}], cache, c)
    r = J.judge_call([{"role": "user", "content": "x"}], cache, Client([]))
    assert r["ok"] and r["cached"] == [True]


def test_aggregation_gate_score_and_nnls():
    dims = {"F": 4.0, "H": 3.0, "S": 5.0, "P": 4.0, "C": None}
    assert A.passes_gate(dims) and A.c_star(dims) == 4.0
    assert A.score(dims, A.EQUAL_WEIGHTS) == 16.0
    assert A.score({**dims, "F": 3.5}, A.EQUAL_WEIGHTS) == -math.inf
    rows = [{"H": h, "S": s, "P": p, "C": None} for h, s, p in
            [(1, 2, 3), (2, 2, 4), (3, 3, 3), (4, 5, 4), (5, 4, 5), (2, 1, 1), (3, 4, 2),
             (4, 4, 4), (5, 5, 3), (1, 1, 2)]]
    y = [2 * r["H"] + 0.5 * r["P"] for r in rows]
    fit = A.fit_nnls(rows, y)
    assert all(w >= 0 for w in fit["weights"].values()) and fit["n"] == 10
    assert fit["weights"]["H"] > fit["weights"]["S"]


@pytest.mark.parametrize(("a", "b", "third"), [({"F": 5, "C": None}, {"F": 3, "C": None}, True),
                                               ({"F": 5, "C": 2}, {"F": 5, "C": None}, False)])
def test_needs_third(a, b, third):
    base = {"H": 3, "S": 3, "P": 3}
    assert A.needs_third({**base, **a}, {**base, **b}) is third
