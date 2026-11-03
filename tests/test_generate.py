import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ha.gen import generate as G


class FakeClient:
    """Deterministic stand-in for the Ollama client: output depends on (prompt, seed)."""

    def __init__(self, fail_on=(), empty_on=()):
        self.calls = 0
        self.fail_on, self.empty_on = set(fail_on), set(empty_on)

    def chat(self, model, messages, think, options):
        self.calls += 1
        seed = options["seed"]
        if seed in self.fail_on:
            raise ConnectionError("ollama down")
        text = "" if seed in self.empty_on else f'1. "how do i handle case {seed} for my account?"'
        return SimpleNamespace(message=SimpleNamespace(content=text), prompt_eval_count=50,
                               eval_count=10)


def test_clean_strips_numbering_quotes_and_extra_lines():
    assert G.clean('1. "What is my refund status?"\nExplanation...') == "What is my refund status?"
    assert G.clean("  - ‘pf transfer kaise kare’  ") == "pf transfer kaise kare"
    assert G.clean("") == ""


def test_filters():
    passage = "Refunds are processed within four weeks of filing. Later text here."
    assert G.filter_reason("refund", passage) == "too_short"
    assert G.filter_reason("refunds are processed within four weeks", passage) == \
        "copies_first_sentence"
    assert G.filter_reason("why is my tax refund late", passage) is None


def test_dedupe_greedy_keep_first():
    v = np.array([[1, 0], [0.999, 0.0447], [0, 1]], dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert G.dedupe(v) == [0, 2]


def test_personas_and_exemplars_deterministic_and_scoped():
    assert len(G.personas_for("fiqa")) == 4 and len(G.personas_for("hiaq")) == 5
    assert G.pick_persona("fiqa", 7) == G.pick_persona("fiqa", 7)
    r = [(f"q{i}", f"text {i}") for i in range(20)]
    a = G.pick_exemplars(r, 0, 5)
    assert a == G.pick_exemplars(r, 0, 5) and len(a) == G.FEWSHOT_K
    assert set(a) <= set(r)
    assert a != G.pick_exemplars(r, 1, 5)


@pytest.fixture
def tiny(monkeypatch):
    corpus = pd.DataFrame({"pid": [f"d{i}" for i in range(6)],
                           "text": [f"Title {i}\nPassage number {i} about accounts. More." for i in
                                    range(6)]})
    monkeypatch.setattr(G, "load_source_corpus", lambda ds: corpus)

    def fake_sample(ds):
        out = corpus.copy()
        out.insert(0, "i", range(len(out)))
        return out

    monkeypatch.setattr(G, "sample_passages", fake_sample)
    return corpus


def _log(spec):
    with open(G.pool_dir(spec) / "generations.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_generate_resume_cache_and_failures(tiny):
    spec = G.PoolSpec("fiqa", "P-std")
    fc = FakeClient(fail_on={1002}, empty_on={1003})
    G.generate_pool(spec, client=fc)
    recs = {r["i"]: r for r in _log(spec)}
    assert len(recs) == 6 and fc.calls == 6
    assert recs[2]["status"] == "failed" and "ConnectionError" in recs[2]["reason"]
    assert recs[3]["status"] == "failed" and recs[3]["reason"] == "empty_output"
    assert recs[0]["status"] == "ok" and recs[0]["query"].startswith("how do i handle case 1000")
    assert recs[0]["model_digest"] and recs[0]["seed"] == 1000
    assert G.generate_pool(spec, client=fc)["new"] == 0          # resume: nothing redone
    (G.pool_dir(spec) / "generations.jsonl").unlink()
    fc2 = FakeClient()
    G.generate_pool(spec, client=fc2)
    # The empty output is a genuine (deterministic) model response and is cached like any other
    # response; only the transport exception left no cache entry, so only it is re-called.
    assert sum(r["cached"] for r in _log(spec) if "cached" in r) == 5
    assert fc2.calls == 1


def test_fewshot_reads_only_R(tiny, monkeypatch):
    parts = []

    def fake_load_split(ds, seed, part):
        parts.append(part)
        return pd.DataFrame({"qid": [f"r{i}" for i in range(10)],
                             "text": [f"real query {i}" for i in range(10)]})

    import ha.data.splits as S
    monkeypatch.setattr(S, "load_split", fake_load_split)
    spec = G.PoolSpec("fiqa", "P-fewshot", split_seed=2)
    G.generate_pool(spec, client=FakeClient())
    assert parts == ["R"]
    assert all(set(r["exemplar_qids"]) <= {f"r{i}" for i in range(10)} for r in _log(spec))


def test_finalize_counts(tiny):
    spec = G.PoolSpec("fiqa", "P-std")
    G.generate_pool(spec, client=FakeClient(empty_on={1005}))
    stats = G.finalize_pool(spec, target=6,
                            embed_fn=lambda t: np.eye(len(t), dtype=np.float32))
    assert stats["attempted"] == 6 and stats["failed"] == 1 and stats["final_unique"] == 5
    q = pd.read_parquet(G.pool_dir(spec) / "queries.parquet")
    assert q["qid"].is_unique and set(q["source_pid"]) <= set(tiny["pid"])
