import math

import pandas as pd
import pytest

from ha.eval.metrics import per_query
from ha.seal import SealedDataError
from ha.systems import retrieval as R
from ha.systems import run_all


def test_rrf_known_example():
    a = pd.DataFrame({"qid": ["q"] * 3, "pid": ["x", "y", "z"], "rank": [1, 2, 3], "score": 0.0})
    b = pd.DataFrame({"qid": ["q"] * 3, "pid": ["y", "w", "x"], "rank": [1, 2, 3], "score": 0.0})
    f = R.rrf([a, b], k=60, depth=10)
    assert f["pid"].tolist()[:2] == ["y", "x"]          # both appear in both runs
    assert f.loc[f.pid == "y", "score"].item() == pytest.approx(1 / 62 + 1 / 61)
    assert f["rank"].tolist() == [1, 2, 3, 4]


def test_rerank_head_rescored_tail_kept_below():
    first = pd.DataFrame({"qid": ["q"] * 5, "pid": list("abcde"), "rank": [1, 2, 3, 4, 5],
                          "score": [5, 4, 3, 2, 1.0]})
    calls = []

    def scorer(pairs):
        calls.append(len(pairs))
        return [{"a": 0.1, "b": 0.9, "c": 0.5}[p] for _, p, _ in pairs]

    out = R.rerank_run(first, {"q": "query"}, {p: p for p in "abcde"}, scorer, depth=3)
    assert out["pid"].tolist() == ["b", "c", "a", "d", "e"]
    assert calls == [3]
    assert out["rank"].tolist() == [1, 2, 3, 4, 5]


def test_per_query_metrics_match_manual_ndcg():
    run = pd.DataFrame({"qid": ["q1", "q1", "q1", "q2"], "pid": ["a", "b", "c", "z"],
                        "rank": [1, 2, 3, 1], "score": [3.0, 2.0, 1.0, 1.0]})
    qrels = pd.DataFrame({"qid": ["q1", "q1", "q2", "q3"], "pid": ["b", "c", "y", "k"],
                          "grade": [1, 2, 1, 1]})
    pq = per_query(run, qrels, ["ndcg@10", "mrr@10", "recall@100"]).set_index("qid")
    dcg = 1 / math.log2(3) + 2 / math.log2(4)                 # linear gain (ranx default)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert pq.loc["q1", "ndcg@10"] == pytest.approx(dcg / idcg)
    assert pq.loc["q1", "mrr@10"] == pytest.approx(0.5)
    assert pq.loc["q2", "ndcg@10"] == 0 and pq.loc["q3", "recall@100"] == 0  # q3 not in run
    assert set(pq.index) == {"q1", "q2", "q3"}


def test_bm25_toy_corpus(tmp_path):
    d = tmp_path / "ha_data" / "processed" / "toy"
    d.mkdir(parents=True)
    pd.DataFrame({"pid": ["d1", "d2", "d3"], "title": ["", "Refunds", ""],
                  "text": ["phone battery drains fast", "refund status and reissue",
                           "update mobile number"]}).to_parquet(d / "corpus.parquet")
    run = R.BM25Index("toy", 0.9, 0.4).search(["q"], ["where is my refund"], k=2)
    assert run["pid"].iloc[0] == "d2"
    again = R.BM25Index("toy", 0.9, 0.4)                    # loads the cached index
    assert again.search(["q"], ["mobile number"], k=1)["pid"].item() == "d3"


def test_heldout_authentic_queries_are_sealed(monkeypatch):
    monkeypatch.setattr(run_all, "dataset_config", lambda ds: {"role": "heldout", "id": "D2"})
    with pytest.raises(SealedDataError):
        run_all.query_set("cqa_android", "auth")
    assert len(run_all.query_set("cqa_android", "smoke")) == len(run_all.SMOKE_QUERIES)
