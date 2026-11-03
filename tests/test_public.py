import pandas as pd
import pytest

from ha.data.public import DatasetCheckError, _normalise_qrels, validate_processed


def _frames():
    corpus = pd.DataFrame({"pid": ["d1", "d2"], "title": ["", "t"], "text": ["a", "b"]})
    queries = pd.DataFrame({"qid": ["q1", "q2"], "text": ["how to x", "why y"]})
    qrels = pd.DataFrame({"qid": ["q1", "q2"], "pid": ["d1", "d2"], "grade": [1, 1]})
    return corpus, queries, qrels


def test_valid_frames_pass():
    validate_processed(*_frames())


@pytest.mark.parametrize("mutate", [
    lambda c, q, r: r.assign(pid=["d1", "missing"]),
    lambda c, q, r: r.assign(qid=["q1", "q9"]),
    lambda c, q, r: r.iloc[:1],                        # q2 has no relevant passage
    lambda c, q, r: r.rename(columns={"grade": "score"}),
])
def test_invalid_frames_rejected(mutate):
    corpus, queries, qrels = _frames()
    with pytest.raises(DatasetCheckError):
        validate_processed(corpus, queries, mutate(corpus, queries, qrels))


def test_qrels_drop_zero_grades_and_duplicates():
    raw = pd.DataFrame({"query-id": ["1", "1", "2"], "corpus-id": ["a", "a", "b"],
                        "score": ["1", "1", "0"]})
    out = _normalise_qrels(raw)
    assert out.to_dict("records") == [{"qid": "1", "pid": "a", "grade": 1}]
