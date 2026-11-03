import json

import numpy as np
import pandas as pd

from ha.neighbours.retrieve import TOP_K, neighbours_for


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _setup(n_r=40, seed=0):
    rng = np.random.default_rng(seed)
    r = pd.DataFrame({"qid": [f"r{i}" for i in range(n_r)],
                      "text": [f"real {i}" for i in range(n_r)],
                      "lang_class": ["EN"] * (n_r - 5) + ["HINGLISH"] * 5})
    r_emb = _unit(rng.normal(size=(n_r, 8)))
    return r, r_emb, rng


def rerank_by_index(pairs):  # prefers higher R index -> reorders the dense top-30
    return [float(p[1][1:]) for p in pairs]


def test_neighbours_only_from_R_and_top5_reranked():
    r, r_emb, rng = _setup()
    q = pd.DataFrame({"qid": ["s1", "s2"], "text": ["a", "b"]})
    q_emb = _unit(rng.normal(size=(2, 8)))
    nb = neighbours_for(q, r, q_emb, r_emb, rerank_by_index)
    for row in nb.itertuples():
        assert set(row.neighbour_qids) <= set(r["qid"]) and len(row.neighbour_qids) == TOP_K
        assert len(row.candidate_qids) == 30
        assert list(row.neighbour_rerank) == sorted(row.neighbour_rerank, reverse=True)
        assert np.isclose(row.nn_mean5_cos, np.mean(row.neighbour_cos))
        assert sum(json.loads(row.nn_lang_mix).values()) == TOP_K
    cos = q_emb @ r_emb.T
    assert np.isclose(nb["nn_max_cos"].iloc[0], cos[0].max())


def test_authentic_never_self_or_near_duplicate():
    r, r_emb, _ = _setup()
    q = r.iloc[[3]][["qid", "text"]].reset_index(drop=True)          # an R query as the item
    q_emb = r_emb[[3]]
    near = r_emb[3] * 0.999 + r_emb[4] * 0.001                       # a near-copy of it in R
    r_emb2 = r_emb.copy()
    r_emb2[7] = near / np.linalg.norm(near)
    nb = neighbours_for(q, r, q_emb, r_emb2, rerank_by_index, authentic=True)
    assert "r3" not in nb["candidate_qids"].iloc[0] and "r7" not in nb["candidate_qids"].iloc[0]
    assert nb["nn_max_cos"].iloc[0] <= 0.97 and not nb["near_duplicate"].iloc[0]


def test_near_duplicate_flag_for_synthetic():
    r, r_emb, _ = _setup()
    q = pd.DataFrame({"qid": ["s1"], "text": ["copy"]})
    nb = neighbours_for(q, r, r_emb[[10]], r_emb, rerank_by_index)
    assert nb["near_duplicate"].iloc[0] and nb["nn_max_cos"].iloc[0] > 0.97


def test_deterministic():
    r, r_emb, rng = _setup()
    q = pd.DataFrame({"qid": ["s1"], "text": ["a"]})
    q_emb = _unit(rng.normal(size=(1, 8)))
    a = neighbours_for(q, r, q_emb, r_emb, rerank_by_index)
    b = neighbours_for(q, r, q_emb, r_emb, rerank_by_index)
    pd.testing.assert_frame_equal(a, b)
