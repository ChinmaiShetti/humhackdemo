import numpy as np
import pandas as pd
import pytest

from ha.data import splits
from ha.seal import SealedDataError


def _strata(n=200):
    qids = [f"q{i:03d}" for i in range(n)]
    labels = [("EN" if i % 4 else "HINGLISH") + f"|{i % 5}" for i in range(n)]
    return pd.Series(labels, index=qids)


def test_splits_deterministic_disjoint_and_forced():
    strata = _strata()
    forced = {"q000", "q004", "q008", "q013"}
    a = splits.assign_splits(strata, forced)
    b = splits.assign_splits(strata, forced)
    pd.testing.assert_frame_equal(a, b)
    for seed in splits.SPLIT_SEEDS:
        col = a[f"split_s{seed}"]
        assert set(col.unique()) <= {"R", "A"}                  # every query exactly one part
        assert all(col[q] == "R" for q in forced)                # CAL-auth always in R
        for _, members in strata.groupby(strata):                # stratified within +-1
            n_r = (col[members.index] == "R").sum()
            assert abs(n_r - round(0.4 * len(members))) <= 1
    assert not a["split_s0"].equals(a["split_s1"])              # seeds differ


def test_cal_auth_draw_is_stratified_and_fixed():
    qids = [f"q{i}" for i in range(100)]
    lang = ["EN"] * 70 + ["HINGLISH"] * 20 + ["ROM_HI"] * 10
    c1 = splits.draw_cal_auth(qids, lang, 30)
    assert c1 == splits.draw_cal_auth(qids, lang, 30)
    assert len(c1) == 30
    by = pd.Series(lang, index=qids)[list(c1)].value_counts()
    assert by["EN"] == 21 and by["HINGLISH"] == 6 and by["ROM_HI"] == 3


def test_cluster_count_rule():
    assert splits.n_topic_clusters(648) == 10
    assert splits.n_topic_clusters(90) == 3
    assert splits.n_topic_clusters(10) == 1


def test_heldout_A_is_sealed(tmp_path, monkeypatch):
    df = pd.DataFrame({"qid": ["a", "b"], "text": ["x", "y"], "split_s0": ["R", "A"]})
    monkeypatch.setattr(splits, "processed_dir", lambda ds: tmp_path)
    df.to_parquet(tmp_path / "splits.parquet", index=False)
    monkeypatch.setattr(splits, "dataset_config", lambda ds: {"role": "heldout"})
    assert splits.load_split("d2", 0, "R")["qid"].tolist() == ["a"]
    with pytest.raises(SealedDataError):
        splits.load_split("d2", 0, "A")
    monkeypatch.setattr(splits, "dataset_config", lambda ds: {"role": "dev"})
    assert splits.load_split("d1", 0, "A")["qid"].tolist() == ["b"]


def test_build_splits_end_to_end_with_fake_embeddings(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    q = pd.DataFrame({"qid": [f"q{i}" for i in range(120)],
                      "text": ["how to update my account"] * 120})
    q.to_parquet(tmp_path / "queries_auth.parquet", index=False)
    monkeypatch.setattr(splits, "processed_dir", lambda ds: tmp_path)
    out = splits.build_splits("x", n_cal_auth=10,
                              embed_fn=lambda texts: rng.normal(size=(len(texts), 8)))
    assert out["cal_auth"].sum() == 10
    assert (out.loc[out["cal_auth"], [f"split_s{s}" for s in range(5)]] == "R").all().all()
    frac = (out["split_s0"] == "R").mean()
    assert 0.35 <= frac <= 0.45
