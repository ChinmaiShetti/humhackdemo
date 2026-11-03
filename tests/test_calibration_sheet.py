import numpy as np
import pandas as pd
import pytest

from ha.config import domain_description
from ha.human import calibration as HC
from ha.judge.calibrate import CalibrationBlocked
from ha.neighbours.retrieve import neighbours_for
from ha.paths import get_paths

REAL_DATA = get_paths().data          # resolved before the isolation fixture runs


def test_passage_rule_is_deterministic_max_grade_and_skips_missing():
    qrels = pd.DataFrame({"qid": ["q1"] * 4 + ["q2"],
                          "pid": ["p9", "p3", "p5", "p7", "p1"], "grade": [1, 2, 2, 2, 1]})
    texts = {"p9": "a", "p3": "b", "p5": "c", "p7": "  ", "p1": "d"}
    pick = HC.select_passage("fiqa", "q1", qrels, texts)
    assert pick in {"p3", "p5"}                                  # max grade, non-empty only
    shuffled = qrels.sample(frac=1, random_state=3)
    assert HC.select_passage("fiqa", "q1", shuffled, texts) == pick   # row order irrelevant
    assert HC.select_passage("fiqa", "q2", qrels, texts) == "p1"
    assert HC.select_passage("fiqa", "q3", qrels, texts) is None      # pending, no substitute
    assert HC.select_passage("fiqa", "q2", qrels, {"p1": ""}) is None


def test_attach_passages_leaves_authentic_pending_without_qrels():
    items = pd.DataFrame({"qid": ["a1", "s1"], "text": ["x", "y"], "dataset": ["hiaq", "hiaq"],
                          "kind": ["authentic", "synthetic"], "pool": ["authentic", "P-std"]})
    out, pending = HC.attach_passages(items, {"s1": "c7"}, {"hiaq": None},
                                      {"hiaq": {"c7": "chunk text"}})
    assert pending == [("hiaq", "a1")]
    assert out.set_index("qid").loc["s1", "passage"] == "chunk text"
    assert pd.isna(out.set_index("qid").loc["a1", "passage"])
    with pytest.raises(CalibrationBlocked):
        HC.make_pack(out.assign(item_id=["C001", "C002"], neighbour_qids=[[], []],
                                neighbour_texts=[[], []]), "calibration")


def _toy(n_r=12, dim=8):
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"qid": [f"r{i}" for i in range(n_r)], "text": [f"rt{i}" for i in range(n_r)],
                      "lang_class": ["EN"] * n_r})
    r_emb = rng.normal(size=(n_r, dim))
    r_emb /= np.linalg.norm(r_emb, axis=1, keepdims=True)
    return r, r_emb


def test_human_neighbours_exclude_all_cal_auth_but_judge_side_is_unchanged():
    r, r_emb = _toy()
    cal = {("fiqa", "r0"), ("fiqa", "r1"), ("fiqa", "r2"), ("hiaq", "r3")}
    # authentic item r0 (in R, as prereg §4.1 forces) and a synthetic item close to r1
    items = pd.DataFrame({"qid": ["r0", "s1"], "text": ["rt0", "st1"],
                          "kind": ["authentic", "synthetic"]})
    q_emb = np.vstack([r_emb[0], r_emb[1] * 0.99 + 0.01])

    def rerank(pairs):
        return [-int(qid[1:]) for _, qid, _ in pairs]          # favours low ids (the CAL-auth)

    before = r.copy()
    rd = HC.human_display_R(r, "fiqa", cal)
    assert r.equals(before)                                        # R itself is not modified
    assert set(rd["qid"]) == {f"r{i}" for i in range(3, 12)}       # only fiqa CAL-auth dropped
    keep = rd.index.map(lambda i: int(rd.qid[i][1:])).to_numpy()
    nb = HC.human_neighbours(items, rd, q_emb, r_emb[keep], rerank, "fiqa", cal)
    shown = {q for lst in nb["neighbour_qids"] for q in lst}
    assert not shown & {"r0", "r1", "r2"}
    assert all(len(t) == 5 for t in nb["neighbour_texts"])
    # the judge-side (frozen) procedure on full R still sees CAL-auth items
    judge = neighbours_for(items[["qid", "text"]], r, q_emb, r_emb, rerank, authentic=False)
    assert {"r1", "r2"} & {q for lst in judge["neighbour_qids"] for q in lst}


def test_human_neighbours_guard_catches_a_leak():
    r, r_emb = _toy()
    items = pd.DataFrame({"qid": ["s1"], "text": ["x"], "kind": ["synthetic"]})
    with pytest.raises(AssertionError):                          # r not filtered -> leak
        HC.human_neighbours(items, r, r_emb[:1], r_emb, lambda p: [0.0] * len(p), "fiqa",
                            {("fiqa", q) for q in r["qid"]})


def _items(n=8):
    return pd.DataFrame({
        "qid": [f"q{i}" for i in range(n)], "text": [f"query {i}" for i in range(n)],
        "dataset": ["fiqa"] * (n // 2) + ["hiaq"] * (n - n // 2),
        "kind": ["authentic", "synthetic"] * (n // 2), "pool": ["authentic", "P-std"] * (n // 2),
        "split": ["cal-dev", "cal-test"] * (n // 2), "passage_pid": [f"p{i}" for i in range(n)],
        "passage": [f"passage {i}" for i in range(n)], "passage_rule": "r",
        "neighbour_qids": [[f"n{i}{j}" for j in range(5)] for i in range(n)],
        "neighbour_texts": [[f"nt{i}{j}" for j in range(5)] for i in range(n)]})


def test_pack_is_blind_and_key_is_separate():
    items = HC.assign_item_ids(_items(), 17, "C")
    assert items.equals(HC.assign_item_ids(_items().sample(frac=1, random_state=1), 17, "C"))
    sheet, key = HC.make_pack(items, "calibration")
    assert list(sheet.columns) == HC.SHEET_COLS
    for leaky in ("qid", "kind", "pool", "split", "passage_pid", "source_pid"):
        assert leaky not in sheet.columns
    assert (sheet[list(HC.DIMS)] == "").all().all()              # nothing pre-filled
    assert set(sheet["domain"]) == {domain_description("fiqa"), domain_description("hiaq")}
    assert list(key.columns) == HC.KEY_COLS and key["item_id"].equals(sheet["item_id"])


def test_domain_descriptions():
    assert domain_description("hiaq").startswith("HiAQ covers questions about Indian digital")
    assert domain_description("fiqa") == "personal finance questions from StackExchange and " \
                                         "Reddit users"


def _filled(blank, aid, vals, c=None):
    s = HC.annotator_sheet(blank, aid)
    for d in ("F", "H", "S", "P", "overall"):
        s[d] = [str(v) for v in vals]
    s["C"] = ["" if x is None else str(x) for x in (c or [None] * len(s))]
    return s


def test_write_refuses_to_overwrite_human_entries(tmp_path):
    blank, _ = HC.make_pack(HC.assign_item_ids(_items(4), 17, "C"), "calibration")
    p = tmp_path / "raw_r1_A1.csv"
    HC.write_csv(HC.annotator_sheet(blank, "A1"), p)
    HC.write_csv(HC.annotator_sheet(blank, "A1"), p)              # still blank: allowed
    HC.write_csv(_filled(blank, "A1", [4, 4, 4, 4]), p)
    with pytest.raises(FileExistsError):
        HC.write_csv(HC.annotator_sheet(blank, "A1"), p)
    assert HC.read_csv(p)["F"].tolist() == ["4"] * 4              # round-trips as text


def test_validate_sheet():
    blank, _ = HC.make_pack(HC.assign_item_ids(_items(4), 17, "C"), "calibration")
    ok = _filled(blank, "A1", [5, 4, 3, 1])
    assert HC.validate_sheet(ok, blank) == []                    # C blank is allowed
    bad = ok.copy()
    bad.loc[0, "H"], bad.loc[1, "S"], bad.loc[2, "query"] = "6", "", "edited"
    probs = " | ".join(HC.validate_sheet(bad, blank))
    assert "H: not an integer" in probs and "S: blank" in probs and "query edited" in probs
    assert HC.validate_sheet(ok.drop(columns="notes"), blank)


def test_adjudication_and_final_labels_are_separate_steps(tmp_path):
    blank, _ = HC.make_pack(HC.assign_item_ids(_items(4), 17, "C"), "calibration")
    a = _filled(blank, "A1", [5, 4, 3, 1], c=[None, 4, None, 2])
    b = _filled(blank, "A2", [4, 2, 3, 1], c=[None, 5, 3, 2])
    a_before = a.copy()
    adj = HC.build_adjudication(a, b)
    assert a.equals(a_before)                                   # raw sheets never modified
    assert adj["F_discuss"].tolist() == [0, 1, 0, 0]            # |4-2| >= 2
    assert adj["C_discuss"].tolist() == [0, 0, 1, 0]            # C given by one annotator only
    assert adj["F_mean"].tolist() == [4.5, 3.0, 3.0, 1.0]
    with pytest.raises(ValueError, match="unsettled"):
        HC.finalize(adj)
    for d in ("F", "H", "S", "P", "overall"):                   # _filled repeats one row
        assert adj.loc[1, f"{d}_discuss"] == 1
        adj.loc[1, f"{d}_settled"] = "3"
    adj.loc[2, "C_settled"] = "NA"
    fin = HC.finalize(adj)
    assert fin["F"].tolist() == [4.5, 3.0, 3.0, 1.0]
    assert fin["F_source"].tolist() == ["mean", "settled", "mean", "mean"]
    assert np.isnan(fin.loc[0, "C"]) and np.isnan(fin.loc[2, "C"]) and fin.loc[1, "C"] == 4.5
    HC.write_csv(adj, tmp_path / "adj.csv")                     # CSV round trip (Excel text)
    back = HC.finalize(HC.read_csv(tmp_path / "adj.csv"))
    pd.testing.assert_frame_equal(back, fin, check_dtype=False)
    adj.loc[0, "H_settled"] = "4"                               # settling an unflagged item
    with pytest.raises(ValueError, match="unflagged"):
        HC.finalize(adj)
    with pytest.raises(ValueError):
        HC.build_adjudication(a, a)                              # same annotator twice


@pytest.mark.skipif(not (REAL_DATA / "synthetic" / "fiqa" / "P-std" / "queries.parquet").exists(),
                    reason="P3 pools not present")
def test_practice_items_are_outside_the_calibration_draw(monkeypatch):
    from ha.judge.calibrate import draw_synthetic
    from ha.neighbours.retrieve import load_U

    monkeypatch.setenv("HA_DATA", str(REAL_DATA))                # read-only use of the pools
    get_paths.cache_clear()

    p = HC.practice_items()
    assert len(p) == 20 and p["pool"].value_counts().to_dict() == {"P-std": 10, "P-persona": 10}
    assert not set(p["qid"]) & set(draw_synthetic(load_U("fiqa"), 70, 17)["qid"])
    assert p.equals(HC.practice_items())


def test_main_pack_blocked_without_hiaq_half():
    if HC.status()["main_pack_ready"]:
        pytest.skip("HiAQ half present")
    with pytest.raises(CalibrationBlocked):
        HC.prepare_main()
