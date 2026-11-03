import json

import pandas as pd
import pytest

from ha.judge import bulk as B
from ha.judge.aggregate import median_dims

QIDS = ["u1", "u2", "u3", "u4"]
R_QIDS = [f"r{i}" for i in range(1, 8)]


def _res(F=5, H=4, third=False):
    out = {"F": F, "H": H, "S": 4, "P": 4, "C": None, "keep": True, "reason": "ok"}
    perms = [{"perm": p, "ok": True, "attempts": 1, "cached": [False], "output": out,
              "prompt_tokens": 600} for p in range(3 if third else 2)]
    return {"ok": True, "perms": perms, "third_run": third, "dims": median_dims([out]),
            "error": None}


def _fail(err):
    perms = [{"perm": 0, "ok": False, "attempts": 2 if err.startswith("validation") else 1,
              "cached": [], "output": None, "error": err}]
    return {"ok": False, "perms": perms, "third_run": False, "dims": None, "error": err}


@pytest.fixture
def env(monkeypatch):
    from ha import llm
    from ha.data import langid, splits
    from ha.gen import generate
    from ha.judge import calibrate
    from ha.judge import run as J
    from ha.neighbours import retrieve

    rec = {"iso": [], "m1": [], "unload": 0, "parts": []}
    U = pd.DataFrame({"qid": QIDS, "text": [f"query {q}" for q in QIDS],
                      "source_pid": [f"p{q}" for q in QIDS],
                      "pool": ["P-std", "P-std", "P-persona", "P-persona"]})
    R = pd.DataFrame({"qid": R_QIDS, "text": [f"real {q}" for q in R_QIDS]})
    for s in range(5):
        nb = pd.DataFrame({"qid": QIDS, "neighbour_qids": [R_QIDS[s % 2:s % 2 + 5]] * 4,
                           "near_duplicate": [False, False, s == 1, False]})
        p = retrieve.out_path("fiqa", s, "U")
        p.parent.mkdir(parents=True, exist_ok=True)
        nb.to_parquet(p, index=False)

    def fake_split(ds, seed, part):
        rec["parts"].append(part)
        assert part == "R", "bulk judging must never load A"
        return R

    def fake_unload():
        rec["unload"] += 1
        return []

    def fake_iso(qid, query, passage, lang, domain, cache=None, client=None):
        rec["iso"].append((qid, passage, lang, domain))
        return rec.get("iso_fn", lambda q: _res())(qid)

    def fake_m1(qid, query, passage, neighbours, lang, domain, cache=None, client=None):
        rec["m1"].append((qid, tuple(neighbours)))
        return _res(F=3 if qid == "u2" else 5, third=qid == "u4")

    monkeypatch.setattr(calibrate, "amendment_recorded", lambda tag: tag == "A3")
    monkeypatch.setattr(llm, "require_digest", lambda tag, d: d)
    monkeypatch.setattr(llm, "unload_all", fake_unload)
    monkeypatch.setattr(retrieve, "load_U", lambda ds: U)
    monkeypatch.setattr(generate, "load_source_corpus", lambda ds: pd.DataFrame(
        {"pid": [f"p{q}" for q in QIDS], "text": [f"passage {q}" for q in QIDS]}))
    monkeypatch.setattr(splits, "load_split", fake_split)
    monkeypatch.setattr(langid, "classify", lambda t: {"label": "EN"})
    monkeypatch.setattr(J, "judge_iso", fake_iso)
    monkeypatch.setattr(J, "judge_m1", fake_m1)
    return rec


def test_full_run_outputs_neighbours_from_R_and_registry(env):
    from ha import registry

    out = B.run_bulk("fiqa")
    iso = pd.read_parquet(B.stage_log("fiqa", None).with_suffix(".parquet"))
    assert iso["qid"].tolist() == QIDS and iso["ok"].all() and iso["gate_pass"].all()
    assert {x[1] for x in env["iso"]} == {f"passage {q}" for q in QIDS}   # own source passage
    for s in range(5):
        m1 = pd.read_parquet(B.stage_log("fiqa", s).with_suffix(".parquet"))
        expect = [q for q in QIDS if not (s == 1 and q == "u3")]           # near-dup skipped
        assert m1["qid"].tolist() == expect and (m1["split_seed"] == s).all()
        assert not m1.set_index("qid").loc["u2", "gate_pass"]               # F = 3 < 4
        assert m1.set_index("qid").loc["u4", "third_run"]
        assert list(m1["neighbour_qids"].iloc[0]) == R_QIDS[s % 2:s % 2 + 5]
        meta = json.loads(B.stage_log("fiqa", s).with_suffix(".meta.json").read_text("utf-8"))
        assert meta["complete"] and meta["near_duplicate_skipped"] == (1 if s == 1 else 0)
        assert meta["amendment"] == "A3" and meta["prompt_version"] == "v0"
    assert all(set(n) <= {f"real {q}" for q in R_QIDS} and len(n) == 5 for _, n in env["m1"])
    assert set(env["parts"]) == {"R"} and env["unload"] >= 6                 # fresh state/stage
    assert len(env["m1"]) == 4 * 5 - 1 and len(env["iso"]) == 4
    reg = registry.load()
    assert len(reg) == 2 + 5 * 3 and (reg["run_id"] == out["registry_run_id"]).all()
    assert set(reg["metric"]) == {"judge_failure_rate", "judge_gate_pass_rate",
                                  "judge_third_run_rate"}


def test_rerun_resumes_without_rejudging_or_rerecording(env):
    from ha import registry

    first = B.run_bulk("fiqa")
    n_iso, n_m1 = len(env["iso"]), len(env["m1"])
    meta = B.stage_log("fiqa", 2).with_suffix(".meta.json")
    before = meta.read_text("utf-8")
    again = B.run_bulk("fiqa")
    assert (len(env["iso"]), len(env["m1"])) == (n_iso, n_m1)
    assert len(registry.load()) == 17
    assert meta.read_text("utf-8") == before                      # the judging run's record kept
    assert again == first and again["registry_run_id"] == first["registry_run_id"]


def test_transport_failures_are_retried_and_a_streak_stops_the_run(env, monkeypatch):
    env["iso_fn"] = lambda q: _fail("ConnectionError: refused") if q == "u2" else \
        (_fail("validation: bad json") if q == "u3" else _res())
    out = B.run_bulk("fiqa", seeds=(0,))
    st = {(x["method"], x["split_seed"]): x for x in out["stages"]}
    assert st[("M1-iso", None)]["failure_rate"] == 0.5
    env["iso_fn"] = lambda q: _res()
    env["iso"].clear()
    B.run_bulk("fiqa", seeds=(0,))
    assert [x[0] for x in env["iso"]] == ["u2"]          # validation failure stays recorded
    iso = pd.read_parquet(B.stage_log("fiqa", None).with_suffix(".parquet"))
    assert iso.set_index("qid")["ok"].to_dict() == {"u1": True, "u2": True, "u3": False,
                                                    "u4": True}
    monkeypatch.setattr(B, "MAX_TRANSPORT_STREAK", 2)
    env["iso_fn"] = lambda q: _fail("ResponseError: server down")
    with pytest.raises(RuntimeError, match="transport failures in a row"):
        B.run_bulk("fiqa", limit=4, seeds=(0,))


def test_smoke_is_separate_and_never_recorded(env):
    from ha import registry

    B.run_bulk("fiqa", seeds=(0,), limit=2)
    assert B.stage_log("fiqa", 0, smoke=True).with_suffix(".parquet").exists()
    assert not B.stage_log("fiqa", 0).exists()
    assert len(registry.load()) == 0


def test_blocked_until_a3_recorded(env, monkeypatch):
    from ha.judge import calibrate

    monkeypatch.setattr(calibrate, "amendment_recorded", lambda tag: False)
    with pytest.raises(B.BulkBlocked, match="A3"):
        B.run_bulk("fiqa")


def test_flatten_and_torn_log_line(tmp_path):
    f = B.flatten(_res(third=True))
    assert f["ok"] and f["gate_pass"] and f["n_calls"] == 3 and f["third_run"]
    assert len(json.loads(f["perm_outputs"])) == 3 and f["keep"] is True
    v = B.flatten(_fail("validation: x"))
    assert not v["gate_pass"] and v["retries"] == 1 and not B.is_transport_failure(v)
    assert B.is_transport_failure(B.flatten(_fail("ConnectionError: x")))
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"qid": "a", "ok": True}) + "\n{\"qid\": \"b\", \"o", encoding="utf-8")
    assert [r["qid"] for r in B.read_log(p)] == ["a"]
