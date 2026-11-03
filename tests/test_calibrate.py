import pandas as pd
import pytest

from ha import prereg
from ha.config import load_config
from ha.judge import calibrate as C

V1 = {"seed": 17, "cal_auth": {"fiqa": 30, "hiaq": 30}, "synthetic": {"fiqa": 70, "hiaq": 70},
      "split": {"cal_dev": 120, "cal_test": 80}, "practice_items": 20}
A2 = {"amendment": "A2", "seed": 17, "cal_auth": {"fiqa": 30, "hiaq": 0},
      "synthetic": {"fiqa": 70, "hiaq": 0}, "split": {"cal_dev": 60, "cal_test": 40},
      "practice_items": 20}


def _U(prefix, n=200):
    return pd.DataFrame({"qid": [f"{prefix}{i:03d}" for i in range(n)],
                         "text": [f"q{i}" for i in range(n)],
                         "pool": ["P-std" if i % 2 else "P-persona" for i in range(n)]})


def _auth(prefix, n=30):
    return pd.DataFrame({"qid": [f"{prefix}a{i}" for i in range(n)], "text": ["real"] * n})


def _use(monkeypatch, cfg, recorded=False):
    monkeypatch.setattr(C, "_cfg", lambda: cfg)
    monkeypatch.setattr(C, "amendment_recorded", lambda tag: recorded)


# ---- prereg-v1 behaviour (kept for reference and regression)
def test_v1_blocked_without_hiaq_half(monkeypatch):
    _use(monkeypatch, V1)
    with pytest.raises(C.CalibrationBlocked):
        C.build_items(_auth("f"), None, _U("f"), _U("h"))


def test_v1_items_and_split_match_prereg(monkeypatch):
    _use(monkeypatch, V1)
    items = C.build_items(_auth("f"), _auth("h"), _U("f"), _U("h"))
    assert len(items) == 200
    syn = items[items.kind == "synthetic"]
    assert syn.groupby(["dataset", "pool"]).size().to_dict() == {
        ("fiqa", "P-persona"): 35, ("fiqa", "P-std"): 35,
        ("hiaq", "P-persona"): 35, ("hiaq", "P-std"): 35}
    sp = C.split_items(items)
    by = sp.groupby(["dataset", "kind", "split"]).size().to_dict()
    assert by[("fiqa", "authentic", "cal-dev")] == 18
    assert by[("hiaq", "synthetic", "cal-test")] == 28
    assert sp.equals(C.split_items(items))                     # deterministic (seed 17)
    assert C.build_items(_auth("f"), _auth("h"), _U("f"), _U("h")).equals(items)


# ---- amendment A2 (D-059)
def test_a2_refuses_until_the_amendment_is_recorded(monkeypatch):
    _use(monkeypatch, A2, recorded=False)
    with pytest.raises(C.CalibrationBlocked, match="A2"):
        C.build_items(_auth("f"), None, _U("f"), None)


def test_a2_items_split_and_fiqa_identity_with_v1(monkeypatch):
    _use(monkeypatch, A2, recorded=True)
    items = C.build_items(_auth("f"), None, _U("f"), None)
    assert len(items) == 100 and set(items["dataset"]) == {"fiqa"}
    assert items["kind"].value_counts().to_dict() == {"synthetic": 70, "authentic": 30}
    assert items[items.kind == "synthetic"]["pool"].value_counts().to_dict() == {
        "P-std": 35, "P-persona": 35}
    sp = C.split_items(items)
    assert sp["split"].value_counts().to_dict() == {"cal-dev": 60, "cal-test": 40}
    by = sp.groupby(["kind", "split"]).size().to_dict()
    assert by == {("authentic", "cal-dev"): 18, ("authentic", "cal-test"): 12,
                  ("synthetic", "cal-dev"): 42, ("synthetic", "cal-test"): 28}
    # removing the HiAQ half changes nothing for the FiQA items: same items, same split
    _use(monkeypatch, V1)
    v1 = C.split_items(C.build_items(_auth("f"), _auth("h"), _U("f"), _U("h")))
    v1_fiqa = v1[v1.dataset == "fiqa"].set_index("qid")["split"]
    assert sp.set_index("qid")["split"].sort_index().equals(v1_fiqa.sort_index())


def test_a2_still_requires_exactly_30_fiqa_cal_auth(monkeypatch):
    _use(monkeypatch, A2, recorded=True)
    with pytest.raises(C.CalibrationBlocked, match="fiqa"):
        C.build_items(_auth("f", n=29), None, _U("f"), None)


def test_shipped_config_is_a2_and_only_drops_the_hiaq_half():
    cfg = load_config("calibration.yaml")
    assert cfg["amendment"] == "A2"
    assert {k: cfg[k] for k in ("seed", "practice_items")} == {"seed": 17, "practice_items": 20}
    assert cfg["cal_auth"] == {"fiqa": 30, "hiaq": 0}
    assert cfg["synthetic"] == {"fiqa": 70, "hiaq": 0}
    assert cfg["split"] == {"cal_dev": 60, "cal_test": 40}
    dev_share = cfg["split"]["cal_dev"] / sum(cfg["split"].values())
    assert dev_share == V1["split"]["cal_dev"] / sum(V1["split"].values())   # same 60/40 ratio


@pytest.mark.parametrize(("tail", "expected"), [
    ("_None yet._\n", False),
    ("### A1 · 2026-10-12 · fallback\ntext\n", False),
    ("### A2 · 2026-10-12 · calibration\ntext\n", True),
    ("Some text mentioning A2 inline.\n", False),
])
def test_amendment_recorded_reads_only_the_amendments_section(tmp_path, monkeypatch, tail,
                                                              expected):
    p = tmp_path / "preregistration.md"
    p.write_text("# Prereg\nBody mentions ### A2 · in a quote? no.\n\n## Amendments\n\n" + tail,
                 encoding="utf-8")
    monkeypatch.setattr(prereg, "PREREG_PATH", p)
    assert C.amendment_recorded("A2") is expected


def test_real_prereg_has_a1_and_a2_recorded():
    # recorded 2026-10-10 (D-060); the frozen body is guarded by tests/test_prereg_frozen.py
    assert C.amendment_recorded("A1") and C.amendment_recorded("A2")
