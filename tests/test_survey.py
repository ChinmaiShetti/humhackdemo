import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from ha.data import survey
from ha.paths import get_paths

REAL_HIAQ = get_paths().data / "hiaq"        # resolved before the isolation fixture runs
GS = Path(__file__).resolve().parents[1] / "tools" / "make_forms.gs"


@pytest.mark.parametrize(("text", "label"), [
    ("my number is 9876543210 call me", "PHONE"),
    ("+91 98765 43210 pe otp nahi aaya", "PHONE"),
    ("aadhaar 1234 5678 9012 update", "AADHAAR"),
    ("mail me at a.b@gmail.com", "EMAIL"),
    ("pan ABCDE1234F link nahi ho raha", "PAN"),
])
def test_pii_is_stripped(text, label):
    clean, found = survey.strip_pii(text)
    assert label in found and f"[{label}]" in clean


def test_no_false_pii_on_normal_queries():
    for q in ["itr refund 2024-25 not received", "pf withdrawal after 2 months",
              "form 16 kaise download kare", "upi limit 100000 per day"]:
        assert survey.strip_pii(q) == (q, [])


def _scenarios():
    return pd.DataFrame({"scenario_id": [f"S{i:03d}" for i in range(150)],
                         "source": ["epfo" if i % 4 == 0 else "incometax" for i in range(150)],
                         "scenario": [f"situation {i}" for i in range(150)],
                         "chunk_id": [f"c{i}" for i in range(150)]})


def test_variants_cover_each_scenario_once_and_mix_sources():
    sc = _scenarios()
    variants = survey.build_variants(sc)
    ids = [it["scenario_id"] for v in variants for it in v["items"]]
    assert len(variants) == 25 and all(len(v["items"]) == 6 for v in variants)
    assert sorted(ids) == sorted(sc["scenario_id"])
    src = sc.set_index("scenario_id")["source"]
    assert all({src[it["scenario_id"]] for it in v["items"]} == {"epfo", "incometax"}
               for v in variants)
    assert variants == survey.build_variants(sc)


def test_ingest_maps_columns_strips_pii_and_tags_language(tmp_path, monkeypatch):
    sc = _scenarios()
    variants = survey.build_variants(sc)
    monkeypatch.setattr(survey, "hiaq_dir", lambda: tmp_path)
    sc.to_csv(tmp_path / "scenario_sheet.csv", index=False, encoding="utf-8-sig")
    (tmp_path / "survey_variants.json").write_text(json.dumps({"variants": variants}),
                                                   encoding="utf-8")
    (tmp_path / "responses").mkdir()
    cols = ["Timestamp"] + [f"q{j}" for j in range(6)] + ["langs", "device"]
    answers = ["upi payment fail ho gaya", "", "call 9876543210", "how to reset password",
               "  pf   kab   aayega ", "refund status"]
    pd.DataFrame([["2026-10-20 10:00:00", *answers, "English", "Phone"]], columns=cols).to_csv(
        tmp_path / "responses" / "form_03.csv", index=False)
    out = survey.ingest()
    assert len(out) == 6
    assert out["scenario_id"].tolist() == [it["scenario_id"] for it in variants[2]["items"]]
    assert out.loc[2, "text"] == "call [PHONE]" and out.loc[2, "pii_removed"] == "PHONE"
    assert out.loc[4, "text"] == "pf kab aayega"          # whitespace only; typos kept
    assert bool(out.loc[1, "empty"]) and out.loc[0, "lang_class"] == "HINGLISH"
    assert out.loc[0, "submitted_date"] == "2026-10-20"


# ---- spec §6.3 exception gate (D-052, D-053)
S090 = ("You are unmarried and want to name someone outside your family to receive your benefit "
        "if you die. You want to know whether that nomination would still apply if you later "
        "marry or have a family.")
S091 = ("A relative who served in the military and was also covered by an employee pension "
        "scheme has died. You want to know whether the family can receive pension support in "
        "addition to the military pension.")
REASON = {"S090": "rare document terms: unmarried", "S091": "rare document terms: military"}


def _sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _gate_fixture(tmp_path, monkeypatch, rejects, warns=(), exceptions=None):
    """150-row sheet (S090/S091 carry the approved texts) plus a check file with the given
    REJECT and WARN rows, in a temp hiaq dir."""
    sc = _scenarios()
    sc["scenario_id"] = [f"S{i + 1:03d}" for i in range(150)]
    sc.loc[sc.scenario_id == "S090", "scenario"] = S090
    sc.loc[sc.scenario_id == "S091", "scenario"] = S091
    status = {sid: "REJECT" for sid in rejects} | {sid: "WARN" for sid in warns}
    check = pd.DataFrame({"scenario_id": sc["scenario_id"],
                          "status": [status.get(i, "PASS") for i in sc["scenario_id"]],
                          "reasons": [REASON.get(i, "rare document terms: foo")
                                      if status.get(i) == "REJECT" else
                                      ("reuses 65% of its words" if status.get(i) == "WARN"
                                       else "") for i in sc["scenario_id"]]})
    monkeypatch.setattr(survey, "hiaq_dir", lambda: tmp_path)
    sc.to_csv(tmp_path / "scenario_sheet.csv", index=False, encoding="utf-8-sig")
    check.to_csv(tmp_path / "scenario_check.csv", index=False, encoding="utf-8-sig")
    if exceptions is None:
        exceptions = {sid: {"decision": "D-052", "checker_reason": REASON[sid],
                            "scenario_sha256": _sha(t)} for sid, t in (("S090", S090),
                                                                       ("S091", S091))}
    monkeypatch.setattr(survey, "load_check_exceptions", lambda: exceptions)
    return sc, check


def _payload(tmp_path):
    return json.loads((tmp_path / "survey_variants.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("sid", ["S090", "S091"])
def test_listed_exception_lets_the_builder_proceed(tmp_path, monkeypatch, sid):
    _gate_fixture(tmp_path, monkeypatch, rejects=[sid])
    variants = survey.write_variants()
    assert len(variants) == 25
    assert [e["scenario_id"] for e in _payload(tmp_path)["check_exceptions"]] == [sid]


def test_both_exceptions_together(tmp_path, monkeypatch):
    _gate_fixture(tmp_path, monkeypatch, rejects=["S090", "S091"])
    survey.write_variants()
    got = {e["scenario_id"]: e["decision"] for e in _payload(tmp_path)["check_exceptions"]}
    assert got == {"S090": "D-052", "S091": "D-052"}


def test_unlisted_reject_still_blocks(tmp_path, monkeypatch):
    _gate_fixture(tmp_path, monkeypatch, rejects=["S090", "S091", "S005"])
    with pytest.raises(RuntimeError, match=r"S005"):
        survey.write_variants()
    assert not (tmp_path / "survey_variants.json").exists()


def test_warn_rows_pass_without_any_exception(tmp_path, monkeypatch):
    _gate_fixture(tmp_path, monkeypatch, rejects=[], warns=["S060", "S066"], exceptions={})
    survey.write_variants()
    assert _payload(tmp_path)["check_exceptions"] == []


@pytest.mark.parametrize("tamper", ["reason", "text", "decision", "status"])
def test_gate_fails_closed(tmp_path, monkeypatch, tamper):
    sc, check = _gate_fixture(tmp_path, monkeypatch, rejects=["S090"])
    exc = survey.load_check_exceptions()
    if tamper == "reason":          # the checker now reports a different/new rare term
        check.loc[check.scenario_id == "S090", "reasons"] = "rare document terms: unmarried, pf"
    elif tamper == "text":          # the approved scenario text was edited
        sc.loc[sc.scenario_id == "S090", "scenario"] = S090 + " "
    elif tamper == "decision":      # an entry without a decision reference
        exc = {"S090": {**exc["S090"], "decision": ""}}
    else:                           # EMPTY is never excusable
        check.loc[check.scenario_id == "S090", "status"] = "EMPTY"
    with pytest.raises(RuntimeError):
        survey.gate_check(sc, check, exc)


def test_stale_check_file_blocks(tmp_path, monkeypatch):
    sc, check = _gate_fixture(tmp_path, monkeypatch, rejects=[])
    with pytest.raises(RuntimeError, match="does not cover"):
        survey.gate_check(sc, check.iloc[:-1], {})


def test_shipped_allowlist_is_exactly_s090_s091():
    exc = survey.load_check_exceptions()
    assert set(exc) == {"S090", "S091"}
    assert {k: v["decision"] for k, v in exc.items()} == {"S090": "D-052", "S091": "D-052"}
    assert {k: v["checker_reason"] for k, v in exc.items()} == REASON
    assert exc["S090"]["scenario_sha256"] == _sha(S090)
    assert exc["S091"]["scenario_sha256"] == _sha(S091)


# ---- form creation record (tools/make_forms.gs -> forms_manifest.json)
def _forms_log(variants, **tamper):
    lines = ["12:00:01 PM Notice Execution started"]
    for v in variants:
        row = {"form_number": v["variant"], "form_id": f"id{v['variant']:02d}",
               "edit_url": f"https://docs.google.com/forms/d/id{v['variant']:02d}/edit",
               "response_url": f"https://docs.google.com/forms/d/e/x{v['variant']:02d}/viewform",
               "scenario_ids": [it["scenario_id"] for it in v["items"]],
               "item_titles": [f'"{it["scenario"]}"' for it in v["items"]],
               "accepting_responses": False, "created_at": "2026-09-29T10:00:00.000Z"}
        if v["variant"] == 3:
            row.update(tamper)
        lines.append(f"12:00:0{v['variant'] % 10} PM Info MANIFEST_JSON " + json.dumps(row))
    return "\n".join(lines + ["12:01:00 PM Notice Execution completed"])


def _manifest_env(tmp_path, monkeypatch, **tamper):
    variants = survey.build_variants(_scenarios())
    monkeypatch.setattr(survey, "hiaq_dir", lambda: tmp_path)
    (tmp_path / "survey_variants.json").write_text(json.dumps({"variants": variants}),
                                                   encoding="utf-8")
    (tmp_path / "forms_creation_log.txt").write_text(_forms_log(variants, **tamper),
                                                     encoding="utf-8")
    return variants


def test_forms_manifest_recorded_when_every_form_matches(tmp_path, monkeypatch):
    variants = _manifest_env(tmp_path, monkeypatch)
    m = survey.record_forms_manifest()
    assert len(m["forms"]) == 25 and [f["form_number"] for f in m["forms"]] == list(range(1, 26))
    assert m["forms"][0]["scenario_ids"] == [it["scenario_id"] for it in variants[0]["items"]]
    assert set(m["forms"][0]) == set(survey.MANIFEST_FIELDS)
    assert json.loads((tmp_path / "forms_manifest.json").read_text(encoding="utf-8")) == m


@pytest.mark.parametrize("tamper", [
    {"item_titles": ['"edited"'] * 6},                 # scenario text changed in the form
    {"scenario_ids": ["S999"] * 6},                    # wrong assignment
    {"accepting_responses": True},                     # opened before pilot/launch
    {"form_id": "id01"},                               # duplicate form
    {"response_url": "https://example.com/form"},      # not a Google Form
])
def test_forms_manifest_refuses_any_mismatch(tmp_path, monkeypatch, tamper):
    _manifest_env(tmp_path, monkeypatch, **tamper)
    with pytest.raises(ValueError):
        survey.record_forms_manifest()
    assert not (tmp_path / "forms_manifest.json").exists()


def test_forms_manifest_refuses_missing_form(tmp_path, monkeypatch):
    _manifest_env(tmp_path, monkeypatch)
    log = (tmp_path / "forms_creation_log.txt").read_text(encoding="utf-8")
    (tmp_path / "forms_creation_log.txt").write_text(
        "\n".join(line for line in log.splitlines() if '"form_number": 7,' not in line),
        encoding="utf-8")
    with pytest.raises(ValueError, match="form numbers"):
        survey.record_forms_manifest()


def test_make_forms_script_is_closed_neutral_and_pinned():
    gs = GS.read_text(encoding="utf-8")
    assert "var VARIANTS = {" in gs and "VARIANTS_JSON = '" not in gs  # object, not a string
    assert "setAcceptingResponses(false)" in gs and "setCollectEmail(false)" in gs
    assert gs.index("addPageBreakItem()") < gs.index("cfg.lang_question")
    assert not re.search(r"hinglish|humanalign|synthetic|llm|retriev", gs.split("*/", 1)[1], re.I)
    real = REAL_HIAQ / "survey_variants.json"
    if real.exists():                                     # the pinned fingerprint is current
        payload = json.loads(real.read_text(encoding="utf-8"))
        pinned = re.search(r"EXPECTED_VARIANTS_SHA256 = '([0-9a-f]{64})'", gs).group(1)
        assert pinned == survey.variants_fingerprint(payload)


def test_pilot_ingest_never_writes_the_dataset_file(tmp_path, monkeypatch):
    sc = _scenarios()
    variants = survey.build_variants(sc)
    monkeypatch.setattr(survey, "hiaq_dir", lambda: tmp_path)
    sc.to_csv(tmp_path / "scenario_sheet.csv", index=False, encoding="utf-8-sig")
    (tmp_path / "survey_variants.json").write_text(json.dumps({"variants": variants}),
                                                   encoding="utf-8")
    (tmp_path / "pilot_responses").mkdir()
    cols = ["Timestamp"] + [f"q{j}" for j in range(6)] + ["langs", "device"]
    pd.DataFrame([["2026-09-30 10:00:00", "a", "b", "c", "d", "e", "f", "English", "Phone"]],
                 columns=cols).to_csv(tmp_path / "pilot_responses" / "form_01.csv", index=False)
    out = survey.ingest(pilot=True)
    assert len(out) == 6 and (tmp_path / "pilot_queries.parquet").exists()
    assert not (tmp_path / "survey_queries.parquet").exists()
    assert out["scenario_id"].tolist() == [it["scenario_id"] for it in variants[0]["items"]]


# ---- collaborative-sheet set (D-056)
def _collab_env(tmp_path, monkeypatch):
    sc = _scenarios()
    payload = {"variants": survey.build_variants(sc)}
    monkeypatch.setattr(survey, "hiaq_dir", lambda: tmp_path)
    sc.to_csv(tmp_path / "scenario_sheet.csv", index=False, encoding="utf-8-sig")
    (tmp_path / "survey_variants.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload, survey.collab_template(payload)


def _fill(t, block, code, answers=None):
    m = t["block"] == block
    t.loc[m, "contributor_code"] = code
    t.loc[m, "answer"] = answers or [f"{code} answer {i}" for i in range(m.sum())]
    return t


def test_collab_template_covers_each_situation_four_times(tmp_path, monkeypatch):
    payload, t = _collab_env(tmp_path, monkeypatch)
    assert len(t) == 600 and t["block"].nunique() == 100 and list(t.columns) == survey.COLLAB_COLS
    assert set(t["scenario_id"].value_counts()) == {4}
    assert t.groupby("block")["form"].nunique().eq(1).all()
    assert t.loc[t.block == "B026", "scenario_id"].tolist() == \
        [it["scenario_id"] for it in payload["variants"][0]["items"]]   # round 2 repeats form 01
    assert (t[["answer", "contributor_code", "languages", "device"]] == "").all().all()
    path = survey.write_collab_template()
    with pytest.raises(FileExistsError):                  # never overwrites a sheet in use
        survey.write_collab_template()
    assert len(pd.read_csv(path, encoding="utf-8-sig")) == 600


def test_collab_ingest_writes_only_the_separate_set(tmp_path, monkeypatch):
    _, t = _collab_env(tmp_path, monkeypatch)
    t = _fill(t, "B001", "C01", ["call 9876543210", "pf  kab aayega", "", "a", "b", "c"])
    t = _fill(t, "B002", "C02")
    t.loc[t.block == "B002", "languages"] = ["English"] + [""] * 5
    (tmp_path / "collab").mkdir()
    t.to_csv(tmp_path / "collab" / "collab_sheet_filled.csv", index=False, encoding="utf-8-sig")
    out = survey.ingest_collab()
    assert len(out) == 12 and set(out["source"]) == {"collab"} and out["block"].nunique() == 2
    assert (tmp_path / "collab" / "collab_queries.parquet").exists()
    assert not (tmp_path / "survey_queries.parquet").exists()
    b1 = out[out.block == "B001"].reset_index(drop=True)
    assert b1.loc[0, "text"] == "call [PHONE]" and b1.loc[1, "text"] == "pf kab aayega"
    assert bool(b1.loc[2, "empty"])
    assert "C01" not in set(out["respondent"])            # contributor codes are hashed
    assert set(out.loc[out.block == "B002", "languages"]) == {"English"}


@pytest.mark.parametrize("problem", ["edited_situation", "no_code", "two_people", "same_form"])
def test_collab_ingest_refuses_bad_sheets(tmp_path, monkeypatch, problem):
    _, t = _collab_env(tmp_path, monkeypatch)
    t = _fill(t, "B001", "C01")
    if problem == "edited_situation":
        t.loc[0, "situation"] = "changed"
    elif problem == "no_code":
        t.loc[t.block == "B002", "answer"] = "x"
    elif problem == "two_people":
        t.loc[t.index[1], "contributor_code"] = "C02"
    else:                                                 # B026 = form 01 again, same person
        t = _fill(t, "B026", "C01")
    (tmp_path / "collab").mkdir()
    t.to_csv(tmp_path / "collab" / "collab_sheet_filled.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError):
        survey.ingest_collab()
    assert not (tmp_path / "collab" / "collab_queries.parquet").exists()
