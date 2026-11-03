import pandas as pd
import pytest

from ha.data import scenarios


def _chunks(n=40):
    rows = []
    for i in range(n):
        src = "incometax" if i % 2 else "epfo"
        rows.append({"chunk_id": f"{src}/c{i:03d}", "source": src,
                     "url": f"https://x.gov.in/iec/foportal/help/topic{i % 7}/p{i}",
                     "heading_path": "Company returns" if i == 5 else "", "kind": "section",
                     "text": "word " * 40, "n_words": 40})
    return pd.DataFrame(rows)


def _sheet(chunks):
    return scenarios.build_sheet(chunks, n=10, seed=17, quota={"incometax": 5, "epfo": 5},
                                 author="team")


def test_replacements_are_deterministic_same_source_and_new():
    chunks = _chunks()
    sheet = _sheet(chunks)
    skip = sheet[sheet["source"] == "incometax"]["scenario_id"].tolist()[:2]
    a = scenarios.draw_replacements(sheet, chunks, skip)
    b = scenarios.draw_replacements(sheet, chunks, skip)
    pd.testing.assert_frame_equal(a, b)
    assert (a["source"] == "incometax").all()
    assert not set(a["new_chunk_id"]) & set(sheet["chunk_id"])
    assert a["new_chunk_id"].is_unique
    assert "incometax/c005" not in set(a["new_chunk_id"])  # professional heading excluded
    assert (a["scenario"] == "").all() and (a["status"] == "NEEDS HUMAN-WRITTEN SCENARIO").all()
    with pytest.raises(ValueError):
        scenarios.draw_replacements(sheet, chunks, ["S999"])


def test_review_sheet_never_writes_official_check(tmp_path, monkeypatch):
    monkeypatch.setattr(scenarios, "hiaq_dir", lambda: tmp_path / "hiaq")
    (tmp_path / "hiaq").mkdir()
    review = tmp_path / "humanside" / "reviewed.csv"
    review.parent.mkdir()
    assert scenarios.check_output_path(review) == tmp_path / "humanside" / "reviewed_check.csv"
    official = tmp_path / "hiaq" / "scenario_sheet.csv"
    assert scenarios.check_output_path(official) == tmp_path / "hiaq" / "scenario_check.csv"


def test_exclude_forces_next_deterministic_candidate():
    chunks = _chunks()
    sheet = _sheet(chunks)
    skip = sheet[sheet["source"] == "incometax"]["scenario_id"].tolist()[:1]
    first = scenarios.draw_replacements(sheet, chunks, skip)["new_chunk_id"].item()
    second = scenarios.draw_replacements(sheet, chunks, skip, exclude=[first])
    assert second["new_chunk_id"].item() != first
    again = scenarios.draw_replacements(sheet, chunks, skip, exclude=[first])
    assert second.equals(again)
