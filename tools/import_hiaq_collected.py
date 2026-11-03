"""Import HiAQ collected responses (D-065): one query per scenario, compiled by the team.

Provenance is the project lead's statement (2026-10-16): queries written by people for the 150
scenarios. The raw file is kept byte-identical in data/hiaq/incoming/<batch>/raw/ (SHA256SUMS).
Text is kept as written apart from PII placeholders and whitespace (as `ha data hiaq-ingest`).
Each row maps to its chunk through the OFFICIAL scenario sheet; rows whose situation wording
differs from the official sheet are flagged, and rows answering a scenario whose chunk was
replaced (D-051) get no chunk, so they are never scored against the wrong passage.

Role (prereg §8, A1): fewer than 300 eligible queries, so HiAQ stays a pilot. These queries
feed only analyses labelled exploratory, never H1-H4's confirmatory or secondary tests.

usage: python tools/import_hiaq_collected.py <batch dir name> <csv file name>
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from difflib import SequenceMatcher

import pandas as pd

from ha.data.langid import classify
from ha.data.survey import strip_pii
from ha.paths import get_paths

REPLACED_CHUNK = {"S017", "S025", "S136", "S147"}      # D-051: original scenario, chunk replaced


def main(batch: str, name: str) -> None:
    data = get_paths().data
    raw = data / "hiaq" / "incoming" / batch / "raw" / name
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    df = pd.read_csv(raw, encoding="utf-8-sig", dtype=str)
    qcol = next(c for c in df.columns if "query" in c.lower())
    off = pd.read_csv(data / "hiaq" / "scenario_sheet.csv", encoding="utf-8-sig", dtype=str)
    off = off.set_index("scenario_id")
    assert df["Scenario ID"].is_unique and set(df["Scenario ID"]) <= set(off.index)
    rows = []
    for r in df.to_dict("records"):
        sid = r["Scenario ID"]
        text, pii = strip_pii(re.sub(r"\s+", " ", str(r[qcol])).strip())
        sim = SequenceMatcher(None, str(r["Situation to imagine"]).strip(),
                              str(off.loc[sid, "scenario"]).strip()).ratio()
        lid = classify(text)
        rows.append({
            "qid": f"hiaq-col-{sid}", "scenario_id": sid, "service": r["Service"],
            "chunk_id": None if sid in REPLACED_CHUNK else off.loc[sid, "chunk_id"],
            "text": text, "pii_removed": pii, "empty": not text,
            "lang_class": lid["label"], "cmi": lid.get("cmi"),
            "situation_official": sim >= 0.999, "situation_similarity": round(sim, 3),
            "chunk_replaced": sid in REPLACED_CHUNK,
            "source": "collected", "provenance": "project lead's statement, 2026-10-16 (D-065)",
            "batch": batch, "raw_sha256": sha})
    out = pd.DataFrame(rows)
    dst = data / "hiaq" / "collected" / f"{batch}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    stats = {"batch": batch, "raw_file": name, "raw_sha256": sha, "rows": len(out),
             "empty": int(out["empty"].sum()), "pii_rows": int(out["pii_removed"].map(bool).sum()),
             "with_chunk": int(out["chunk_id"].notna().sum()),
             "situation_not_official": out.loc[~out["situation_official"], "scenario_id"].tolist(),
             "lang_class": out["lang_class"].value_counts().to_dict(),
             "service": out["service"].value_counts().to_dict()}
    dst.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
