"""Import an EXTERNAL synthetic query set (not a pre-registered P3 pool) for one dataset (D-058).

Read-only on the input CSV (columns: scenario_id, source, language_mix, question_text). Writes
HA_DATA/synthetic_external/<ds>/<name>/{queries.parquet, stats.json}. This is deliberately outside
data/synthetic/, so load_U() and every pre-registered pool path never see it. Text is kept exactly
as given (no normalisation). Each scenario maps to its official source chunk (`source_pid`) via the
official scenario sheet. Refuses to overwrite.

Usage: .venv\\Scripts\\python tools\\import_external_synthetic.py CSV --name X-team150
       --dataset hiaq --provenance "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ha.data.langid import classify
from ha.data.survey import strip_pii
from ha.paths import get_paths
from ha.utils import git_sha, utc_now_iso


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default="hiaq")
    ap.add_argument("--provenance", required=True)
    a = ap.parse_args()

    raw = Path(a.csv).read_bytes()
    src = pd.read_csv(a.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    cols = ["scenario_id", "source", "language_mix", "question_text"]
    assert list(src.columns) == cols, src.columns
    sheet_path = get_paths().data / "hiaq" / "scenario_sheet.csv"
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sheet = sheet.set_index("scenario_id")
    assert src["scenario_id"].is_unique and set(src["scenario_id"]) <= set(sheet.index)
    assert (src["question_text"].str.strip() != "").all()

    out_dir = get_paths().data / "synthetic_external" / a.dataset / a.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; not overwriting")
    q = pd.DataFrame({
        "qid": [f"{a.dataset}-{a.name}-{s}" for s in src["scenario_id"]],
        "text": src["question_text"],                       # exact, unmodified
        "source_pid": [sheet.at[s, "chunk_id"] for s in src["scenario_id"]],
        "i": range(len(src)),
        "seed": None, "pool": a.name, "split_seed": None, "dataset": a.dataset,
        "template": "external", "template_hash": None, "model_digest": None,
        "scenario_id": src["scenario_id"],
        "service": [sheet.at[s, "source"] for s in src["scenario_id"]],
        "language_mix_given": src["language_mix"],
        "provenance": a.provenance,
    })
    assert q["text"].tolist() == src["question_text"].tolist()
    lid = [classify(t) for t in q["text"]]
    stats = {
        "dataset": a.dataset, "pool": a.name,
        "kind": "EXTERNAL synthetic (not a pre-registered pool)",
        "provenance": a.provenance, "n": len(q),
        "source_file": Path(a.csv).name, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "scenario_sheet_sha256": hashlib.sha256(sheet_path.read_bytes()).hexdigest(),
        "service": q["service"].value_counts().to_dict(),
        "language_mix_given": q["language_mix_given"].value_counts().to_dict(),
        "lid_automated": pd.Series([x["label"] for x in lid]).value_counts().to_dict(),
        "chars_median": int(q["text"].str.len().median()),
        "pii_rows": int(sum(bool(strip_pii(t)[1]) for t in q["text"])),
        "allowed_use": ("exploratory, labelled analyses only; never authentic HiAQ data, "
                        "calibration items, anchors, R/A, neighbours or any pre-registered pool"),
        "git_sha": git_sha(), "created_at": utc_now_iso(),
    }
    out_dir.mkdir(parents=True)
    q.to_parquet(out_dir / "queries.parquet", index=False)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("n", "service", "language_mix_given", "lid_automated",
                                            "pii_rows", "source_sha256")}, indent=1))


if __name__ == "__main__":
    main()
