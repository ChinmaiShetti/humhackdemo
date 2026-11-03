"""Build the LID adjudication table for all 100 audit examples (D-043, D-044).

Read-only on the inputs. Writes data/humanside/langid_adjudication_table.csv.
- `human_final_label` is HUMAN-CONTROLLED: this script never fills it; it only carries over
  what a human has already entered in langid_review_100.csv.
- ChatGPT's recommendation is the `provisional_label` column of langid_review_100.csv
  (optionally overridden by `--recs PATH`). It goes into `chatgpt_recommendation` only, never
  into `human_final_label`.
- System (classifier) predictions are deliberately NOT included, so the review stays blind.

Usage: .venv\\Scripts\\python tools\\build_lid_adjudication.py [--recs PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "humanside" / "langid_review_100.csv"
OUT = ROOT / "data" / "humanside" / "langid_adjudication_table.csv"
HIGH_PRIORITY = ["L14", "L29", "L53", "L54", "L55", "L65", "L88", "L89", "L99", "L100"]
REC_COLUMNS = ["assistant_recommendation", "chatgpt_recommendation", "recommended_label",
               "assistant_label", "recommendation"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recs", default=None, help="CSV with ChatGPT recommendations by example_id")
    args = ap.parse_args()

    audit = pd.read_csv(AUDIT, encoding="utf-8-sig", keep_default_na=False, dtype=str)
    table = audit[["example_id", "text", "source", "current_label", "provisional_label",
                   "review_status", "human_final_label"]].copy()
    # In langid_review_100.csv the ChatGPT recommendation IS `provisional_label` (team, 2026-09-28).
    table["chatgpt_recommendation"] = table["provisional_label"]
    rec_note = "provisional_label column of langid_review_100.csv"
    if args.recs:
        recs = pd.read_csv(args.recs, encoding="utf-8-sig", keep_default_na=False, dtype=str)
        col = next((c for c in REC_COLUMNS if c in recs.columns), None)
        if col is None:
            raise SystemExit(f"no recommendation column in {args.recs}; expected one of "
                             f"{REC_COLUMNS}, found {list(recs.columns)}")
        rec = dict(zip(recs["example_id"], recs[col], strict=True))
        missing = set(table["example_id"]) - set(rec)
        if missing:
            raise SystemExit(f"recommendations missing for {sorted(missing)}")
        table["chatgpt_recommendation"] = table["example_id"].map(rec)
        rec_note = f"from {args.recs} column {col}"

    def attention(r) -> str:
        reasons = []
        if r.example_id in HIGH_PRIORITY:
            reasons.append("team high-priority case")
        if r.review_status in ("HUMAN_REVIEW", "REVIEW"):
            reasons.append(f"review_status={r.review_status}")
        if r.chatgpt_recommendation != r.provisional_label:
            reasons.append("ChatGPT recommendation differs from provisional label")
        if r.current_label and r.current_label != r.chatgpt_recommendation:
            reasons.append("original test label differs from ChatGPT recommendation")
        return "YES: " + "; ".join(reasons) if reasons else "no (spot-check optional)"

    table["human_attention"] = [attention(r) for r in table.itertuples()]
    table = table.rename(columns={
        "current_label": "original_test_label (30 originals only)",
        "provisional_label": "provisional_label (audit draft)",
        "human_final_label": "human_final_label (HUMAN-CONTROLLED)",
    })
    order = ["example_id", "text", "source", "original_test_label (30 originals only)",
             "provisional_label (audit draft)", "chatgpt_recommendation",
             "human_final_label (HUMAN-CONTROLLED)", "human_attention", "review_status"]
    table[order].to_csv(OUT, index=False, encoding="utf-8-sig")
    n_att = table["human_attention"].str.startswith("YES").sum()
    print(f"wrote {OUT} ({len(table)} rows; {n_att} need human attention; recommendations: "
          f"{rec_note}; human_final_label filled: "
          f"{(table['human_final_label (HUMAN-CONTROLLED)'].str.strip() != '').sum()})")


if __name__ == "__main__":
    main()
