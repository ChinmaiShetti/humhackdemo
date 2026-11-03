"""Aggregate the study's results into app/web/public/lab.json for the demo's lab page.

Every number is read from results/eval/*.json or paper/latex/figdata/ (the same sources as the
paper's tables and figures). Only aggregates are written: no query text, no per-query rows.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "eval"
FIGDATA = ROOT / "paper" / "latex" / "figdata"
OUT = ROOT / "app" / "web" / "public" / "lab.json"

MAIN = {"fiqa": "dev-fiqa", "cqa_android": "heldout-cqa_android", "hiaq": "exploratory-hiaq"}
NAMES = {
    "fiqa": ("FiQA", "Personal-finance forum questions"),
    "cqa_android": ("Android", "Android Stack Exchange questions"),
    "hiaq": ("HiAQ", "Income Tax and EPFO help: English, Hinglish, Hindi"),
    "cqa_webmasters": ("Webmasters", "Stack Exchange"),
    "nfcorpus": ("NFCorpus", "Nutrition and medicine"),
    "scifact": ("SciFact", "Scientific claims"),
    "cqa_mathematica": ("Mathematica", "Stack Exchange"),
    "cqa_programmers": ("Programmers", "Stack Exchange"),
}
SYSTEMS = {
    "C1": ("BM25", "word match (k1 0.9, b 0.4)"),
    "C2": ("BM25 alt", "word match (k1 1.2, b 0.75)"),
    "C3": ("Dense", "meaning match (BGE-M3)"),
    "C4": ("Hybrid", "words + meaning, fused"),
    "C5": ("Dense + rerank", "meaning, then reranker"),
    "C6": ("BM25 + rerank", "words, then reranker"),
    "C7": ("Hybrid + rerank", "fused, then reranker"),
}
SET_NAMES = {"A": "Real users", "B0": "Random synthetic", "M1": "Judge-picked synthetic",
             "OVL": "Overlap-matched synthetic"}


def load(name: str) -> dict:
    return json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))


def r4(x: float) -> float:
    return round(float(x), 4)


def ndcg_by_set(ds: str) -> dict[str, dict[str, float]]:
    """Mean nDCG@10 per system and set, over seeds (and over draws for drawn sets)."""
    rows = load(MAIN[ds])["score_vectors"][ds]
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for c in SYSTEMS:
            acc[r["set"]][c].append(r[c])
    return {s: {c: r4(np.mean(v)) for c, v in cs.items()} for s, cs in acc.items()
            if s in ("A", "B0", "M1")}


def overlap_hist(ds: str, bins: int = 10) -> dict[str, list[float]]:
    """Share of questions per overlap bin: real (R of split 0) vs the synthetic pool U."""
    vals: dict[str, list[float]] = {"R": [], "U": []}
    with open(FIGDATA / "query_features.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == ds:
                vals[row["part"]].append(float(row["overlap"]))
    edges = np.linspace(0, 1, bins + 1)
    out = {}
    for part, v in vals.items():
        h, _ = np.histogram(np.clip(v, 0, 1), bins=edges)
        out[part] = [r4(x) for x in h / max(len(v), 1)]
    out["n"] = {k: len(v) for k, v in vals.items()}
    return out


def main() -> None:
    lab: dict = {"systems": {k: {"name": a, "how": b} for k, (a, b) in SYSTEMS.items()},
                 "sets": SET_NAMES, "datasets": {}, "replication": {}}
    ov = load("explore_overlap")["datasets"]
    for ds in MAIN:
        tau = load(MAIN[ds])["bootstrap"]["tau_seed_mean"][ds]
        v2 = load(f"v2-{ds}")["bootstrap"]["tau_seed_mean"]
        taus = {m: {k: r4(tau[m][k]) for k in ("tau_b", "ci_low", "ci_high")}
                for m in ("B0", "B1", "M1", "M1-iso")}
        taus.update({m: {k: r4(v2[m][k]) for k in ("tau_b", "ci_low", "ci_high")}
                     for m in ("OVL", "NN")})
        comp = load(f"v2-{ds}")["bootstrap"]["comparisons"]["OVL-B0"]
        m1 = load(MAIN[ds])["bootstrap"]["comparisons"]["M1-B0"]
        m1 = m1.get(ds, m1["pooled"])
        sm = ov[ds]["seed_mean"]
        lab["datasets"][ds] = {
            "name": NAMES[ds][0], "about": NAMES[ds][1],
            "role": {"fiqa": "development", "cqa_android": "held-out test",
                     "hiaq": "exploratory pilot"}[ds],
            "ndcg": ndcg_by_set(ds), "tau": taus,
            "ovl_vs_b0": {k: r4(v) for k, v in (comp[ds] if ds in comp else comp).items()},
            "m1_vs_b0": {k: r4(v) for k, v in m1.items()},
            "overlap": {k: r4(sm[k]) for k in ("A_overlap", "B0_overlap", "M1_overlap",
                                               "A_lexical_adv", "B0_lexical_adv",
                                               "M1_lexical_adv", "rho_M1score_overlap")},
            "overlap_hist": overlap_hist(ds),
        }
    for tag in ("R1", "R2"):
        b = load(f"{tag}-replication")["bootstrap"]
        for ds in b["datasets"]:
            t = b["tau_seed_mean"][ds]
            lab["replication"][ds] = {
                "name": NAMES[ds][0], "about": NAMES[ds][1], "round": tag,
                "tau": {m: r4(t[m]["tau_b"]) for m in ("OVL", "B0", "B1")},
                "ovl_vs_b0": {k: r4(v) for k, v in b["comparisons"]["OVL-B0"][ds].items()},
            }
        lab["replication"][f"pooled_{tag}"] = {
            # R2 reused R1's code, so its verdict is stored under "R1_supported" too
            "datasets": b["datasets"],
            "supported": bool(next(v for k, v in b.items() if k.endswith("_supported"))),
            "ovl_vs_b0": {k: r4(v) for k, v in b["comparisons"]["OVL-B0"]["pooled"].items()},
            "ovl_vs_b1": {k: r4(v) for k, v in b["comparisons"]["OVL-B1"]["pooled"].items()},
        }
    meta = json.loads((ROOT / "data" / "processed" / "hiaq" / "meta.json").read_text())
    lab["hiaq_languages"] = meta["lang_class"]
    lab["hiaq_queries"] = meta["queries"]
    main_boot = load("H1-fiqa+cqa_android")["decision"]
    lab["h1_rejected"] = main_boot["rejected"]
    OUT.write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.1f} kB)")
    for ds, d in lab["datasets"].items():
        print(ds, {s: d["ndcg"][s] for s in ("A", "M1")}, {m: d["tau"][m]["tau_b"]
                                                            for m in ("B0", "M1", "OVL")},
              d["ovl_vs_b0"])
    for k, v in lab["replication"].items():
        print(k, v)


if __name__ == "__main__":
    main()
