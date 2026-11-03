"""Per-query retrieval metrics via ranx (spec §13.1; prereg §1: ranx `ndcg@10`, linear gain)."""

from __future__ import annotations

import pandas as pd
from ranx import Qrels, Run, evaluate


def _qrels_dict(qrels: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for q, p, g in qrels[["qid", "pid", "grade"]].itertuples(index=False):
        out.setdefault(str(q), {})[str(p)] = int(g)
    return out


def _run_dict(run: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for q, p, s in run[["qid", "pid", "score"]].itertuples(index=False):
        out.setdefault(str(q), {})[str(p)] = float(s)
    return out


def per_query(run: pd.DataFrame, qrels: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """One row per query that has qrels; queries missing from the run score 0."""
    qd = _qrels_dict(qrels)
    rd = _run_dict(run)
    for q in qd:
        rd.setdefault(q, {})
    rd = {q: rd[q] for q in qd}
    r = Run(rd)
    evaluate(Qrels(qd), r, metrics, return_mean=True, make_comparable=True)
    df = pd.DataFrame({m: pd.Series(r.scores[m]) for m in metrics})
    df.index.name = "qid"
    return df.reset_index()
