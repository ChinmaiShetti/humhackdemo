"""Authentic-neighbour retrieval, P4.1 (spec §8; prereg §5.2 and §4.1; D-003).

For every query (synthetic pool queries, or authentic calibration items) and split seed s:
  1. bge-m3 dense cosine against R_s only (A is never loaded; `load_split(..., "R")`);
  2. top-30 by cosine -> bge-reranker-v2-m3 at max_length 128 -> top-5 by rerank score;
  3. features: nn_max_cos (highest cosine over R_s), nn_mean5_cos (mean cosine of the final
     top-5), nn_lang_mix (language classes of the top-5), near_duplicate = nn_max_cos > 0.97.
Authentic queries never get themselves, or any R item with cosine > 0.97 to them, as a
neighbour. Output: HA_DATA/neighbours/<ds>/s<seed>/<set>.parquet (+ .meta.json)
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ha.config import load_config
from ha.paths import get_paths
from ha.utils import git_sha, utc_now_iso

CANDIDATES, TOP_K, NEAR_DUP_COS, RERANK_MAX_LEN = 30, 5, 0.97, 128


def _cfg() -> dict[str, Any]:
    n = load_config("judge.yaml")["neighbours"]
    assert (n["candidates"], n["k"], n["near_duplicate_cos"]) == (CANDIDATES, TOP_K, NEAR_DUP_COS)
    return n


def out_path(ds: str, seed: int, name: str) -> Path:
    return get_paths().data / "neighbours" / ds / f"s{seed}" / f"{name}.parquet"


def load_U(ds: str) -> pd.DataFrame:
    """U = P-std ∪ P-persona as retained in P3 (D-048)."""
    frames = []
    for pool in ("P-std", "P-persona"):
        q = pd.read_parquet(get_paths().data / "synthetic" / ds / pool / "queries.parquet")
        frames.append(q[["qid", "text", "source_pid", "pool"]])
    return pd.concat(frames, ignore_index=True)


def neighbours_for(queries: pd.DataFrame, r: pd.DataFrame, q_emb: np.ndarray, r_emb: np.ndarray,
                   rerank_fn, authentic: bool = False) -> pd.DataFrame:
    """Pure function (testable). queries: qid, text. r: qid, text, lang_class (R_s only)."""
    cos = q_emb @ r_emb.T
    r_qids = r["qid"].astype(str).to_numpy()
    if authentic:  # D-003: never self, never a near-duplicate of self
        self_mask = r_qids[None, :] == queries["qid"].astype(str).to_numpy()[:, None]
        cos = np.where(self_mask | (cos > NEAR_DUP_COS), -np.inf, cos)
    order = np.lexsort((np.broadcast_to(r_qids, cos.shape), -cos), axis=1)[:, :CANDIDATES]
    pairs, owners = [], []
    for qi, row in enumerate(order):
        for j in row:
            if np.isfinite(cos[qi, j]):
                pairs.append((queries["text"].iloc[qi], r_qids[j], r["text"].iloc[j]))
                owners.append((qi, j))
    scores = rerank_fn(pairs) if pairs else []
    by_q: dict[int, list[tuple[float, float, str, int]]] = {}
    for (qi, j), s in zip(owners, scores, strict=True):
        by_q.setdefault(qi, []).append((float(s), float(cos[qi, j]), r_qids[j], j))
    lang = r["lang_class"].to_numpy() if "lang_class" in r else np.array(["EN"] * len(r))
    rows = []
    for qi in range(len(queries)):
        cands = sorted(by_q.get(qi, []), key=lambda t: (-t[0], -t[1], t[2]))
        top = cands[:TOP_K]
        finite = cos[qi][np.isfinite(cos[qi])]
        nn_max = float(finite.max()) if finite.size else float("nan")
        rows.append({
            "qid": queries["qid"].iloc[qi],
            "candidate_qids": [c[2] for c in cands], "candidate_cos": [c[1] for c in cands],
            "candidate_rerank": [c[0] for c in cands],
            "neighbour_qids": [c[2] for c in top], "neighbour_cos": [c[1] for c in top],
            "neighbour_rerank": [c[0] for c in top],
            "nn_max_cos": nn_max,
            "nn_mean5_cos": float(np.mean([c[1] for c in top])) if top else float("nan"),
            "nn_lang_mix": json.dumps(dict(Counter(lang[c[3]] for c in top)), sort_keys=True),
            "near_duplicate": bool(nn_max > NEAR_DUP_COS),
        })
    return pd.DataFrame(rows)


def build(ds: str, seeds=(0, 1, 2, 3, 4), name: str = "U", queries: pd.DataFrame | None = None,
          authentic: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    """Compute and write neighbours for (ds, seed). Uses only R_s. Returns per-seed meta."""
    from ha import llm
    from ha.data.splits import load_split
    from ha.systems.retrieval import Reranker, embed_queries

    _cfg()
    queries = load_U(ds) if queries is None else queries
    if limit is not None:
        queries = queries.head(limit)
    rs = {s: load_split(ds, s, "R")[["qid", "text", "lang_class"]] for s in seeds}  # R only
    t0 = time.perf_counter()
    q_emb = embed_queries(queries["text"].tolist())
    r_emb = {s: embed_queries(rs[s]["text"].tolist()) for s in seeds}
    t_embed = time.perf_counter() - t0
    llm.unload_all()                                   # VRAM rule before torch
    reranker = Reranker(max_length=RERANK_MAX_LEN)
    metas = []
    for s in seeds:
        t1 = time.perf_counter()
        nb = neighbours_for(queries, rs[s], q_emb, r_emb[s], reranker.score_pairs, authentic)
        assert set(q for lst in nb["neighbour_qids"] for q in lst) <= set(rs[s]["qid"]), \
            "neighbour outside R_s"
        path = out_path(ds, s, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        nb.to_parquet(path, index=False)
        meta = {"dataset": ds, "split_seed": s, "set": name, "n_queries": len(nb),
                "n_R": len(rs[s]), "near_duplicates": int(nb["near_duplicate"].sum()),
                "nn_max_cos_median": float(nb["nn_max_cos"].median()),
                "nn_mean5_cos_median": float(nb["nn_mean5_cos"].median()),
                "rerank_max_length": RERANK_MAX_LEN, "candidates": CANDIDATES, "k": TOP_K,
                "embed_seconds_all_seeds": round(t_embed, 1),
                "rerank_seconds": round(time.perf_counter() - t1, 1),
                "git_sha": git_sha(), "created_at": utc_now_iso(), "authentic": authentic}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                  encoding="utf-8")
        metas.append(meta)
    del reranker
    import torch

    torch.cuda.empty_cache()
    return metas
