"""Run C1-C7 over a query set, incrementally, and the E0 sanity experiment (plan P2).

Query sets:
  auth   all authentic queries (R ∪ A). For held-out datasets this includes sealed A, so it is
         only reachable after the freeze (prereg §3.3); D2 gets corpus-side work only (D-039).
  smoke  fixed made-up queries, pipeline check only; never scored.
  U      the synthetic pool U = P-std ∪ P-persona (P5.3). It uses only the corpus, so it runs
         before the freeze on held-out datasets too (prereg §3.3); scored against each
         query's source passage, grade 1 (§5.4).
GPU rule: the Ollama stages (C3 and C4 via C3) run first; Ollama is unloaded before the
reranker (C5-C7) loads.
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

import pandas as pd
from loguru import logger

from ha import registry
from ha.config import config_hash
from ha.data.public import dataset_config, processed_dir
from ha.eval.metrics import per_query
from ha.paths import get_paths
from ha.seal import require_unsealed
from ha.systems import retrieval as R
from ha.utils import set_seed, utc_now_iso

SMOKE_QUERIES = [
    "how do i check my balance", "money deducted but payment failed", "update mobile number",
    "how long does a refund take", "forgot password cannot login", "phone not charging",
    "app keeps crashing after update", "tax on savings interest", "transfer to new account",
    "cancel a request i made by mistake",
]


def query_set(ds: str, name: str) -> pd.DataFrame:
    if name == "auth":
        if dataset_config(ds)["role"] == "heldout":
            require_unsealed("heldout_A", what="authentic queries incl. A (run/score)", dataset=ds)
        return pd.read_parquet(processed_dir(ds) / "queries_auth.parquet")[["qid", "text"]]
    if name == "smoke":
        return pd.DataFrame({"qid": [f"smoke-{i}" for i in range(len(SMOKE_QUERIES))],
                             "text": SMOKE_QUERIES})
    if name == "U":
        from ha.neighbours.retrieve import load_U

        return load_U(ds)[["qid", "text"]]
    raise ValueError(f"unknown query set {name!r}")


def run_path(ds: str, cfg_id: str, qset: str):
    return get_paths().runs / ds / cfg_id / f"{qset}.run.parquet"


def _log_resources(ds: str, rec: dict[str, Any]) -> None:
    path = get_paths().runs / ds / "resources.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"logged_at": utc_now_iso(), **rec}) + "\n")


def _save(ds: str, cfg_id: str, qset: str, run: pd.DataFrame, meta: dict) -> None:
    p = run_path(ds, cfg_id, qset)
    p.parent.mkdir(parents=True, exist_ok=True)
    run.to_parquet(p, index=False)
    p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def _fresh(ds: str, cfg_id: str, qset: str, meta: dict) -> bool:
    m = run_path(ds, cfg_id, qset).with_suffix(".meta.json")
    return m.exists() and json.loads(m.read_text()) == meta


def run_configs(ds: str, qset: str, configs: list[str] | None = None) -> dict[str, pd.DataFrame]:
    scfg = R.systems_config()
    depth, rdepth = scfg["retrieval_depth"], scfg["rerank"]["depth"]
    queries = query_set(ds, qset)
    qids, texts = queries["qid"].tolist(), queries["text"].tolist()
    qhash = config_hash(sorted(zip(qids, texts, strict=True)))
    wanted = configs or list(scfg["configs"])
    runs: dict[str, pd.DataFrame] = {}

    def meta(cid: str) -> dict:
        return {"config": scfg["configs"][cid], "systems_hash": config_hash(scfg),
                "queries_hash": qhash, "n_queries": len(qids)}

    def cached(cid: str) -> pd.DataFrame | None:
        if _fresh(ds, cid, qset, meta(cid)):
            return pd.read_parquet(run_path(ds, cid, qset))
        return None

    order = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    needed = set(wanted)
    for cid in order:  # add dependencies
        c = scfg["configs"][cid]
        if cid in needed:
            needed |= set(c.get("inputs", [])) | ({c["input"]} if "input" in c else set())
    reranker = None
    for cid in [c for c in order if c in needed]:
        hit = cached(cid)
        if hit is not None:
            runs[cid] = hit
            continue
        c, t0 = scfg["configs"][cid], time.perf_counter()
        if c["type"] == "bm25":
            run = R.BM25Index(ds, c["k1"], c["b"]).search(qids, texts, depth)
        elif c["type"] == "dense":
            run = R.DenseIndex(ds).search(qids, texts, depth)
        elif c["type"] == "rrf":
            run = R.rrf([runs[i] for i in c["inputs"]], c["k"], depth)
        else:
            if reranker is None:
                reranker = R.Reranker()
                corpus = R.load_corpus(ds)
                ptext = dict(zip(corpus["pid"], R.passage_texts(corpus), strict=True))
                qtext = dict(zip(qids, texts, strict=True))
            run = R.rerank_run(runs[c["input"]], qtext, ptext, reranker.score_pairs, rdepth)
        runs[cid] = run
        _save(ds, cid, qset, run, meta(cid))
        rec = {"stage": "run", "dataset": ds, "config": cid, "query_set": qset,
               "n_queries": len(qids), "seconds": round(time.perf_counter() - t0, 2)}
        if c["type"] == "rerank":
            import torch
            rec["torch_peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        _log_resources(ds, rec)
        logger.info(f"{ds}/{qset}/{cid}: {rec['seconds']} s")
    return {k: v for k, v in runs.items() if k in wanted}


def measure_latency(ds: str) -> dict[str, float]:
    """Median of 3 timed passes over a fixed 200-query sample, caches disabled, ms/query.
    Ollama (query embedding) and torch (reranker) stages are timed separately (VRAM rule)
    and summed for C5-C7."""
    scfg = R.systems_config()
    lat, depth = scfg["latency"], scfg["retrieval_depth"]
    q = query_set(ds, "auth").sample(lat["sample_size"], random_state=lat["sample_seed"])
    qids, texts = q["qid"].tolist(), q["text"].tolist()
    n = len(qids)

    def timed(fn) -> float:
        times = []
        for _ in range(lat["repeats"]):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000 / n)
        return statistics.median(times)

    out: dict[str, float] = {}
    bm = {cid: R.BM25Index(ds, c["k1"], c["b"]) for cid, c in scfg["configs"].items()
          if c["type"] == "bm25"}
    for cid, idx in bm.items():
        out[cid] = timed(lambda idx=idx: idx.search(qids, texts, depth))
    dense = R.DenseIndex(ds)
    out["C3"] = timed(lambda: dense.search(qids, texts, depth, use_cache=False))
    r1, r3 = bm["C1"].search(qids, texts, depth), dense.search(qids, texts, depth)
    out["C4"] = out["C1"] + out["C3"] + timed(lambda: R.rrf([r1, r3], 60, depth))
    r4 = R.rrf([r1, r3], 60, depth)
    reranker = R.Reranker(use_cache=False)
    corpus = R.load_corpus(ds)
    ptext = dict(zip(corpus["pid"], R.passage_texts(corpus), strict=True))
    qtext = dict(zip(qids, texts, strict=True))
    rd = scfg["rerank"]["depth"]
    for cid, first, base in [("C5", r3, out["C3"]), ("C6", r1, out["C1"]), ("C7", r4, out["C4"])]:
        out[cid] = base + timed(lambda f=first: R.rerank_run(f, qtext, ptext,
                                                             reranker.score_pairs, rd))
    return {k: round(v, 2) for k, v in out.items()}


def run_e0(ds: str = "fiqa", with_latency: bool = True) -> dict[str, Any]:
    """E0 sanity (D1 only): metrics of C1-C7 on all authentic queries vs published references."""
    if dataset_config(ds)["role"] != "dev":
        raise RuntimeError("E0 runs on the development dataset only (D-022)")
    set_seed(0)
    scfg = R.systems_config()
    runs = run_configs(ds, "auth")
    qrels = pd.read_parquet(processed_dir(ds) / "qrels_auth.parquet")
    run_id = registry.new_run_id("E0")
    rows, summary = [], {}
    for cid, run in runs.items():
        pq = per_query(run, qrels, scfg["metrics"])
        pq.insert(0, "query_set", "auth")
        pq.to_parquet(run_path(ds, cid, "auth").with_name("auth.perquery.parquet"), index=False)
        summary[cid] = {m: round(float(pq[m].mean()), 4) for m in scfg["metrics"]}
        for m in scfg["metrics"]:
            rows.append({"run_id": run_id, "exp": "E0", "dataset": dataset_config(ds)["id"],
                         "split_seed": None, "method": "authentic_all", "system_set": cid,
                         "metric": m, "value": float(pq[m].mean()), "ci_low": None,
                         "ci_high": None, "n_queries": len(pq)})
    checks = {}
    for cid, ref in scfg["e0_reference"].get(ds, {}).items():
        got = summary[cid][ref["metric"]]
        ok = abs(got - ref["value"]) <= ref["tol"]
        checks[cid] = {"got": got, "reference": ref["value"], "tol": ref["tol"],
                       "within": ok, "gate": ref["gate"], "source": ref["source"]}
    latency = measure_latency(ds) if with_latency else {}
    for cid, ms in latency.items():
        rows.append({"run_id": run_id, "exp": "E0", "dataset": dataset_config(ds)["id"],
                     "split_seed": None, "method": "latency_sample200", "system_set": cid,
                     "metric": "latency_ms_per_query", "value": float(ms), "ci_low": None,
                     "ci_high": None, "n_queries": scfg["latency"]["sample_size"]})
    registry.record(rows, config=scfg)
    result = {"run_id": run_id, "dataset": ds, "metrics": summary, "checks": checks,
              "latency_ms_per_query": latency,
              "gate_passed": all(v["within"] for v in checks.values() if v["gate"])}
    out = get_paths().results / "E0.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
