"""Bulk judging, P5.1 (prereg §5.2 as amended by A3; spec §9; configs/judge.yaml, unchanged).

For one dataset, over U = P-std ∪ P-persona (D-048):
  1. M1-iso: one call per U query, with no neighbour block. The judge never sees R, so the
     result does not depend on the split seed and is computed once.
  2. M1: for every split seed s, 2-3 permutations with the top-5 authentic neighbours from R_s
     (data/neighbours/<ds>/s<s>/U.parquet). Queries flagged near_duplicate for seed s are not
     judged for that seed, because prereg §5.2 excludes them from every set.
The passage is the query's own source passage. A is never loaded (`load_split(..., "R")`).

The judge is prompt v0 with the frozen decoding. Amendment A3 fixes v0 as the final prompt
(no human calibration), so run_bulk refuses to start until A3 is recorded in the prereg.

Resumable. Each stage appends one JSON line per query to <stage>.jsonl and is then finalised to
<stage>.parquet + .meta.json. A rerun skips every query with a recorded outcome, but retries
transport failures (for example Ollama down); five in a row stop the run. Every stage starts
from a fresh model state (D-046) and visits queries in the fixed U order, so a --limit smoke
run is an exact prefix of the full run. Output: HA_DATA/judge/<ds>/; --limit writes to
HA_DATA/judge_smoke/<ds>/ and never to the registry.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ha.judge.aggregate import DIMS, passes_gate

AMENDMENT = "A3"
ISO, M1 = "M1-iso", "M1"
FULL_SEEDS = (0, 1, 2, 3, 4)
MAX_TRANSPORT_STREAK = 5
LOG_EVERY = 50


class BulkBlocked(RuntimeError):
    pass


def out_dir(ds: str, smoke: bool = False) -> Path:
    from ha.paths import get_paths

    return get_paths().data / ("judge_smoke" if smoke else "judge") / ds


def stage_log(ds: str, seed: int | None, smoke: bool = False) -> Path:
    """<ds>/M1-iso.jsonl for M1-iso (seed None), <ds>/s<seed>/M1.jsonl for M1."""
    d = out_dir(ds, smoke)
    return d / f"{ISO}.jsonl" if seed is None else d / f"s{seed}" / f"{M1}.jsonl"


def flatten(res: dict) -> dict[str, Any]:
    """One judge_m1 / judge_iso result -> a flat record. Dims are the permutation medians;
    keep and reason come from permutation 0 (informational, not used by selection)."""
    perms = res.get("perms", [])
    dims = res.get("dims") or {}
    first = (perms[0].get("output") if perms else None) or {}
    return {
        "ok": bool(res["ok"]),
        **{d: dims.get(d) for d in DIMS},
        "gate_pass": bool(res["ok"] and passes_gate(dims)),
        "keep": first.get("keep"),
        "reason": first.get("reason"),
        "third_run": bool(res.get("third_run")),
        "n_calls": sum(p.get("attempts", 0) for p in perms),
        "retries": sum(1 for p in perms if p.get("attempts", 0) == 2),
        "cached_calls": sum(sum(p.get("cached", [])) for p in perms),
        "prompt_tokens_max": max((p.get("prompt_tokens") or 0 for p in perms), default=0),
        "error": res.get("error"),
        "perm_outputs": json.dumps([p.get("output") for p in perms], ensure_ascii=False),
    }


def is_transport_failure(rec: dict) -> bool:
    """A failure that is not a schema-validation failure: retried on the next run."""
    return not rec["ok"] and not str(rec.get("error") or "").startswith("validation")


def read_log(path: Path) -> list[dict]:
    """All complete records; a torn last line (crash mid-write) is ignored."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def run_stage(items: list[dict], path: Path, judge_fn: Callable[[dict], dict], label: str,
              fresh_state: Callable[[], Any]) -> dict[str, Any]:
    """Judge every item not yet recorded in `path`, appending one line each.

    items: {"qid", "meta" (extra record fields), ...judge inputs}, in the fixed order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["qid"] for r in read_log(path) if not is_transport_failure(r)}
    todo = [it for it in items if it["qid"] not in done]
    t0, n, streak = time.perf_counter(), 0, 0
    if todo:
        fresh_state()                                  # D-046: every stage from a fresh state
        from ha.utils import utc_now_iso

        with open(path, "a", encoding="utf-8") as f:
            for it in todo:
                rec = {"qid": it["qid"], **it["meta"], **flatten(judge_fn(it)),
                       "timestamp": utc_now_iso()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                streak = streak + 1 if is_transport_failure(rec) else 0
                if streak >= MAX_TRANSPORT_STREAK:
                    raise RuntimeError(f"{label}: {streak} transport failures in a row, "
                                       f"stopped; rerun to resume. Last: {rec['error']}")
                if n % LOG_EVERY == 0:
                    per_q = (time.perf_counter() - t0) / n
                    logger.info(f"{label}: {len(items) - len(todo) + n}/{len(items)} "
                                f"({per_q:.1f} s/query, ~{per_q * (len(todo) - n) / 3600:.1f} h "
                                "left in stage)")
    return {"resumed_from": len(items) - len(todo), "new": n,
            "seconds": round(time.perf_counter() - t0, 1)}


def finalize_stage(path: Path, order: list[str], meta: dict) -> dict[str, Any]:
    """Last record per qid, in the fixed order -> <stage>.parquet; stats -> <stage>.meta.json."""
    recs = pd.DataFrame(read_log(path))
    if len(recs):
        pos = {q: i for i, q in enumerate(order)}
        recs = recs[recs["qid"].isin(pos)].drop_duplicates("qid", keep="last")
        recs = recs.sort_values("qid", key=lambda c: c.map(pos)).reset_index(drop=True)
    n = len(recs)
    ok = int(recs["ok"].sum()) if n else 0
    stats = {
        "n_expected": len(order), "n_recorded": n, "complete": n == len(order),
        "n_ok": ok, "n_failed": n - ok,
        "failure_rate": (n - ok) / n if n else None,
        "transport_failures": int(recs.apply(is_transport_failure, axis=1).sum()) if n else 0,
        "n_gate_pass": int(recs["gate_pass"].sum()) if n else 0,
        "gate_pass_rate": float(recs["gate_pass"].mean()) if n else None,
        "third_run_rate": float(recs["third_run"].mean()) if n else None,
        "n_calls": int(recs["n_calls"].sum()) if n else 0,
        "retries": int(recs["retries"].sum()) if n else 0,
        "cached_calls": int(recs["cached_calls"].sum()) if n else 0,
        "prompt_tokens_max": int(recs["prompt_tokens_max"].max()) if n else 0,
    }
    recs.to_parquet(path.with_suffix(".parquet"), index=False)
    full = {**meta, **stats}
    path.with_suffix(".meta.json").write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    return full


def finish_stage(path: Path, order: list[str], meta: dict, run: dict) -> dict[str, Any]:
    """Finalise a stage, unless this call judged nothing new in a stage that was already final:
    then the record written by the run that judged it (timings, counts) is kept as it is."""
    from ha.utils import utc_now_iso

    old = path.with_suffix(".meta.json")
    if run["new"] == 0 and old.exists():
        prev = json.loads(old.read_text(encoding="utf-8"))
        if prev.get("complete") and prev.get("n_recorded") == len(order):
            return {**prev, "new": 0}
    return finalize_stage(path, order, {**meta, **run, "finished_at": utc_now_iso()})


def _registry_rows(ds: str, run_id: str, stages: list[dict]) -> list[dict]:
    rows = []
    for st in stages:
        metrics = ["failure_rate", "gate_pass_rate"] + (["third_run_rate"] if st["method"] == M1
                                                        else [])
        for m in metrics:
            rows.append({"run_id": run_id, "exp": "P5.1-judge", "dataset": ds,
                         "split_seed": st["split_seed"], "method": st["method"],
                         "system_set": None, "metric": f"judge_{m}", "value": float(st[m]),
                         "ci_low": None, "ci_high": None, "n_queries": st["n_recorded"],
                         "label": "judge bookkeeping (prompt v0, A3)"})
    return rows


def run_bulk(ds: str, seeds: tuple[int, ...] = FULL_SEEDS, limit: int | None = None,
             iso: bool = True) -> dict[str, Any]:
    from ha import llm, registry
    from ha.cache import KVCache
    from ha.config import domain_description, load_config
    from ha.data.langid import classify
    from ha.data.splits import load_split
    from ha.gen.generate import load_source_corpus
    from ha.judge.calibrate import amendment_recorded
    from ha.judge.run import judge_iso, judge_m1
    from ha.neighbours.retrieve import TOP_K, load_U, out_path
    from ha.paths import get_paths
    from ha.utils import git_sha, utc_now_iso

    if not amendment_recorded(AMENDMENT):
        raise BulkBlocked(f"bulk judging uses prompt v0 as final, which amendment {AMENDMENT} "
                          "fixes; it is not recorded under '## Amendments' in the prereg")
    cfg = load_config("judge.yaml")
    digest = llm.require_digest(cfg["model"]["tag"], cfg["model"]["digest"])
    smoke = limit is not None
    U = load_U(ds)
    if smoke:
        U = U.head(limit)
    order = U["qid"].tolist()
    passages = load_source_corpus(ds).set_index("pid")["text"]
    missing = set(U["source_pid"]) - set(passages.index)
    assert not missing, f"{len(missing)} source passages missing"
    domain = domain_description(ds)
    base = {"dataset": ds, "amendment": AMENDMENT, "prompt_version": cfg["prompt"]["version"],
            "model": cfg["model"]["tag"], "model_digest": digest, "decoding": cfg["decoding"],
            "git_sha": git_sha(), "smoke": smoke}
    inputs = [{"qid": r.qid, "query": r.text, "passage": passages.loc[r.source_pid],
               "lang": classify(r.text)["label"], "pool": r.pool} for r in U.itertuples()]
    cache = KVCache(get_paths().cache / "judge.sqlite")
    started, stages = utc_now_iso(), []
    try:
        if iso:
            path = stage_log(ds, None, smoke)
            items = [{**x, "meta": {"pool": x["pool"], "lang_class": x["lang"]}} for x in inputs]
            run = run_stage(items, path, lambda it: judge_iso(
                it["qid"], it["query"], it["passage"], it["lang"], domain, cache),
                f"{ds}/{ISO}", llm.unload_all)
            stages.append(finish_stage(path, order, {**base, "method": ISO, "split_seed": None},
                                       run))
        for s in seeds:
            nb = pd.read_parquet(out_path(ds, s, "U")).set_index("qid")
            assert set(order) <= set(nb.index), f"neighbours for seed {s} do not cover U"
            r_text = load_split(ds, s, "R").set_index("qid")["text"]          # R only, never A
            near = {q for q in order if bool(nb.loc[q, "near_duplicate"])}
            items = []
            for x in inputs:
                if x["qid"] in near:
                    continue
                nq = list(nb.loc[x["qid"], "neighbour_qids"])
                assert len(nq) == TOP_K and set(nq) <= set(r_text.index), \
                    f"{x['qid']}: neighbours must be {TOP_K} queries from R_{s}"
                items.append({**x, "neighbours": [r_text.loc[q] for q in nq],
                              "meta": {"pool": x["pool"], "lang_class": x["lang"],
                                       "split_seed": s, "neighbour_qids": nq}})
            path = stage_log(ds, s, smoke)
            run = run_stage(items, path, lambda it: judge_m1(
                it["qid"], it["query"], it["passage"], it["neighbours"], it["lang"], domain,
                cache), f"{ds}/s{s}/{M1}", llm.unload_all)
            stages.append(finish_stage(path, [q for q in order if q not in near],
                                       {**base, "method": M1, "split_seed": s,
                                        "near_duplicate_skipped": len(near)}, run))
    finally:
        cache.close()
    llm.unload_all()
    summary_path = out_dir(ds, smoke) / "summary.json"
    if not any(st["new"] for st in stages) and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))   # nothing new: keep record
    summary = {**base, "started_at": started, "finished_at": utc_now_iso(),
               "stages": [{k: st[k] for k in ("method", "split_seed", "n_expected", "n_recorded",
                                              "complete", "failure_rate", "gate_pass_rate",
                                              "third_run_rate", "n_calls", "resumed_from",
                                              "seconds")} for st in stages]}
    # Registry: only a full run (M1-iso + seeds 0-4) that completed every stage in this call or
    # finished an interrupted one; a rerun with nothing new to judge records nothing again.
    full = iso and tuple(seeds) == FULL_SEEDS and not smoke
    if full and all(st["complete"] for st in stages) and any(st["new"] for st in stages):
        run_id = registry.new_run_id(f"P5.1-judge-{ds}")
        registry.record(_registry_rows(ds, run_id, stages), {"judge": cfg, "amendment": AMENDMENT})
        summary["registry_run_id"] = run_id
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
