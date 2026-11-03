"""P4.2 judge pilot + 50-call throughput benchmark (spec §9.2). Engineering check, not a result.

D1 fiqa, split seed 0, the first 12 U queries that have pilot neighbours: M1 (2-3 permutations)
plus M1-iso (1 call) each, i.e. about 50 judge calls. Checks: schema validity, retries and
failures, permutation disagreement (third-run rate), gate pass rates, the leakage invariant
(neighbours come from R_0 only; the passage is the query's own source passage), and
determinism (fresh-state uncached repeat). Output: reports/phase_4_judge_pilot.json and
data/judge/pilot/fiqa_s0.jsonl
"""

from __future__ import annotations

import json
import statistics
import time

import pandas as pd

from ha import llm, registry
from ha.cache import KVCache
from ha.config import load_config
from ha.data.langid import classify
from ha.data.splits import load_split
from ha.gen.generate import load_source_corpus
from ha.judge.aggregate import passes_gate
from ha.judge.prompt import judge_messages
from ha.judge.run import judge_call, judge_iso, judge_m1
from ha.neighbours.retrieve import load_U, out_path
from ha.paths import REPO_ROOT, get_paths
from ha.utils import git_sha, utc_now_iso

DS, SEED, N = "fiqa", 0, 12


def main() -> None:
    cfg = load_config("judge.yaml")
    llm.require_digest(cfg["model"]["tag"], cfg["model"]["digest"])
    run_id = registry.new_run_id("P4-judge-pilot")
    domain = load_config(f"datasets/{DS}.yaml")["domain_description"]
    nb = pd.read_parquet(out_path(DS, SEED, "U_pilot")).head(N)
    U = load_U(DS).set_index("qid")
    R = load_split(DS, SEED, "R").set_index("qid")               # R only
    passages = load_source_corpus(DS).set_index("pid")["text"]
    cache = KVCache(get_paths().cache / "judge.sqlite")
    llm.unload_all()                                             # fresh model state
    rows, times = [], []
    for r in nb.itertuples():
        assert set(r.neighbour_qids) <= set(R.index), "neighbour outside R_0"
        q, src = U.loc[r.qid, "text"], U.loc[r.qid, "source_pid"]
        lang = classify(q)["label"]
        neigh = [R.loc[x, "text"] for x in r.neighbour_qids]
        t = time.perf_counter()
        m1 = judge_m1(r.qid, q, passages.loc[src], neigh, lang, domain, cache)
        iso = judge_iso(r.qid, q, passages.loc[src], lang, domain, cache)
        n_calls = sum(p["attempts"] for p in m1["perms"]) + iso["perms"][0]["attempts"]
        times.append((time.perf_counter() - t) / n_calls)
        rows.append({"qid": r.qid, "pool": U.loc[r.qid, "pool"], "query": q, "lang_class": lang,
                     "m1": m1, "iso": iso, "n_calls": n_calls})
    # determinism: fresh model state, cache bypassed, the first 4 M1 perm-0 prompts
    llm.unload_all()
    det = []
    for row in rows[:4]:
        q = row["query"]
        src = U.loc[row["qid"], "source_pid"]
        neigh = [R.loc[x, "text"] for x in nb.set_index("qid").loc[row["qid"], "neighbour_qids"]]
        msgs = judge_messages(row["qid"], q, passages.loc[src], neigh, row["lang_class"], domain, 0)
        again = judge_call(msgs, None)
        det.append(again.get("output") == row["m1"]["perms"][0].get("output"))
    llm.unload_all()
    calls = [p for x in rows for p in x["m1"]["perms"] + x["iso"]["perms"]]
    summary = {
        "run_id": run_id, "git_sha": git_sha(), "created_at": utc_now_iso(),
        "label": "ENGINEERING pilot (not a result)", "dataset": DS, "split_seed": SEED,
        "queries": len(rows), "judge_calls": sum(x["n_calls"] for x in rows),
        "json_valid_first_try": sum(1 for p in calls if p["ok"] and p["attempts"] == 1),
        "retries": sum(1 for p in calls if p["attempts"] == 2),
        "failed_calls": sum(1 for p in calls if not p["ok"]),
        "m1_ok": sum(x["m1"]["ok"] for x in rows), "iso_ok": sum(x["iso"]["ok"] for x in rows),
        "third_run_rate": sum(x["m1"]["third_run"] for x in rows) / len(rows),
        "m1_gate_pass": sum(1 for x in rows if x["m1"]["ok"] and passes_gate(x["m1"]["dims"])),
        "iso_gate_pass": sum(1 for x in rows if x["iso"]["ok"] and passes_gate(x["iso"]["dims"])),
        "sec_per_call_median": round(statistics.median(times), 2),
        "sec_per_call_mean": round(statistics.mean(times), 2),
        "prompt_tokens_median": statistics.median(p["prompt_tokens"] for p in calls
                                                  if p.get("prompt_tokens")),
        "deterministic_fresh_uncached_repeat": f"{sum(det)}/{len(det)}",
        "vram_note": "judge alone on GPU (neighbour reranker unloaded before)",
    }
    out_dir = get_paths().data / "judge" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{DS}_s{SEED}.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
    (REPO_ROOT / "reports" / "phase_4_judge_pilot.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for x in rows:
        m, i = x["m1"]["dims"], x["iso"]["dims"]
        print(f"[{x['pool']:9s}] M1 {m}  ISO {i}  | {x['query'][:70]}")


if __name__ == "__main__":
    main()
