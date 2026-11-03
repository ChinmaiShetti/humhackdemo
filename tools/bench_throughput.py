"""P0.3 mini throughput benchmark (engineering measurement, not a research result).

10 generation calls + 20 judge calls (10 queries x 2 neighbour permutations) on qwen3:4b,
200+200 bge-m3 embeddings, and bge-reranker pair throughput. Uses FiQA passages and FiQA
*train-split* queries as stand-in neighbours, so the 648 test queries are never touched.
Output: reports/phase_0_benchmark.json. Run: .venv\\Scripts\\python tools\\bench_throughput.py
"""

from __future__ import annotations

import json
import random
import statistics
import time

import numpy as np
import pandas as pd

from ha import llm
from ha.judge.schema import JudgeOutput
from ha.models import reranker_entry
from ha.paths import REPO_ROOT, get_paths
from ha.prompts import render
from ha.utils import set_seed, utc_now_iso

MODEL = "qwen3:4b"
EMBED = "bge-m3"
SEED = 0


def _stats(xs: list[float]) -> dict[str, float]:
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3),
            "median": round(statistics.median(xs), 3), "max": round(max(xs), 3)}


def _chat(messages, fmt=None, options=None) -> tuple[str, dict]:
    t0 = time.perf_counter()
    r = llm.client().chat(model=MODEL, messages=messages, format=fmt, options=options,
                          think=False)
    wall = time.perf_counter() - t0
    return r.message.content, {
        "wall_s": wall,
        "prompt_tokens": r.prompt_eval_count,
        "output_tokens": r.eval_count,
        "prompt_tok_per_s": r.prompt_eval_count / (r.prompt_eval_duration / 1e9)
        if r.prompt_eval_duration else None,
        "gen_tok_per_s": r.eval_count / (r.eval_duration / 1e9) if r.eval_duration else None,
        "load_s": (r.load_duration or 0) / 1e9,
    }


def main() -> None:
    set_seed(SEED)
    rng = random.Random(SEED)
    proc = get_paths().processed / "fiqa"
    corpus = pd.read_parquet(proc / "corpus.parquet")
    test_qids = set(pd.read_parquet(proc / "queries_auth.parquet")["qid"])
    raw_q = pd.read_parquet(next((get_paths().data / "raw" / "fiqa").rglob("queries-*.parquet")))
    train_queries = raw_q[~raw_q["_id"].astype(str).isin(test_qids)]["text"].tolist()

    lens = corpus["text"].str.split().str.len()
    passages = corpus[(lens >= 60) & (lens <= 250)].sample(10, random_state=SEED)["text"].tolist()
    out: dict = {"started_at": utc_now_iso(), "model": MODEL,
                 "digest": llm.model_digest(MODEL), "ollama": llm.server_version()}

    # ---- warm-up (cold load measured separately)
    _, cold = _chat([{"role": "user", "content": "Say ok."}], options={"num_predict": 4})
    out["cold_load_s"] = round(cold["load_s"], 2)

    # ---- generation: 10 calls, spec §7 decoding
    gen_rows, queries = [], []
    for i, p in enumerate(passages):
        text, m = _chat([{"role": "user", "content": render("gen_std_v1", passage=p)}],
                        options={"temperature": 0.7, "top_p": 0.9, "num_predict": 64,
                                 "seed": SEED + i})
        queries.append(text.strip().strip('"'))
        gen_rows.append(m)
    ps = llm.client().ps().models
    out["vram_qwen_gib"] = round(ps[0].size_vram / 2**30, 2) if ps else None
    out["generation"] = {k: _stats([r[k] for r in gen_rows])
                         for k in ("wall_s", "prompt_tokens", "output_tokens", "gen_tok_per_s")}
    out["sample_generated_queries"] = queries[:3]

    # ---- judge: 10 queries x 2 permutations = 20 calls, spec §9.2 decoding
    judge_rows, valid = [], 0
    system = render("judge_v0_system")
    for p, q in zip(passages, queries, strict=True):
        neigh = rng.sample(train_queries, 5)
        for perm in range(2):
            order = neigh[:] if perm == 0 else list(reversed(neigh))
            user = render("judge_v0_user", domain_description="personal finance Q&A",
                          passage=p, neighbours=order, query=q, lang_class="EN")
            text, m = _chat([{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                            fmt=JudgeOutput.model_json_schema(),
                            options={"temperature": 0, "seed": SEED, "num_ctx": 4096})
            try:
                JudgeOutput.model_validate_json(text)
                valid += 1
            except Exception:  # noqa: BLE001
                pass
            judge_rows.append(m)
    out["judge"] = {k: _stats([r[k] for r in judge_rows])
                    for k in ("wall_s", "prompt_tokens", "output_tokens", "prompt_tok_per_s",
                              "gen_tok_per_s")}
    out["judge"]["json_valid_rate"] = valid / len(judge_rows)

    # ---- bge-m3 embeddings via Ollama
    emb_passages = corpus["text"].sample(200, random_state=SEED).tolist()
    emb_queries = rng.sample(train_queries, 200)
    for name, texts in [("passages", emb_passages), ("queries", emb_queries)]:
        t0 = time.perf_counter()
        for b in range(0, len(texts), 32):
            np.asarray(llm.client().embed(model=EMBED, input=texts[b:b + 32]).embeddings)
        dt = time.perf_counter() - t0
        out[f"embed_{name}_per_s"] = round(len(texts) / dt, 1)

    # ---- reranker (torch): VRAM rule, unload Ollama first
    llm.unload_all()
    import torch
    from FlagEmbedding import FlagReranker

    torch.cuda.reset_peak_memory_stats()
    rr = FlagReranker(reranker_entry()["local_path"], use_fp16=True, devices="cuda:0")
    qp = [[rng.choice(emb_queries), p] for p in emb_passages] * 2          # 400 query-passage
    qq = [[rng.choice(emb_queries), rng.choice(train_queries)] for _ in range(400)]  # query-query
    rr.compute_score(qp[:16], max_length=512)  # warm-up
    for name, pairs, max_len in [("query_passage", qp, 512), ("query_query", qq, 128)]:
        t0 = time.perf_counter()
        rr.compute_score(pairs, batch_size=32, max_length=max_len)
        out[f"rerank_{name}_pairs_per_s"] = round(len(pairs) / (time.perf_counter() - t0), 1)
    out["rerank_peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out["finished_at"] = utc_now_iso()

    path = REPO_ROOT / "reports" / "phase_0_benchmark.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
