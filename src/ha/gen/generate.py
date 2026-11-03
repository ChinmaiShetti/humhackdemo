"""Synthetic query generation, P3 (spec §7; prereg-v1 §5.2; configs/generation.yaml, frozen).

One pool = (dataset, pool_id[, split_seed]). For each of the 800 sampled source passages
(uniform, seed 7), one query is generated with the frozen decoding (T 0.7, top_p 0.9, max 64
tokens, seed 1000 + i) by the digest-pinned generator with think=False. Every call is cached by
(prompt, model digest, params). Every attempt is logged, failures included. A run can resume:
indices already in the log are skipped.

Post-processing (spec §7): strip quotes and numbering; drop < 2 tokens; drop > 60% token
overlap with the passage's first sentence; within-pool dedupe at BGE-M3 cosine > 0.95.

Leakage guards: P-std and P-persona see only the passage. P-fewshot sees only R_s texts
(ha.data.splits.load_split(..., "R")); held-out A is never loaded.

Output: HA_DATA/synthetic/<ds>/<pool_dir>/{generations.jsonl, queries.parquet, stats.json}
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import regex
from loguru import logger

from ha import llm
from ha.cache import KVCache
from ha.config import load_config
from ha.paths import get_paths
from ha.prompts import render, template_hash
from ha.utils import git_sha, utc_now_iso

FEWSHOT_TEMPLATE = "gen_fewshot_v1"   # spec Appendix B.3 (verbatim)
FEWSHOT_K = 5                         # spec §7: 5 exemplars sampled from R
PERSONA_SEED = 11                     # persona choice per passage (D-046)
EXEMPLAR_SEED = 13                    # exemplar draw per (split seed, passage) (D-046)
MIN_TOKENS = 2
MAX_FIRST_SENTENCE_OVERLAP = 0.6
DEDUPE_COS = 0.95
HINGLISH_DATASETS = {"miracl_hi", "hiaq"}   # D3 / D4 (spec B.2)
TOKEN_RE = regex.compile(r"[\p{L}\p{N}]+")
QUOTES = "\"'“”‘’`«»"
NUMBERING_RE = re.compile(r"^\s*(?:\(?\d{1,3}[.):]|[-*•–]|Q\d*[.):])\s+")


@dataclass(frozen=True)
class PoolSpec:
    dataset: str
    pool: str                       # P-std | P-persona | P-fewshot
    split_seed: int | None = None   # P-fewshot only

    @property
    def dir_name(self) -> str:
        return self.pool if self.split_seed is None else f"{self.pool}__s{self.split_seed}"

    @property
    def template(self) -> str:
        cfg = load_config("generation.yaml")["pools"]
        return {"P-std": cfg["P-std"]["template"], "P-persona": cfg["P-persona"]["template"],
                "P-fewshot": FEWSHOT_TEMPLATE}[self.pool]


def pool_dir(spec: PoolSpec, pilot: bool = False) -> Path:
    root = "synthetic_pilot" if pilot else "synthetic"
    return get_paths().data / root / spec.dataset / spec.dir_name


# ----------------------------------------------------------------------------- inputs


def load_source_corpus(ds: str) -> pd.DataFrame:
    """(pid, text) for generation. D1/D2 use the retrieval passage text; D4 uses the chunks."""
    if ds == "hiaq":
        c = pd.read_parquet(get_paths().data / "hiaq" / "chunks.parquet")
        return pd.DataFrame({"pid": c["chunk_id"], "text": c["text"]})
    from ha.systems.retrieval import load_corpus, passage_texts

    c = load_corpus(ds)
    return pd.DataFrame({"pid": c["pid"], "text": passage_texts(c)})


def sample_passages(ds: str) -> pd.DataFrame:
    cfg = load_config("generation.yaml")
    corpus = load_source_corpus(ds)
    rng = np.random.default_rng(cfg["source_sample_seed"])
    idx = rng.choice(len(corpus), size=cfg["source_passages_per_dataset"], replace=False)
    out = corpus.iloc[idx].reset_index(drop=True)
    out.insert(0, "i", range(len(out)))
    return out


def personas_for(ds: str) -> list[str]:
    p = load_config("generation.yaml")["pools"]["P-persona"]
    return list(p["personas"]) + ([p["hinglish_persona"]] if ds in HINGLISH_DATASETS else [])


def pick_persona(ds: str, i: int) -> str:
    ps = personas_for(ds)
    return ps[int(np.random.default_rng([PERSONA_SEED, i]).integers(len(ps)))]


def pick_exemplars(r_texts: list[tuple[str, str]], split_seed: int, i: int) -> list[tuple]:
    """k (qid, text) exemplars from R_s only; deterministic per (split seed, passage)."""
    rng = np.random.default_rng([EXEMPLAR_SEED, split_seed, i])
    idx = rng.choice(len(r_texts), size=FEWSHOT_K, replace=False)
    return [r_texts[j] for j in sorted(idx)]


def build_prompt(spec: PoolSpec, passage: str, i: int,
                 r_texts: list[tuple[str, str]] | None = None) -> tuple[str, dict]:
    meta: dict[str, Any] = {}
    if spec.pool == "P-std":
        return render(spec.template, passage=passage), meta
    if spec.pool == "P-persona":
        meta["persona"] = pick_persona(spec.dataset, i)
        return render(spec.template, passage=passage, persona_description=meta["persona"]), meta
    if spec.pool == "P-fewshot":
        if r_texts is None or spec.split_seed is None:
            raise ValueError("P-fewshot needs R_s texts and a split seed")
        ex = pick_exemplars(r_texts, spec.split_seed, i)
        meta["exemplar_qids"] = [q for q, _ in ex]
        return render(spec.template, passage=passage, exemplars=[t for _, t in ex]), meta
    raise ValueError(f"unknown pool {spec.pool}")


# ----------------------------------------------------------------------------- post-processing


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def clean(raw: str) -> str:
    """Strip quotes and numbering (spec §7); keep the first non-empty line; collapse spaces."""
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    text = lines[0] if lines else ""
    for _ in range(3):
        prev = text
        text = NUMBERING_RE.sub("", text).strip().strip(QUOTES).strip()
        if text == prev:
            break
    return re.sub(r"\s+", " ", text)


def first_sentence(passage: str) -> str:
    body = passage.split("\n", 1)[-1] if "\n" in passage else passage  # skip a title line
    parts = re.split(r"(?<=[.!?])\s+", body.strip(), maxsplit=1)
    return parts[0] if parts else body


def first_sentence_overlap(query: str, passage: str) -> float:
    q = tokens(query)
    if not q:
        return 0.0
    fs = set(tokens(first_sentence(passage)))
    return sum(t in fs for t in q) / len(q)


def filter_reason(query: str, passage: str) -> str | None:
    if len(tokens(query)) < MIN_TOKENS:
        return "too_short"
    if first_sentence_overlap(query, passage) > MAX_FIRST_SENTENCE_OVERLAP:
        return "copies_first_sentence"
    return None


def dedupe(vectors: np.ndarray, threshold: float = DEDUPE_COS) -> list[int]:
    """Greedy keep-first: drop a query whose cosine with an already kept one is > threshold."""
    kept: list[int] = []
    for i in range(len(vectors)):
        if kept and float(np.max(vectors[kept] @ vectors[i])) > threshold:
            continue
        kept.append(i)
    return kept


# ----------------------------------------------------------------------------- generation


def _gen_cfg() -> dict[str, Any]:
    return load_config("generation.yaml")


def call_generator(prompt: str, seed: int, cache: KVCache | None, client=None) -> dict:
    """One cached call. Returns {'raw', 'cached', 'prompt_tokens', 'output_tokens'}."""
    cfg = _gen_cfg()
    dec = cfg["decoding"]
    options = {"temperature": dec["temperature"], "top_p": dec["top_p"],
               "num_predict": dec["num_predict"], "seed": seed}
    key = KVCache.key_for({"prompt": prompt, "model": cfg["model"]["digest"],
                           "options": options, "think": cfg["model"]["think"]})
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
    r = (client or llm.client()).chat(model=cfg["model"]["tag"],
                                      messages=[{"role": "user", "content": prompt}],
                                      think=cfg["model"]["think"], options=options)
    out = {"raw": r.message.content, "prompt_tokens": r.prompt_eval_count,
           "output_tokens": r.eval_count}
    if cache is not None and out["raw"] is not None:
        cache.set(key, out)
    return {**out, "cached": False}


def _load_r(spec: PoolSpec) -> list[tuple[str, str]]:
    from ha.data.splits import load_split

    r = load_split(spec.dataset, spec.split_seed, "R")  # R only; A is never requested
    return list(zip(r["qid"].astype(str), r["text"].astype(str), strict=True))


def generate_pool(spec: PoolSpec, limit: int | None = None, client=None,
                  pilot: bool = False) -> dict[str, Any]:
    """Generate (resumably) the pool's raw log. `limit` caps the passages; `pilot` writes to
    data/synthetic_pilot/. Reproducibility (D-046): the model is unloaded first, so every run
    starts from a fresh model state and processes passages in fixed index order. Ollama's KV
    prefix reuse otherwise makes repeated identical prompts sample differently at T 0.7."""
    cfg = _gen_cfg()
    digest = cfg["model"]["digest"] if client is not None else llm.require_digest(
        cfg["model"]["tag"], cfg["model"]["digest"])
    out_dir = pool_dir(spec, pilot)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "generations.jsonl"
    if client is None:
        llm.unload_all()  # fresh model state -> reproducible call sequence (D-046)
    done = set()
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            done = {json.loads(line)["i"] for line in f if line.strip()}
    passages = sample_passages(spec.dataset)
    if limit is not None:
        passages = passages.head(limit)
    r_texts = _load_r(spec) if spec.pool == "P-fewshot" else None
    r_ids = {q for q, _ in r_texts} if r_texts else set()
    cache = KVCache(get_paths().cache / "generation.sqlite")
    ctx_limit = 4096 - cfg["decoding"]["num_predict"]   # Ollama default num_ctx (D-046)
    n_new, t0 = 0, time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as log:
        for row in passages.itertuples():
            if row.i in done:
                continue
            prompt, meta = build_prompt(spec, row.text, row.i, r_texts)
            if spec.pool == "P-fewshot":
                assert set(meta["exemplar_qids"]) <= r_ids, "exemplar outside R_s"
            seed = cfg["decoding"]["seed_base"] + row.i
            rec: dict[str, Any] = {
                "i": int(row.i), "dataset": spec.dataset, "pool": spec.pool,
                "split_seed": spec.split_seed, "source_pid": row.pid, "seed": seed,
                "template": spec.template, "template_hash": template_hash(spec.template),
                "model": cfg["model"]["tag"], "model_digest": digest,
                "decoding": cfg["decoding"], "git_sha": git_sha(), **meta,
            }
            try:
                res = call_generator(prompt, seed, cache, client)
                text = clean(res["raw"])
                rec.update(raw=res["raw"], query=text, cached=res["cached"],
                           prompt_tokens=res["prompt_tokens"],
                           output_tokens=res["output_tokens"],
                           possible_truncation=bool(res["prompt_tokens"]
                                                    and res["prompt_tokens"] >= ctx_limit))
                if not text:
                    rec.update(status="failed", reason="empty_output")
                else:
                    reason = filter_reason(text, row.text)
                    rec.update(status="dropped" if reason else "ok", reason=reason)
            except Exception as e:  # noqa: BLE001 - recorded, never silent
                rec.update(status="failed", reason=f"{type(e).__name__}: {e}"[:300], query="")
            rec["timestamp"] = utc_now_iso()
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            n_new += 1
            if n_new % 100 == 0:
                logger.info(f"{spec.dataset}/{spec.dir_name}: {len(done) + n_new} done "
                            f"({n_new / (time.perf_counter() - t0):.2f}/s)")
    cache.close()
    return {"new": n_new, "seconds": round(time.perf_counter() - t0, 1)}


def finalize_pool(spec: PoolSpec, target: int, embed_fn=None,
                  pilot: bool = False) -> dict[str, Any]:
    """Exact + semantic within-pool dedupe; write queries.parquet and stats.json."""
    if embed_fn is None:
        from ha.systems.retrieval import embed_queries as embed_fn
    out_dir = pool_dir(spec, pilot)
    with open(out_dir / "generations.jsonl", encoding="utf-8") as f:
        log = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    log = log.sort_values("i").drop_duplicates("i", keep="last")
    attempted = len(log)
    failed = int((log["status"] == "failed").sum())
    dropped = log[log["status"] == "dropped"]["reason"].value_counts().to_dict()
    ok = log[log["status"] == "ok"].copy()
    ok["norm"] = ok["query"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    exact_dupes = int(ok.duplicated("norm").sum())
    ok = ok.drop_duplicates("norm", keep="first").reset_index(drop=True)
    vecs = embed_fn(ok["query"].tolist()) if len(ok) else np.zeros((0, 1024), np.float32)
    keep = dedupe(np.asarray(vecs, dtype=np.float32))
    semantic_dupes = len(ok) - len(keep)
    final = ok.iloc[keep].reset_index(drop=True)
    sid = "" if spec.split_seed is None else f"s{spec.split_seed}-"
    final["qid"] = [f"{spec.dataset}-{spec.pool}-{sid}{i:04d}" for i in final["i"]]
    cols = ["qid", "query", "source_pid", "i", "seed", "pool", "split_seed", "dataset",
            "template", "template_hash", "model_digest"] + [
        c for c in ("persona", "exemplar_qids") if c in final.columns]
    final[cols].rename(columns={"query": "text"}).to_parquet(out_dir / "queries.parquet",
                                                             index=False)
    stats = {
        "dataset": spec.dataset, "pool": spec.pool, "split_seed": spec.split_seed,
        "target": target, "attempted": attempted, "failed": failed,
        "failure_rate": round(failed / attempted, 4) if attempted else None,
        "dropped_by_filter": dropped, "exact_duplicates": exact_dupes,
        "semantic_duplicates_cos_gt_0.95": semantic_dupes, "final_unique": len(final),
        "pct_of_target": round(100 * len(final) / target, 1),
        "possible_truncation": int(log.get("possible_truncation", pd.Series(dtype=bool))
                                   .fillna(False).sum()),
        "cached_calls": int(log.get("cached", pd.Series(dtype=bool)).fillna(False).sum()),
        "model": log["model"].iloc[0] if attempted else None,
        "model_digest": log["model_digest"].iloc[0] if attempted else None,
        "decoding": log["decoding"].iloc[0] if attempted else None,
        "seed_rule": "seed = 1000 + i (i = passage index in the seed-7 sample of 800)",
        "gate_size_ok": len(final) >= 0.9 * target,
        "gate_failure_ok": (failed / attempted < 0.05) if attempted else False,
        "finalized_at": utc_now_iso(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n",
                                        encoding="utf-8")
    return stats
