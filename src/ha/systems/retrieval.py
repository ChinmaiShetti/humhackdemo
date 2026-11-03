"""Retrieval configurations C1-C7 (prereg-v1 §5.3; configs/systems.yaml).

Run format everywhere: DataFrame(qid, pid, rank, score), rank 1 = best, top `retrieval_depth`.
- BM25 (bm25s, lucene): index cached per (dataset, k1, b).
- Dense: Ollama bge-m3 corpus embeddings in a resumable float16 memmap; exact inner-product
  search (FAISS IndexFlatIP) on L2-normalised vectors.
- RRF: sum of 1/(k + rank) over the input runs, top-100.
- Rerank: bge-reranker-v2-m3 re-scores the first-stage top-50; ranks 51-100 keep their
  first-stage order below them. Pair scores are cached so C5/C6/C7 share work.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ha import llm
from ha.cache import KVCache
from ha.config import load_config
from ha.data.public import processed_dir
from ha.paths import get_paths


def systems_config() -> dict[str, Any]:
    return load_config("systems.yaml")


def passage_texts(corpus: pd.DataFrame) -> list[str]:
    return [f"{t}\n{x}" if t.strip() else x
            for t, x in zip(corpus["title"], corpus["text"], strict=True)]


def load_corpus(ds: str) -> pd.DataFrame:
    return pd.read_parquet(processed_dir(ds) / "corpus.parquet")


def _run_frame(qids: list[str], pids: np.ndarray, scores: np.ndarray, pid_of) -> pd.DataFrame:
    rows = []
    for qi, q in enumerate(qids):
        for r, (p, s) in enumerate(zip(pids[qi], scores[qi], strict=True), start=1):
            rows.append((q, pid_of(p), r, float(s)))
    return pd.DataFrame(rows, columns=["qid", "pid", "rank", "score"])


# ----------------------------------------------------------------------------- BM25


class BM25Index:
    def __init__(self, ds: str, k1: float, b: float) -> None:
        import bm25s

        self.bm25s = bm25s
        self.corpus = load_corpus(ds)
        self.pids = self.corpus["pid"].tolist()
        path = get_paths().cache / "index" / ds / f"bm25_k1{k1}_b{b}"
        if (path / "params.index.json").exists():
            self.model = bm25s.BM25.load(str(path))
        else:
            self.model = bm25s.BM25(k1=k1, b=b, method="lucene")
            tokens = bm25s.tokenize(passage_texts(self.corpus), lower=True, stopwords="en",
                                    stemmer=None, show_progress=False)
            self.model.index(tokens, show_progress=False)
            path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))

    def search(self, qids: list[str], texts: list[str], k: int) -> pd.DataFrame:
        tokens = self.bm25s.tokenize(texts, lower=True, stopwords="en", stemmer=None,
                                     show_progress=False)
        docs, scores = self.model.retrieve(tokens, k=k, show_progress=False, n_threads=0)
        return _run_frame(qids, docs, scores, lambda i: self.pids[int(i)])


# ----------------------------------------------------------------------------- dense


def _embed_model() -> dict[str, Any]:
    return next(m for m in load_config("models.yaml")["ollama"] if m["kind"] == "embed")


def embed_batch(texts: list[str]) -> np.ndarray:
    cfg, dcfg = _embed_model(), systems_config()["dense"]
    vecs = llm.client().embed(model=cfg["tag"], input=texts, truncate=dcfg["truncate"],
                              options={"num_ctx": dcfg["num_ctx"]}).embeddings
    arr = np.asarray(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def embed_queries(texts: list[str], use_cache: bool = True) -> np.ndarray:
    cfg, dcfg = _embed_model(), systems_config()["dense"]
    digest = llm.require_digest(cfg["tag"], cfg["digest"])
    if not use_cache:
        return embed_batch(texts)
    cache = KVCache(get_paths().cache / "embed_queries.sqlite")
    keys = [KVCache.key_for({"model": digest, "text": t, "num_ctx": dcfg["num_ctx"],
                             "truncate": dcfg["truncate"]}) for t in texts]
    out = [cache.get(k) for k in keys]
    todo = [i for i, v in enumerate(out) if v is None]
    bs = dcfg["batch_size"]
    for b in range(0, len(todo), bs):
        idx = todo[b:b + bs]
        for i, v in zip(idx, embed_batch([texts[i] for i in idx]), strict=True):
            cache.set(keys[i], v.tolist())
            out[i] = v.tolist()
    cache.close()
    return np.asarray(out, dtype=np.float32)


def corpus_embedding_paths(ds: str) -> tuple[Path, Path]:
    d = get_paths().cache / "emb" / ds
    return d / "bge-m3.f16.npy", d / "bge-m3.progress.json"


def encode_corpus(ds: str, log_every: int = 50) -> dict[str, Any]:
    """Resumable corpus encoding into a float16 memmap. Returns resource/timing stats."""
    import psutil

    cfg, dcfg = _embed_model(), systems_config()["dense"]
    digest = llm.require_digest(cfg["tag"], cfg["digest"])
    corpus = load_corpus(ds)
    texts = passage_texts(corpus)
    n = len(texts)
    arr_path, prog_path = corpus_embedding_paths(ds)
    arr_path.parent.mkdir(parents=True, exist_ok=True)
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"done": 0}
    if prog.get("digest", digest) != digest or prog.get("n", n) != n:
        raise RuntimeError(f"{prog_path} was built with a different model or corpus")
    mode = "r+" if arr_path.exists() else "w+"
    mm = np.lib.format.open_memmap(arr_path, mode=mode, dtype=np.float16, shape=(n, 1024))
    evicted = [m for m in llm.unload_all() if not m.startswith("bge-m3")]
    start, t0 = prog["done"], time.perf_counter()
    bs, vram, proc = dcfg["batch_size"], [], psutil.Process()
    for bi, b in enumerate(range(start, n, bs)):
        mm[b:b + bs] = embed_batch(texts[b:b + bs]).astype(np.float16)
        if bi % log_every == 0:
            mm.flush()
            done = min(b + bs, n)
            prog_path.write_text(json.dumps({"done": done, "n": n, "digest": digest}))
            vram += [m.size_vram for m in llm.client().ps().models if m.model.startswith("bge-m3")]
            rate = (done - start) / max(time.perf_counter() - t0, 1e-9)
            logger.info(f"{ds}: encoded {done}/{n} ({rate:.1f} passages/s, "
                        f"eta {(n - done) / max(rate, 1e-9) / 60:.1f} min)")
    mm.flush()
    del mm
    prog_path.write_text(json.dumps({"done": n, "n": n, "digest": digest}))
    secs = time.perf_counter() - t0
    return {"stage": "encode_corpus", "dataset": ds, "passages": n, "encoded_this_run": n - start,
            "seconds": round(secs, 1), "passages_per_s": round((n - start) / max(secs, 1e-9), 2),
            "ollama_vram_gib_max": round(max(vram) / 2**30, 2) if vram else None,
            "rss_gib_peak": round(proc.memory_info().rss / 2**30, 2),
            "evicted_models": evicted, "num_ctx": dcfg["num_ctx"]}


class DenseIndex:
    def __init__(self, ds: str) -> None:
        import faiss

        arr_path, prog_path = corpus_embedding_paths(ds)
        prog = json.loads(prog_path.read_text())
        if prog["done"] != prog["n"]:
            raise RuntimeError(f"{ds}: corpus encoding incomplete ({prog['done']}/{prog['n']})")
        self.pids = load_corpus(ds)["pid"].tolist()
        vecs = np.load(arr_path, mmap_mode="r").astype(np.float32)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, qids: list[str], texts: list[str], k: int, use_cache=True) -> pd.DataFrame:
        q = embed_queries(texts, use_cache=use_cache)
        scores, docs = self.index.search(q, k)
        return _run_frame(qids, docs, scores, lambda i: self.pids[int(i)])


# ----------------------------------------------------------------------------- fusion


def rrf(runs: list[pd.DataFrame], k: int, depth: int) -> pd.DataFrame:
    allr = pd.concat(runs, ignore_index=True)
    allr["rrf"] = 1.0 / (k + allr["rank"])
    fused = allr.groupby(["qid", "pid"], as_index=False)["rrf"].sum()
    fused = fused.sort_values(["qid", "rrf", "pid"], ascending=[True, False, True])
    fused["rank"] = fused.groupby("qid").cumcount() + 1
    fused = fused[fused["rank"] <= depth].rename(columns={"rrf": "score"})
    return fused[["qid", "pid", "rank", "score"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- rerank


class Reranker:
    def __init__(self, use_cache: bool = True, max_length: int | None = None) -> None:
        """`max_length` overrides systems.yaml (512) only for callers with their own frozen
        value: neighbour reranking uses 128 (prereg §5.2). The pair cache key includes it."""
        import torch
        from FlagEmbedding import FlagReranker

        entry = load_config("models.yaml")["hf_local"][0]
        self.rcfg = dict(systems_config()["rerank"])
        if max_length is not None:
            self.rcfg["max_length"] = max_length
        self.revision = entry["revision"]
        llm.unload_all()  # VRAM rule
        torch.cuda.reset_peak_memory_stats()
        self.model = FlagReranker(entry["local_path"], use_fp16=self.rcfg["fp16"],
                                  devices="cuda:0")
        self.cache = KVCache(get_paths().cache / "rerank_pairs.sqlite") if use_cache else None

    def score_pairs(self, pairs: list[tuple[str, str, str]]) -> list[float]:
        """pairs: (query_text, pid, passage_text). Cached by (revision, max_length, query, pid)."""
        keys = [KVCache.key_for({"rev": self.revision, "max_len": self.rcfg["max_length"],
                                 "q": q, "pid": p}) for q, p, _ in pairs]
        out = [self.cache.get(k) if self.cache else None for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            scores = self.model.compute_score([[pairs[i][0], pairs[i][2]] for i in todo],
                                              batch_size=self.rcfg["batch_size"],
                                              max_length=self.rcfg["max_length"])
            scores = [scores] if isinstance(scores, float) else scores
            for i, s in zip(todo, scores, strict=True):
                out[i] = float(s)
                if self.cache:
                    self.cache.set(keys[i], float(s))
        return out  # type: ignore[return-value]


def rerank_run(first: pd.DataFrame, qtext: dict[str, str], ptext: dict[str, str],
               scorer, depth: int) -> pd.DataFrame:
    """Re-score the top-`depth` of each query; the rest keeps first-stage order below."""
    head = first[first["rank"] <= depth].copy()
    tail = first[first["rank"] > depth].copy()
    pairs = [(qtext[q], p, ptext[p]) for q, p in zip(head["qid"], head["pid"], strict=True)]
    scores = scorer(pairs)
    head["score"] = scores
    head = head.sort_values(["qid", "score", "pid"], ascending=[True, False, True])
    head["rank"] = head.groupby("qid").cumcount() + 1
    floor = head.groupby("qid")["score"].min()
    tail = tail.sort_values(["qid", "rank"])
    tail["score"] = tail["qid"].map(floor).fillna(0.0) - tail["rank"] * 1e-3
    out = pd.concat([head, tail], ignore_index=True)
    out = out.sort_values(["qid", "score"], ascending=[True, False])
    out["rank"] = out.groupby("qid").cumcount() + 1
    return out[["qid", "pid", "rank", "score"]].reset_index(drop=True)
