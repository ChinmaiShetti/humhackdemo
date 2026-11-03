"""Dense embeddings with the locked Ollama bge-m3 (D-024), digest-checked and cached.

Short texts (queries) are cached per text in SQLite; corpora get array caches in P2.
"""

from __future__ import annotations

import numpy as np

from ha import llm
from ha.cache import KVCache
from ha.config import load_config
from ha.paths import get_paths


def _embed_cfg() -> dict:
    return next(m for m in load_config("models.yaml")["ollama"] if m["kind"] == "embed")


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """L2-normalised float32 embeddings, shape (len(texts), 1024)."""
    cfg = _embed_cfg()
    digest = llm.require_digest(cfg["tag"], cfg["digest"])
    cache = KVCache(get_paths().cache / "embed_queries.sqlite")
    keys = [KVCache.key_for({"model": digest, "text": t}) for t in texts]
    out: list[list[float] | None] = [cache.get(k) for k in keys]
    todo = [i for i, v in enumerate(out) if v is None]
    for b in range(0, len(todo), batch_size):
        idx = todo[b:b + batch_size]
        vecs = llm.client().embed(model=cfg["tag"], input=[texts[i] for i in idx]).embeddings
        for i, v in zip(idx, vecs, strict=True):
            cache.set(keys[i], v)
            out[i] = v
    cache.close()
    arr = np.asarray(out, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
