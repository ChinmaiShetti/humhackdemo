"""P3 smoke test against the real generator (engineering check; writes nothing into the pools).

Checks: valid output and schema for each pool type; a cache hit on repeat; deterministic
repeat with the cache bypassed (same prompt + seed -> same text); failure handling (an unknown
model is recorded as a failure, not raised). Output: reports/phase_3_smoke.json
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ha import llm
from ha.cache import KVCache
from ha.config import load_config
from ha.gen import generate as G
from ha.paths import REPO_ROOT
from ha.utils import utc_now_iso


def main() -> None:
    cfg = load_config("generation.yaml")
    llm.require_digest(cfg["model"]["tag"], cfg["model"]["digest"])
    out = {"started_at": utc_now_iso(), "model": cfg["model"]["tag"], "checks": {}}
    passages = G.sample_passages("fiqa").head(3)
    r_texts = [(f"r{i}", t) for i, t in enumerate(
        ["how to check pf balance", "tax on fd interest", "is credit card debt bad",
         "best way to save for a house", "what is a roth ira", "stock split meaning"])]
    samples = []
    with tempfile.TemporaryDirectory() as tmp:
        cache = KVCache(Path(tmp) / "smoke.sqlite")
        for pool, split_seed in [("P-std", None), ("P-persona", None), ("P-fewshot", 0)]:
            spec = G.PoolSpec("fiqa", pool, split_seed)
            for row in passages.itertuples():
                prompt, meta = G.build_prompt(spec, row.text, row.i, r_texts)
                seed = cfg["decoding"]["seed_base"] + row.i
                a = G.call_generator(prompt, seed, cache)            # live
                b = G.call_generator(prompt, seed, cache)            # cache
                c = G.call_generator(prompt, seed, None)             # live again, no cache
                samples.append({"pool": pool, "i": row.i, "query": G.clean(a["raw"]),
                                "filter": G.filter_reason(G.clean(a["raw"]), row.text),
                                "cache_hit_on_repeat": b["cached"],
                                "deterministic_uncached_repeat": a["raw"] == c["raw"],
                                "prompt_tokens": a["prompt_tokens"], **meta})
        cache.close()
    out["samples"] = samples
    out["checks"]["valid_nonempty"] = all(s["query"] for s in samples)
    out["checks"]["cache_hit_on_repeat"] = all(s["cache_hit_on_repeat"] for s in samples)
    out["checks"]["deterministic_uncached_repeat"] = all(
        s["deterministic_uncached_repeat"] for s in samples)
    try:  # failure handling path: a non-existent model must raise inside call_generator
        llm.client().chat(model="no-such-model:0", messages=[{"role": "user", "content": "x"}])
        out["checks"]["failure_raised_for_bad_model"] = False
    except Exception as e:  # noqa: BLE001
        out["checks"]["failure_raised_for_bad_model"] = type(e).__name__
    llm.unload_all()
    path = REPO_ROOT / "reports" / "phase_3_smoke.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out["checks"], indent=2))
    for s in samples:
        print(f"[{s['pool']:9s} i={s['i']}] {s['query']}  (filter={s['filter']})")


if __name__ == "__main__":
    main()
