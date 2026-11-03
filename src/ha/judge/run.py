"""Judge calls, P4.2 (spec §9.2; prereg §5.2; configs/judge.yaml, frozen).

One call = the digest-pinned judge, temperature 0, seed 0, num_ctx 4096, think=False, with the
JudgeOutput JSON schema as `format`. Cached in cache/judge.sqlite, keyed by (messages, model
digest, decoding options). On a JSON validation failure the call is retried ONCE, with the
validation error appended; if it still fails, the call is marked failed. Transport errors (for
example a prompt over the context) are failures and are not retried.

M1: permutations 0 and 1, plus 2 if any dimension differs by >= 2; median per dimension. Any
failed permutation -> the query fails (not selectable). M1-iso: one call, no neighbour block.
Reproducibility: a run starts from a fresh model state and processes queries in fixed order
(the same Ollama KV-reuse effect as in P3, D-046).
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ha import llm
from ha.cache import KVCache
from ha.config import load_config
from ha.judge.aggregate import median_dims, needs_third
from ha.judge.prompt import judge_messages
from ha.judge.schema import JudgeOutput


def _cfg() -> dict[str, Any]:
    return load_config("judge.yaml")


def _options() -> dict[str, Any]:
    d = _cfg()["decoding"]
    return {"temperature": d["temperature"], "seed": d["seed"], "num_ctx": d["num_ctx"]}


def _raw_call(messages, cache: KVCache | None, client) -> dict:
    cfg = _cfg()
    key = KVCache.key_for({"messages": messages, "model": cfg["model"]["digest"],
                           "options": _options(), "think": cfg["model"]["think"],
                           "schema": JudgeOutput.model_json_schema()})
    if cache is not None and (hit := cache.get(key)) is not None:
        return {**hit, "cached": True}
    r = (client or llm.client()).chat(model=cfg["model"]["tag"], messages=messages,
                                      format=JudgeOutput.model_json_schema(),
                                      think=cfg["model"]["think"], options=_options())
    out = {"raw": r.message.content, "prompt_tokens": r.prompt_eval_count,
           "output_tokens": r.eval_count}
    if cache is not None and out["raw"] is not None:
        cache.set(key, out)
    return {**out, "cached": False}


def judge_call(messages: list[dict], cache: KVCache | None = None, client=None) -> dict:
    """Returns {'ok', 'output' (dict|None), 'attempts', 'error', 'raw', ...}."""
    rec: dict[str, Any] = {"attempts": 0, "cached": []}
    msgs = list(messages)
    for attempt in (1, 2):
        rec["attempts"] = attempt
        try:
            res = _raw_call(msgs, cache, client)
        except Exception as e:  # noqa: BLE001 - transport or context error: failure, no retry
            rec.update(ok=False, output=None, error=f"{type(e).__name__}: {e}"[:300])
            return rec
        rec["cached"].append(res["cached"])
        rec.update(raw=res["raw"], prompt_tokens=res["prompt_tokens"])
        try:
            out = JudgeOutput.model_validate_json(res["raw"] or "")
            rec.update(ok=True, output=out.model_dump(), error=None)
            return rec
        except ValidationError as e:
            err = str(e).splitlines()[0][:300]
            rec["error"] = f"validation: {err}"
            if attempt == 1:  # retry once with the validation error appended (spec §9.2)
                msgs = msgs + [
                    {"role": "assistant", "content": res["raw"] or ""},
                    {"role": "user", "content": f"That output was invalid: {err}. Return only "
                                                "JSON matching the schema."}]
    rec.update(ok=False, output=None)
    return rec


def judge_m1(query_id: str, query: str, passage: str, neighbours: list[str], lang_class: str,
             domain: str, cache: KVCache | None = None, client=None) -> dict:
    perms = []
    for p in (0, 1):
        c = judge_call(judge_messages(query_id, query, passage, neighbours, lang_class, domain, p),
                       cache, client)
        perms.append({"perm": p, **c})
        if not c["ok"]:
            return {"ok": False, "perms": perms, "dims": None, "third_run": False,
                    "error": c["error"]}
    third = needs_third(perms[0]["output"], perms[1]["output"])
    if third:
        c = judge_call(judge_messages(query_id, query, passage, neighbours, lang_class, domain, 2),
                       cache, client)
        perms.append({"perm": 2, **c})
        if not c["ok"]:
            return {"ok": False, "perms": perms, "dims": None, "third_run": True,
                    "error": c["error"]}
    return {"ok": True, "perms": perms, "third_run": third,
            "dims": median_dims([p["output"] for p in perms]), "error": None}


def judge_iso(query_id: str, query: str, passage: str, lang_class: str, domain: str,
              cache: KVCache | None = None, client=None) -> dict:
    c = judge_call(judge_messages(query_id, query, passage, None, lang_class, domain, 0),
                   cache, client)
    return {"ok": c["ok"], "perms": [{"perm": 0, **c}], "third_run": False,
            "dims": median_dims([c["output"]]) if c["ok"] else None, "error": c["error"]}
