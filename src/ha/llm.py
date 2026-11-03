"""Thin Ollama wrapper shared by generation, judging and qrels pre-labelling.

VRAM rule (spec §4.4): never keep an Ollama model and a torch model on the 6 GB GPU together.
Call `unload_all()` before any torch GPU work; pass keep_alive=0 for one-off calls.
"""

from __future__ import annotations

import os
from typing import Any

import ollama
from pydantic import BaseModel


def client() -> ollama.Client:
    return ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))


def model_digest(tag: str) -> str | None:
    """Full digest of a pulled model, as listed by `ollama list`."""
    for m in client().list().models:
        if m.model == tag or m.model == f"{tag}:latest":
            return m.digest
    return None


class ModelDigestError(RuntimeError):
    pass


def require_digest(tag: str, expected: str) -> str:
    """Refuse to run unless `tag` is pulled with exactly the configured digest (D-028)."""
    actual = model_digest(tag)
    if actual != expected:
        raise ModelDigestError(
            f"{tag}: digest {actual!r} != configured {expected!r}. The model changed or is "
            "missing; do not re-pull without team approval."
        )
    return actual


def server_version() -> str | None:
    import requests

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        return requests.get(f"{host}/api/version", timeout=3).json().get("version")
    except Exception:  # noqa: BLE001
        return None


def unload_all() -> list[str]:
    """Evict every model Ollama currently holds in memory. Returns the evicted tags."""
    c = client()
    evicted = []
    for m in c.ps().models:
        c.generate(model=m.model, prompt="", keep_alive=0)
        evicted.append(m.model)
    return evicted


def chat_json(
    model: str,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    options: dict[str, Any] | None = None,
    think: bool = False,
    keep_alive: str | int | None = None,
) -> tuple[BaseModel, str]:
    """Structured-output chat call. Returns (validated object, raw content).

    Raises pydantic.ValidationError on schema mismatch; callers own the retry policy.
    """
    resp = client().chat(
        model=model,
        messages=messages,
        format=schema.model_json_schema(),
        options=options or {},
        think=think,
        keep_alive=keep_alive,
    )
    raw = resp.message.content
    return schema.model_validate_json(raw), raw
