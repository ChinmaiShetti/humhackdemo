"""Verify the locked local model stack (configs/models.yaml); never downloads anything.

Results (digest / revision, dims, peak VRAM, status) go to env/models.json, which later runs
(and any Kaggle run) compare against to prove they used identical models.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from ha import llm
from ha.config import load_config
from ha.paths import get_paths
from ha.utils import setup_logging, utc_now_iso

SMOKE_TEXTS = [
    "digilocker marksheet not showing what to do",
    "upi payment fail ho gaya par paise kat gaye",
    "आधार में मोबाइल नंबर कैसे अपडेट करें",
]
# Identify cached blobs by content, so the repair needs no network access.
_BLOB_SIGNATURES = {
    "model.safetensors": lambda head: b'"__metadata__"' in head[:200],
    "tokenizer.json": lambda head: head.lstrip().startswith(b'{') and b'"truncation"' in head,
    "config.json": lambda head: b'"architectures"' in head,
    "tokenizer_config.json": lambda head: b'"added_tokens_decoder"' in head,
}


def _manifest_path() -> Path:
    return get_paths().env_dir / "models.json"


def _save_manifest(m: dict[str, Any]) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reranker_entry() -> dict[str, Any]:
    return load_config("models.yaml")["hf_local"][0]


# --------------------------------------------------------------------------- repair


def repair_reranker() -> Path:
    """Build a clean model folder from hard links to the cached blobs (0 bytes, no download)."""
    entry = reranker_entry()
    blobs = Path(entry["cache_repo"]) / "blobs"
    target = Path(entry["local_path"])
    target.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    for blob in blobs.iterdir():
        if blob.suffix in (".lock", ".incomplete") or not blob.is_file():
            continue
        with open(blob, "rb") as f:
            head = f.read(400)
        for name, matches in _BLOB_SIGNATURES.items():
            if name not in found and matches(head):
                found[name] = blob
                break
    missing = set(_BLOB_SIGNATURES) - set(found)
    if missing:
        raise RuntimeError(
            f"cached blobs missing for {sorted(missing)}; ask the team (no download)"
        )
    for name, blob in found.items():
        link = target / name
        if link.exists():
            if os.path.samefile(link, blob):
                continue
            raise RuntimeError(f"{link} exists and is not the cached blob; inspect it manually")
        os.link(blob, link)  # NTFS hard link on the same volume: no copy, no extra space
        logger.info(f"linked {name} -> {blob.name[:12]}")
    return target


# --------------------------------------------------------------------------- verify


class _Smoke(BaseModel):
    language: str
    is_question: bool


def _verify_chat(tag: str) -> dict[str, Any]:
    digest = llm.model_digest(tag)
    if digest is None:
        return {"status": "missing"}
    samples, t0 = [], time.perf_counter()
    try:
        for text in SMOKE_TEXTS:
            obj, _ = llm.chat_json(
                tag,
                [{"role": "user", "content": f"Classify this text: {text!r}. Return JSON."}],
                _Smoke,
                options={"temperature": 0, "seed": 0, "num_ctx": 4096},
                think=False,
            )
            samples.append(obj.model_dump())
        status = "ok"
    except Exception as e:  # noqa: BLE001
        status = f"error: {type(e).__name__}: {e}"[:300]
    return {
        "digest": digest,
        "ollama_version": llm.server_version(),
        "status": status,
        "structured_json_samples": samples,
        "seconds_3_calls": round(time.perf_counter() - t0, 2),
    }


def _verify_embed(tag: str, dim: int) -> dict[str, Any]:
    import numpy as np

    digest = llm.model_digest(tag)
    if digest is None:
        return {"status": "missing"}
    try:
        vecs = np.asarray(llm.client().embed(model=tag, input=SMOKE_TEXTS).embeddings)
        norms = np.linalg.norm(vecs, axis=1)
        ok = vecs.shape == (len(SMOKE_TEXTS), dim)
        return {
            "digest": digest,
            "status": "ok" if ok else f"error: shape {vecs.shape}",
            "dim": int(vecs.shape[1]),
            "norms": [round(float(n), 4) for n in norms],
            "cos_en_hinglish": round(float(vecs[0] @ vecs[1] / (norms[0] * norms[1])), 4),
        }
    except Exception as e:  # noqa: BLE001
        return {"digest": digest, "status": f"error: {type(e).__name__}: {e}"[:300]}


def _verify_reranker() -> dict[str, Any]:
    import torch

    entry = reranker_entry()
    path = Path(entry["local_path"])
    info: dict[str, Any] = {"local_path": str(path)}
    if not (path / "model.safetensors").exists():
        return {**info, "status": "not repaired yet: run `ha models repair-reranker`"}
    info["revision"] = next((p.name for p in (Path(entry["cache_repo"]) / "snapshots").iterdir()),
                            None)
    torch.cuda.reset_peak_memory_stats()
    try:
        from FlagEmbedding import FlagReranker

        m = FlagReranker(str(path), use_fp16=True, devices="cuda:0")
        scores = m.compute_score([
            [SMOKE_TEXTS[0], "How to fetch a marksheet in DigiLocker"],
            [SMOKE_TEXTS[0], "IRCTC tatkal booking timings"],
        ])
        info["scores_relevant_vs_not"] = [round(float(s), 3) for s in scores]
        info["status"] = "ok" if scores[0] > scores[1] else "error: relevant pair not ranked higher"
    except Exception as e:  # noqa: BLE001
        info["status"] = f"error: {type(e).__name__}: {e}"[:300]
    info["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return info


def verify() -> bool:
    setup_logging()
    cfg = load_config("models.yaml")
    manifest: dict[str, Any] = {"verified_at": utc_now_iso(), "ollama": {}, "hf_local": {}}
    for entry in cfg["ollama"]:
        logger.info(f"verify ollama {entry['tag']}")
        res = (_verify_chat(entry["tag"]) if entry["kind"] == "chat"
               else _verify_embed(entry["tag"], entry["dim"]))
        if res.get("digest") != entry["digest"]:
            res["status"] = f"error: digest {res.get('digest')} != configured {entry['digest']}"
        manifest["ollama"][entry["tag"]] = {"role": entry["role"], **res}
        _save_manifest(manifest)
    evicted = llm.unload_all()  # VRAM rule: nothing from Ollama on the GPU during torch work
    if evicted:
        logger.info(f"unloaded Ollama models {evicted}")
    entry = reranker_entry()
    logger.info(f"verify reranker {entry['id']}")
    manifest["hf_local"][entry["id"]] = {"role": entry["role"], **_verify_reranker()}
    _save_manifest(manifest)
    bad = [k for sec in ("ollama", "hf_local") for k, v in manifest[sec].items()
           if v.get("status") != "ok"]
    if bad:
        logger.warning(f"models not ok: {bad}")
    return not bad
