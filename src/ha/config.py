"""YAML config loading and stable hashing (config_hash in the registry, cache keys)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ha.paths import get_paths


def canonical_json(obj: Any) -> str:
    """Key-order-independent JSON; the basis for every hash in the project."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def config_hash(obj: Any) -> str:
    return sha256_hex(obj)[:16]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(relpath: str) -> dict[str, Any]:
    """Load `configs/<relpath>` (e.g. "judge.yaml", "datasets/fiqa.yaml")."""
    return load_yaml(get_paths().configs / relpath)


def domain_description(ds: str) -> str:
    """Dataset config first (D1/D2, hash-locked), else configs/domains.yaml (D4; D-050)."""
    p = get_paths().configs / "datasets" / f"{ds}.yaml"
    if p.exists() and "domain_description" in (cfg := load_yaml(p)):
        return cfg["domain_description"]
    return load_config("domains.yaml")[ds]
