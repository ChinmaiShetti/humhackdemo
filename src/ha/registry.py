"""Append-only results registry (spec §0.1, §20.2).

Every number that can reach a table, figure, README or the paper is a row here with a run_id.
git_sha, config_hash, frozen_hash and timestamp are filled in by this module, never by callers.
"""

from __future__ import annotations

import json
import math
import secrets
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ha.config import config_hash, sha256_hex
from ha.paths import get_paths
from ha.utils import git_sha, utc_now_iso

REQUIRED_FIELDS = (
    "run_id",
    "exp",
    "dataset",
    "split_seed",
    "method",
    "system_set",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n_queries",
)
AUTO_FIELDS = ("git_sha", "config_hash", "frozen_hash", "timestamp")
NULLABLE = {"split_seed", "ci_low", "ci_high", "system_set", "method"}


class RegistryError(ValueError):
    pass


def new_run_id(exp: str) -> str:
    return f"{exp}-{datetime.now():%Y%m%dT%H%M%S}-{secrets.token_hex(3)}"


def frozen_hash() -> str | None:
    path = get_paths().frozen
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return sha256_hex(json.load(f))[:16]


def _validate(row: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if missing:
        raise RegistryError(f"registry row missing fields {missing}: {dict(row)}")
    for k in REQUIRED_FIELDS:
        if row[k] is None and k not in NULLABLE:
            raise RegistryError(f"registry field {k!r} may not be None")
    if not row["run_id"]:
        raise RegistryError("registry row needs a non-empty run_id")
    for k in ("value", "ci_low", "ci_high"):
        v = row[k]
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise RegistryError(f"registry field {k!r} must be a finite number, got {v!r}")
    clash = [k for k in AUTO_FIELDS if k in row]
    if clash:
        raise RegistryError(f"fields {clash} are filled by the registry; do not pass them")


def record(rows: Iterable[Mapping[str, Any]], config: Any, path: Path | None = None) -> int:
    """Validate and append rows. `config` is the experiment config the rows were produced with."""
    rows = list(rows)
    for row in rows:
        _validate(row)
    auto = {
        "git_sha": git_sha(),
        "config_hash": config_hash(config),
        "frozen_hash": frozen_hash(),
        "timestamp": utc_now_iso(),
    }
    path = path or get_paths().registry
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            out = {k: row[k] for k in REQUIRED_FIELDS}
            out.update({k: v for k, v in row.items() if k not in out})  # extra fields kept
            out.update(auto)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return len(rows)


def load(path: Path | None = None):
    """Registry as a pandas DataFrame (empty frame with the schema if nothing recorded yet)."""
    import pandas as pd

    path = path or get_paths().registry
    if not path.exists():
        return pd.DataFrame(columns=[*REQUIRED_FIELDS, *AUTO_FIELDS])
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])
