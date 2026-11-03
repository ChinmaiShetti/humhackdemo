"""Sealed-data guard (spec §0.1, §20.3; scoped per plan I-4).

Scopes:
  cal_test         the 80-item calibration test split
  heldout_A        authentic test sets A of held-out datasets (D2-D4)
  heldout_systems  metrics of the held-out system half

HA_UNSEAL: "0" or unset = everything sealed; "1" = everything open;
otherwise a comma-separated list of scopes. Every granted access is logged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from ha.paths import get_paths
from ha.utils import git_sha, utc_now_iso

SCOPES = frozenset({"cal_test", "heldout_A", "heldout_systems"})

F = TypeVar("F", bound=Callable[..., Any])


class SealedDataError(RuntimeError):
    pass


def unsealed_scopes() -> frozenset[str]:
    raw = os.environ.get("HA_UNSEAL", "0").strip()
    if raw in ("", "0"):
        return frozenset()
    if raw == "1":
        return SCOPES
    scopes = frozenset(s.strip() for s in raw.split(",") if s.strip())
    unknown = scopes - SCOPES
    if unknown:
        raise ValueError(f"Unknown HA_UNSEAL scope(s): {sorted(unknown)}; valid: {sorted(SCOPES)}")
    return scopes


def require_unsealed(scope: str, what: str, dataset: str | None = None) -> None:
    """Raise SealedDataError unless `scope` is unsealed; log the access when it is."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown seal scope {scope!r}")
    if scope not in unsealed_scopes():
        raise SealedDataError(
            f"{what} is sealed (scope={scope!r}, dataset={dataset!r}). "
            "It opens only after the relevant gate; see docs/PLAN.md I-4."
        )
    if scope == "heldout_A":
        from ha.freeze import is_signed

        if not is_signed():
            raise SealedDataError(
                f"{what} stays sealed until results/FROZEN.json is written and signed, and "
                "unchanged since (prereg §6 as amended by A3; `ha freeze`)."
            )
    log = get_paths().unseal_log
    log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": utc_now_iso(),
        "scope": scope,
        "what": what,
        "dataset": dataset,
        "git_sha": git_sha(),
        "pid": os.getpid(),
    }
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def sealed(scope: str, what: str | None = None) -> Callable[[F], F]:
    """Decorator for loaders. A `dataset` kwarg, if present, is recorded in the log."""

    def deco(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            require_unsealed(scope, what or fn.__qualname__, kwargs.get("dataset"))
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return deco
