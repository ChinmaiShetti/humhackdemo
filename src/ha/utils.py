"""Seeding, logging, progress, checkpointing, git and time helpers (spec §20.4, §20.6)."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache

from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ha.paths import REPO_ROOT, get_paths


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@lru_cache(maxsize=1)
def git_sha() -> str:
    """HEAD sha, suffixed with '-dirty' when tracked files have uncommitted changes. The
    append-only results registry is output, not code, so appending to it is not 'dirty'."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
             ":(exclude)results/registry.jsonl"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch; make cuDNN deterministic."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_logging_ready = False


def setup_logging(level: str = "INFO") -> None:
    """loguru to stderr and logs/<date>.log (UTF-8). Idempotent."""
    global _logging_ready
    if _logging_ready:
        return
    logs = get_paths().logs
    logs.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(logs / f"{datetime.now():%Y-%m-%d}.log", level="DEBUG", encoding="utf-8")
    _logging_ready = True


def progress() -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


class Checkpointer:
    """Call `save_fn` every `every` items and on `flush()` (long jobs checkpoint every 200)."""

    def __init__(self, save_fn: Callable[[], None], every: int = 200) -> None:
        self.save_fn = save_fn
        self.every = every
        self._pending = 0

    def step(self, n: int = 1) -> None:
        self._pending += n
        if self._pending >= self.every:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            self.save_fn()
            self._pending = 0
