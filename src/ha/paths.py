"""Every filesystem location, resolved from `.env` / environment (spec §4.2).

Never hard-code paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# Environment wins over .env so tests and Kaggle can override.
load_dotenv(REPO_ROOT / ".env", override=False)

# Temp files stay off C: (D-019). The Windows default TEMP is always set, so .env must win here.
_project_tmp = os.environ.get("HA_TMP") or os.path.join(
    os.environ.get("HA_CACHE", str(REPO_ROOT / "cache")), "tmp"
)
if os.name == "nt":
    os.makedirs(_project_tmp, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = _project_tmp


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    cache: Path
    results: Path
    logs: Path

    @property
    def configs(self) -> Path:
        return REPO_ROOT / "configs"

    @property
    def docs(self) -> Path:
        return REPO_ROOT / "docs"

    @property
    def reports(self) -> Path:
        return REPO_ROOT / "reports"

    @property
    def env_dir(self) -> Path:
        return REPO_ROOT / "env"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def registry(self) -> Path:
        return self.results / "registry.jsonl"

    @property
    def frozen(self) -> Path:
        return self.results / "FROZEN.json"

    @property
    def unseal_log(self) -> Path:
        return self.results / "unseal_log.jsonl"

    @property
    def runs(self) -> Path:
        return self.results / "runs"


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    root = _env_path("HA_ROOT", REPO_ROOT)
    return Paths(
        root=root,
        data=_env_path("HA_DATA", root / "data"),
        cache=_env_path("HA_CACHE", root / "cache"),
        results=_env_path("HA_RESULTS", REPO_ROOT / "results"),
        logs=_env_path("HA_LOGS", REPO_ROOT / "logs"),
    )
