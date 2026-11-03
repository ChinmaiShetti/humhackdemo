"""Guard for the frozen pre-registration (D-030): the body above `## Amendments` never changes."""

from __future__ import annotations

import hashlib
import json

from ha.paths import REPO_ROOT

AMENDMENTS_HEADING = "## Amendments"
PREREG_PATH = REPO_ROOT / "docs" / "preregistration.md"
RECORD_PATH = REPO_ROOT / "results" / "PREREG.json"


def frozen_body_sha256(text: str) -> str:
    text = text.replace("\r\n", "\n")
    body, sep, _ = text.partition(AMENDMENTS_HEADING)
    if not sep:
        raise ValueError(f"pre-registration has no '{AMENDMENTS_HEADING}' section")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def check_frozen() -> None:
    """Raise if the frozen body of docs/preregistration.md differs from results/PREREG.json."""
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    actual = frozen_body_sha256(PREREG_PATH.read_text(encoding="utf-8"))
    if actual != record["frozen_body_sha256"]:
        raise RuntimeError(
            "docs/preregistration.md frozen body changed since prereg-v1. Only dated amendments "
            "below '## Amendments' are allowed; methodological changes need an explicit v2."
        )
