"""Versioned Jinja prompt templates (src/ha/gen/prompts/*.j2) and their content hashes."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPT_DIR = Path(__file__).parent / "gen" / "prompts"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPT_DIR),
        undefined=StrictUndefined,
        keep_trailing_newline=False,
        autoescape=False,
    )


def render(name: str, **kwargs: object) -> str:
    return _env().get_template(f"{name}.j2").render(**kwargs).strip()


def template_hash(name: str) -> str:
    """sha256 of the template source; goes into cache keys and FROZEN.json.

    Line endings are normalised so a Windows (CRLF) and a Linux/Kaggle (LF) checkout hash
    identically.
    """
    source = (PROMPT_DIR / f"{name}.j2").read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(source).hexdigest()[:16]
