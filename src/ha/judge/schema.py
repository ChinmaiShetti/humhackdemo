"""Structured judge output (spec §9.2), passed to Ollama as the `format=` JSON schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Reason = Literal[
    "ok",
    "not_answerable",
    "too_document_like",
    "over_specific",
    "implausible_need",
    "unnatural_code_mix",
    "near_duplicate_of_real",
]


class JudgeOutput(BaseModel):
    F: int = Field(ge=1, le=5)
    H: int = Field(ge=1, le=5)
    S: int = Field(ge=1, le=5)
    P: int = Field(ge=1, le=5)
    C: int | None = Field(default=None, ge=1, le=5)
    keep: bool
    reason: Reason
