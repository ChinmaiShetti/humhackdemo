"""Token-level language ID, Code-Mixing Index and language classes (spec §6.5, D-032).

Rules and lexicons only (no model). Token tags:
  hi   Hindi (Devanagari script, or a romanised-Hindi lexicon word)
  en   English (FiQA-derived English vocabulary)
  ne   named entity / service name / all-caps acronym (excluded from CMI)
  univ digits, other scripts, unknown Latin words (language-independent; excluded from CMI)
Ambiguous words (in the Hindi lexicon AND frequent in English: to, main, me, is ...) take the
majority language of the unambiguous hi/en tokens in the same text (ties -> en).

CMI (Das & Gambäck 2014): 100 * (1 - max(n_hi, n_en) / (n - u)), 0 when n - u == 0,
where n = all tokens and u = ne + univ tokens.
Classes: DEV_HI | OTHER (by script share), else EN / HINGLISH / ROM_HI by Hindi share
n_hi / (n_hi + n_en); texts with no hi/en tokens are EN if they are Latin script.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import regex

from ha.config import load_config

LEXICON_DIR = Path(__file__).parent / "lexicons"
TOKEN_RE = regex.compile(r"\p{L}[\p{L}\p{M}]*|\p{N}+")
DEVANAGARI_RE = regex.compile(r"\p{Devanagari}")
LATIN_RE = regex.compile(r"\p{Latin}")


def _read_list(name: str) -> list[str]:
    lines = (LEXICON_DIR / name).read_text(encoding="utf-8").splitlines()
    return [ln.strip().lower() for ln in lines if ln.strip() and not ln.startswith("#")]


@dataclass(frozen=True)
class Lexicons:
    hindi: frozenset[str]
    english: frozenset[str]
    ambiguous: frozenset[str]
    named_entities: frozenset[str]


@lru_cache(maxsize=1)
def lexicons() -> Lexicons:
    cfg = load_config("langid.yaml")
    lx = cfg["lexicons"]
    counts: dict[str, int] = {}
    for line in _read_list(lx["english"]):
        word, count = line.split("\t")
        counts[word] = int(count)
    hindi = frozenset(_read_list(lx["hindi"]))
    english = frozenset(w for w, c in counts.items() if c >= lx["english_count_min"])
    strong_en = {w for w, c in counts.items() if c >= cfg["ambiguity"]["en_count_min"]}
    return Lexicons(
        hindi=hindi,
        english=english - (hindi - strong_en),  # rare "English" hits of Hindi words stay Hindi
        ambiguous=frozenset(hindi & strong_en),
        named_entities=frozenset(_read_list(lx["named_entities"])),
    )


def _script(tok: str) -> str:
    if DEVANAGARI_RE.search(tok):
        return "deva"
    if LATIN_RE.search(tok):
        return "latin"
    if tok.isdigit():
        return "digit"
    return "other"


def tag_tokens(text: str) -> list[tuple[str, str]]:
    """Return [(token, tag)] with tag in {hi, en, ne, univ}."""
    lx = lexicons()
    raw = TOKEN_RE.findall(text)
    tags: list[str | None] = []
    for tok in raw:
        low, script = tok.lower(), _script(tok)
        if script == "deva":
            tags.append("hi")
        elif script != "latin":
            tags.append("univ")
        elif low in lx.named_entities or (tok.isupper() and len(tok) >= 2):
            tags.append("ne")
        elif low in lx.ambiguous:
            tags.append(None)  # resolved below
        elif low in lx.hindi:
            tags.append("hi")
        elif low in lx.english:
            tags.append("en")
        else:
            tags.append("univ")
    n_hi, n_en = tags.count("hi"), tags.count("en")
    resolved = "hi" if n_hi > n_en else "en"
    return [(tok, tag if tag is not None else resolved) for tok, tag in zip(raw, tags, strict=True)]


def cmi(tags: list[str]) -> float:
    n = len(tags)
    u = sum(t in ("ne", "univ") for t in tags)
    if n - u == 0:
        return 0.0
    return 100.0 * (1 - max(tags.count("hi"), tags.count("en")) / (n - u))


def classify(text: str, shift: float = 0.0) -> dict[str, Any]:
    """Language class of a query. `shift` moves both Hindi-share thresholds (sensitivity ±0.05)."""
    cfg = load_config("langid.yaml")["classes"]
    tagged = tag_tokens(text)
    tags = [t for _, t in tagged]
    letters = [tok for tok, _ in tagged if not tok.isdigit()]
    scripts = [_script(tok) for tok in letters]
    n_letters = max(len(scripts), 1)
    n_hi, n_en = tags.count("hi"), tags.count("en")
    hi_share = n_hi / (n_hi + n_en) if (n_hi + n_en) else 0.0
    if scripts.count("deva") / n_letters >= cfg["dev_min_script_share"]:
        label = "DEV_HI"
    elif scripts.count("other") / n_letters >= cfg["other_min_script_share"]:
        label = "OTHER"
    elif hi_share < cfg["en_max_hi_share"] + shift:
        label = "EN"
    elif hi_share > cfg["rom_hi_min_hi_share"] + shift:
        label = "ROM_HI"
    else:
        label = "HINGLISH"
    return {"label": label, "hi_share": round(hi_share, 4), "cmi": round(cmi(tags), 2),
            "n_hi": n_hi, "n_en": n_en, "tags": tagged}
