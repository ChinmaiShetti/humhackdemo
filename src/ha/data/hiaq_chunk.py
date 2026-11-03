"""Heading-aware chunking of scraped HiAQ help pages (spec §6.2).

- FAQ items become one chunk each: `<details><summary>Q</summary>A</details>` accordions, or a
  short block ending in "?" followed by its answer blocks.
- Other text is grouped by h1-h4 section and split into windows of <= 350 words, 50 overlap.
- Boilerplate: block texts on >= 30% of a source's pages (and >= 3 pages) are dropped.
- Dedupe: exact (normalised) and near-duplicate (MinHash Jaccard >= 0.9 on word 3-shingles).
"Tokens" are whitespace words (D-033).
Output: HA_DATA/hiaq/chunks.parquet + chunk_stats.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Tag
from datasketch import MinHash, MinHashLSH
from loguru import logger

from ha.config import load_config
from ha.data.hiaq_scrape import raw_dir
from ha.paths import get_paths

MAX_WORDS, OVERLAP, MIN_SECTION_WORDS, MIN_ANSWER_WORDS = 350, 50, 20, 3
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
BLOCKS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd", "td", "th", "summary",
          "div", "pre", "blockquote"}
STRIP = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg",
         "button", "select", "iframe"]
QUESTION_MAX_WORDS = 40
QUESTION_MIN_WORDS = 3
LINK_BLOCK_MAX_WORDS = 12


@dataclass
class Block:
    kind: str          # heading | question | text
    text: str
    level: int = 0


@dataclass
class Chunk:
    kind: str          # qa | section
    heading_path: list[str]
    text: str
    extra: dict[str, Any] = field(default_factory=dict)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_leaf_block(el: Tag) -> bool:
    return el.name in BLOCKS and not any(
        isinstance(d, Tag) and d.name in BLOCKS for d in el.descendants
    )


def extract_blocks(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(STRIP):
        t.decompose()
    root = soup.find("main") or soup.body or soup
    blocks: list[Block] = []
    for details in root.find_all("details"):
        summary = details.find("summary")
        if summary is None:
            continue
        q = _norm(summary.get_text(" "))
        summary.decompose()
        a = _norm(details.get_text(" "))
        details.replace_with(soup.new_string(""))
        blocks.append(Block("question", q))
        blocks.append(Block("text", a))
    for el in root.find_all(True):
        if not _is_leaf_block(el):
            continue
        text = _norm(el.get_text(" "))
        if not text:
            continue
        link_text = _norm(" ".join(a.get_text(" ") for a in el.find_all("a")))
        if link_text == text and len(text.split()) <= LINK_BLOCK_MAX_WORDS:
            continue  # navigation / menu item, not content
        n_words = len(text.split())
        if el.name in HEADINGS:
            blocks.append(Block("heading", text, HEADINGS[el.name]))
        elif text.endswith("?") and QUESTION_MIN_WORDS <= n_words <= QUESTION_MAX_WORDS:
            blocks.append(Block("question", text))
        else:
            blocks.append(Block("text", text))
    return blocks


def _windows(words: list[str]) -> list[list[str]]:
    if len(words) <= MAX_WORDS:
        return [words]
    step = MAX_WORDS - OVERLAP
    return [words[i:i + MAX_WORDS] for i in range(0, len(words) - OVERLAP, step)]


def chunk_blocks(blocks: list[Block], boilerplate: set[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    path: dict[int, str] = {}
    section: list[str] = []
    question: str | None = None
    answer: list[str] = []

    def heading_path() -> list[str]:
        return [path[k] for k in sorted(path)]

    def flush_section() -> None:
        words = " ".join(section).split()
        if len(words) >= MIN_SECTION_WORDS:
            for w in _windows(words):
                chunks.append(Chunk("section", heading_path(), " ".join(w)))
        section.clear()

    def flush_qa() -> None:
        nonlocal question
        if question is not None:
            words = " ".join(answer).split()
            if len(words) >= MIN_ANSWER_WORDS:
                for w in _windows(words):
                    chunks.append(Chunk("qa", heading_path(), f"{question}\n{' '.join(w)}",
                                        {"question": question}))
        question = None
        answer.clear()

    for b in blocks:
        if b.text in boilerplate:
            continue
        if b.kind == "heading":
            flush_qa()
            flush_section()
            path = {k: v for k, v in path.items() if k < b.level}
            path[b.level] = b.text
        elif b.kind == "question":
            flush_qa()
            flush_section()
            question = b.text
        elif question is not None:
            answer.append(b.text)
        else:
            section.append(b.text)
    flush_qa()
    flush_section()
    return chunks


def _minhash(text: str) -> MinHash:
    words = text.lower().split()
    m = MinHash(num_perm=128, seed=1)
    for i in range(max(len(words) - 2, 1)):
        m.update(" ".join(words[i:i + 3]).encode("utf-8"))
    return m


def build_chunks() -> pd.DataFrame:
    cfg = load_config("hiaq_sources.yaml")
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for source in cfg["sources"]:
        index = raw_dir(source) / "pages.jsonl"
        if not index.exists():
            continue
        pages = [json.loads(line) for line in open(index, encoding="utf-8")]
        pages = [p for p in pages if p.get("status") == 200 and "sha1" in p]
        page_blocks = {
            p["sha1"]: extract_blocks((raw_dir(source) / f"{p['sha1']}.html").read_text(
                encoding="utf-8"))
            for p in pages
        }
        df_counts = Counter(t for bl in page_blocks.values() for t in {b.text for b in bl})
        min_pages = max(3, int(0.3 * len(pages)))
        boilerplate = (
            {t for t, c in df_counts.items() if c >= min_pages} if len(pages) >= 3 else set()
        )
        n_before = len(rows)
        for p in pages:
            for i, ch in enumerate(chunk_blocks(page_blocks[p["sha1"]], boilerplate)):
                rows.append({
                    "source": source,
                    "doc_id": f"{source}/{p['sha1'][:12]}",
                    "chunk_id": f"{source}/{p['sha1'][:12]}/{i:03d}",
                    "url": p["final_url"],
                    "heading_path": " > ".join(ch.heading_path),
                    "kind": ch.kind,
                    "text": ch.text,
                    "n_words": len(ch.text.split()),
                    "fetched_at": p["fetched_at"],
                })
        stats[source] = {"pages": len(pages), "boilerplate_blocks": len(boilerplate),
                         "chunks_raw": len(rows) - n_before}
    df = pd.DataFrame(rows)

    # exact then near-duplicate removal (first occurrence kept)
    df["norm"] = df["text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    n_raw = len(df)
    df = df.drop_duplicates("norm").reset_index(drop=True)
    n_exact = len(df)
    lsh = MinHashLSH(threshold=0.9, num_perm=128)
    keep = []
    for i, text in enumerate(df["text"]):
        m = _minhash(text)
        if lsh.query(m):
            keep.append(False)
            continue
        lsh.insert(str(i), m)
        keep.append(True)
    df = df[keep].drop(columns="norm").reset_index(drop=True)
    stats["totals"] = {"raw": n_raw, "after_exact_dedupe": n_exact, "final": len(df),
                       "by_kind": df["kind"].value_counts().to_dict(),
                       "by_source": df["source"].value_counts().to_dict(),
                       "words_median": float(df["n_words"].median()),
                       "words_p90": float(df["n_words"].quantile(0.9))}
    out = get_paths().data / "hiaq"
    df.to_parquet(out / "chunks.parquet", index=False)
    (out / "chunk_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    logger.info(f"HiAQ chunks: {stats['totals']}")
    return df
