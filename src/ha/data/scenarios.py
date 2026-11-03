"""Scenario sheet for the HiAQ survey and the automatic leakage check (spec §6.3).

Sheet: 150 chunks, allocated across services in proportion to sqrt(#chunks) (so the small
service is not drowned out), stratified by topic (the URL section) within a service, seed 17.
Leakage check, for each written scenario vs its source chunk:
  (a) rare-term overlap: any token in the 5% rarest of the corpus vocabulary (by document
      frequency; lowercase, >= 3 letters, no stopwords) appears in both scenario and chunk;
  (b) BM25 give-away: BM25(scenario -> all chunks) ranks the source chunk first with a score
      more than 2x the second-best.
Either one rejects the scenario; the team rewrites it by hand and re-runs the check.
  (c) WARN (D-036): >= 60% of the scenario's content words occur in the chunk; a human reviews.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from urllib.parse import urlparse

import bm25s
import numpy as np
import pandas as pd
import regex

from ha.paths import get_paths

SHEET_SEED = 17
N_SCENARIOS = 150
RARE_FRACTION = 0.05
BM25_MARGIN = 2.0
# D-036: extra signal, WARN only (the two spec rules above are the only automatic rejects).
# A scenario reusing >= 60% of its content words from the chunk echoes the document's wording:
# every FAQ question tested as a pseudo-scenario scored >= 0.6; hand-written situations <= 0.5.
OVERLAP_WARN = 0.6
MIN_CHUNK_WORDS = 30
# Scenario chunks come only from topics an individual user faces (D-035); professional topics
# (statutory forms, companies, firms, CAs, deductors ...) stay in the corpus as distractors.
PROFESSIONAL_TOPIC_RE = re.compile(
    r"statutory-form|popular-form|company|\bfirms?\b|\bllp\b|partnership|/ca/|addclient|"
    r"authori[sz]ed?-?signator|tax-?deductor|non-company|\bhuf\b|\btrusts?\b|\baop\b|"
    r"\bboi\b|local-authority|artificial-juridical|intermediar|\bdsc\b|\beris?\b|"
    r"\(ca\)|chartered|type of user|by eris|"
    r"form-?(?:15c|3c|10|29|34|35|49|67)", re.I)
WORD_RE = regex.compile(r"\p{L}{3,}")
STOPWORDS = set(bm25s.stopwords.STOPWORDS_EN)


def _words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text)} - STOPWORDS


def topic_of(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    return "/".join(parts[3:5]) if len(parts) > 3 else "/".join(parts)


def build_sheet(chunks: pd.DataFrame, n: int = N_SCENARIOS, seed: int = SHEET_SEED,
                quota: dict[str, int] | None = None, author: str | None = None) -> pd.DataFrame:
    """`quota` fixes rows per service (D-038); otherwise sqrt(#chunks) allocation (D-035)."""
    rng = np.random.default_rng(seed)
    pool = chunks[(chunks["n_words"] >= MIN_CHUNK_WORDS)
                  & ~chunks["url"].str.contains(PROFESSIONAL_TOPIC_RE)
                  & ~chunks["heading_path"].str.contains(PROFESSIONAL_TOPIC_RE)].copy()
    pool["topic"] = pool["url"].map(topic_of)
    if quota:
        alloc = pd.Series(quota)
        short = {s: q for s, q in quota.items() if (pool["source"] == s).sum() < q}
        if short:
            raise ValueError(f"not enough eligible chunks for quota {short}")
    else:
        weights = pool.groupby("source").size().map(math.sqrt)
        alloc = (weights / weights.sum() * n).round().astype(int)
        while alloc.sum() != n:
            alloc[alloc.idxmax()] += 1 if alloc.sum() < n else -1
    picked = []
    for source, k in alloc.items():
        src = pool[pool["source"] == source]
        topics = sorted(src["topic"].unique())
        rng.shuffle(topics)
        per_topic = {t: list(rng.permutation(src.index[src["topic"] == t])) for t in topics}
        chosen: list[int] = []
        while len(chosen) < min(k, len(src)):  # round-robin over topics
            for t in topics:
                if per_topic[t] and len(chosen) < k:
                    chosen.append(per_topic[t].pop())
        picked += chosen
    sheet = pool.loc[picked, ["chunk_id", "source", "topic", "heading_path", "url", "text"]]
    sheet = sheet.sample(frac=1, random_state=seed).reset_index(drop=True)
    sheet.insert(0, "scenario_id", [f"S{i + 1:03d}" for i in range(len(sheet))])
    sheet["author"] = author or ["A" if i % 2 == 0 else "B" for i in range(len(sheet))]
    sheet["scenario"] = ""
    sheet["notes"] = ""
    return sheet.rename(columns={"text": "chunk_text"})


class LeakageChecker:
    def __init__(self, chunks: pd.DataFrame) -> None:
        self.chunk_ids = chunks["chunk_id"].tolist()
        docs = [_words(t) for t in chunks["text"]]
        df_counts: dict[str, int] = {}
        for d in docs:
            for w in d:
                df_counts[w] = df_counts.get(w, 0) + 1
        threshold = np.quantile(list(df_counts.values()), RARE_FRACTION)
        self.rare = {w for w, c in df_counts.items() if c <= threshold}
        self.text = dict(zip(chunks["chunk_id"], chunks["text"], strict=True))
        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize(chunks["text"].tolist(), stopwords="en",
                                       show_progress=False), show_progress=False)

    def check(self, scenario: str, chunk_id: str) -> dict:
        shared_rare = sorted(_words(scenario) & _words(self.text[chunk_id]) & self.rare)
        ids, scores = self.bm25.retrieve(bm25s.tokenize([scenario], stopwords="en",
                                                        show_progress=False),
                                         k=2, show_progress=False)
        top_id = self.chunk_ids[ids[0][0]]
        s1, s2 = float(scores[0][0]), float(scores[0][1])
        giveaway = top_id == chunk_id and s1 > BM25_MARGIN * max(s2, 1e-9)
        words = _words(scenario)
        overlap = len(words & _words(self.text[chunk_id])) / max(len(words), 1)
        reasons = []
        if shared_rare:
            reasons.append(f"rare document terms: {', '.join(shared_rare)}")
        if giveaway:
            reasons.append(f"BM25 finds the chunk first with margin {s1 / max(s2, 1e-9):.1f}x")
        warn = [] if reasons or overlap < OVERLAP_WARN else [
            f"reuses {overlap:.0%} of its words from the chunk; describe the situation instead"]
        return {"pass": not reasons, "warn": "; ".join(warn), "reasons": "; ".join(reasons),
                "shared_rare": shared_rare, "word_overlap": round(overlap, 2),
                "bm25_top_is_source": top_id == chunk_id,
                "bm25_margin": round(s1 / max(s2, 1e-9), 2)}


def hiaq_dir():
    return get_paths().data / "hiaq"


def write_sheet() -> pd.DataFrame:
    from ha.config import load_config

    cfg = load_config("survey.yaml")
    chunks = pd.read_parquet(hiaq_dir() / "chunks.parquet")
    sheet = build_sheet(chunks, n=sum(cfg["scenario_quota"].values()),
                        quota=cfg["scenario_quota"], author=cfg["author"])
    # UTF-8 with BOM so Excel / Google Sheets open Devanagari and ₹ correctly
    sheet.to_csv(hiaq_dir() / "scenario_sheet.csv", index=False, encoding="utf-8-sig")
    return sheet


def check_output_path(sheet_path: Path) -> Path:
    """The official check file is written only for the official sheet; any other (review)
    sheet gets a sibling `<stem>_check.csv`, so review artifacts never touch official files."""
    official = hiaq_dir() / "scenario_sheet.csv"
    if sheet_path.resolve() == official.resolve():
        return hiaq_dir() / "scenario_check.csv"
    return sheet_path.with_name(f"{sheet_path.stem}_check.csv")


def check_sheet(path=None, out_path=None) -> pd.DataFrame:
    """Read-only on the sheet. Writes only the check output (see check_output_path)."""
    sheet_path = Path(path) if path else hiaq_dir() / "scenario_sheet.csv"
    out_path = Path(out_path) if out_path else check_output_path(sheet_path)
    chunks = pd.read_parquet(hiaq_dir() / "chunks.parquet")
    sheet = pd.read_csv(sheet_path, encoding="utf-8-sig", keep_default_na=False)
    checker = LeakageChecker(chunks)
    rows = []
    for r in sheet.itertuples():
        text = re.sub(r"\s+", " ", str(r.scenario)).strip()
        if not text:
            rows.append({"scenario_id": r.scenario_id, "status": "EMPTY", "reasons": ""})
            continue
        res = checker.check(text, r.chunk_id)
        status = "REJECT" if not res["pass"] else ("WARN" if res["warn"] else "PASS")
        rows.append({"scenario_id": r.scenario_id, "status": status,
                     "reasons": res["reasons"] or res["warn"], "word_overlap": res["word_overlap"],
                     "bm25_margin": res["bm25_margin"]})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def draw_replacements(sheet: pd.DataFrame, chunks: pd.DataFrame, skip_ids: list[str],
                      seed: int = SHEET_SEED, exclude: list[str] | None = None) -> pd.DataFrame:
    """Deterministic replacement CHUNKS (never scenarios) for rows marked skip (D-044).

    For each skipped row, in scenario_id order: same source; eligible = the sheet's own
    eligibility rules (>= MIN_CHUNK_WORDS words, not a professional topic by the current
    PROFESSIONAL_TOPIC_RE) and not already in the sheet or already drawn; topics not yet in
    the sheet are preferred; ties broken by a seeded permutation of chunk_id-sorted candidates.
    """
    rng = np.random.default_rng(seed)
    pool = chunks[(chunks["n_words"] >= MIN_CHUNK_WORDS)
                  & ~chunks["url"].str.contains(PROFESSIONAL_TOPIC_RE)
                  & ~chunks["heading_path"].str.contains(PROFESSIONAL_TOPIC_RE)].copy()
    pool["topic"] = pool["url"].map(topic_of)
    pool = pool.sort_values("chunk_id").reset_index(drop=True)
    pool["order"] = rng.permutation(len(pool))
    used_chunks = set(sheet["chunk_id"]) | set(exclude or [])
    used_topics = set(sheet["topic"])
    skipped = sheet[sheet["scenario_id"].isin(skip_ids)].sort_values("scenario_id")
    missing = set(skip_ids) - set(skipped["scenario_id"])
    if missing:
        raise ValueError(f"unknown scenario ids: {sorted(missing)}")
    rows = []
    for r in skipped.itertuples():
        cand = pool[(pool["source"] == r.source) & ~pool["chunk_id"].isin(used_chunks)].copy()
        cand["new_topic"] = ~cand["topic"].isin(used_topics)
        cand = cand.sort_values(["new_topic", "order"], ascending=[False, True])
        if cand.empty:
            raise ValueError(f"no eligible replacement for {r.scenario_id}")
        pick = cand.iloc[0]
        used_chunks.add(pick["chunk_id"])
        used_topics.add(pick["topic"])
        rows.append({"replaces_scenario_id": r.scenario_id, "replaced_chunk_id": r.chunk_id,
                     "new_chunk_id": pick["chunk_id"], "source": pick["source"],
                     "topic": pick["topic"], "heading_path": pick["heading_path"],
                     "url": pick["url"], "chunk_text": pick["text"],
                     "scenario": "", "status": "NEEDS HUMAN-WRITTEN SCENARIO"})
    return pd.DataFrame(rows)
