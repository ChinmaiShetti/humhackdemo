# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# HumanAlign-RAG

A research **prototype** for a paper (RVCE AIML course project, two-member team).

## Session protocol (mandatory)
- **Start:** read [docs/AGENT_CONTEXT.md](docs/AGENT_CONTEXT.md) first. It is the live snapshot: current phase, blockers, locked items and the next steps in the plan.
- **After every piece of work:** add it to [docs/PAPER_LOG.md](docs/PAPER_LOG.md), under the matching paper section. Results cite their registry `run_id`; exploratory items are tagged [EXPLORATORY] and engineering checks [ENGINEERING]; no unsourced numbers.
- **End:** overwrite `docs/AGENT_CONTEXT.md` (snapshot), append to `docs/lab_notebook.md`, update the tracker in `docs/PLAN.md` §9, add D-entries for open choices, then run `pytest -q` and `ruff check .` and commit.
- **Research question:** can neighbour-grounded LLM judging and curation make synthetic RAG evaluation queries better proxies for authentic human information needs?
- **Execution plan (authoritative for scope):** [docs/PLAN.md](docs/PLAN.md) v2 defines three tiers: CORE PROTOTYPE, OPTIONAL ROBUSTNESS and FULL-SCALE. It also holds the work packages (WPs) and the status tracker (§9). At the start of a session, pick the next unblocked WP; at the end, update the tracker and `docs/lab_notebook.md`.
- **Decisions:** [docs/decisions.md](docs/decisions.md) (D-001 onwards) override the spec.
- **Methodology reference:** [docs/HUMANALIGN_RAG_BUILD_SPEC.md](docs/HUMANALIGN_RAG_BUILD_SPEC.md). Read its scope banner; § numbers refer to it.

**Primary result = system-ranking agreement (τ_b) with held-out authentic queries.** The judge's scores are a means to that result, never the headline.

## Hard rules
- **Never download a model without asking the team first** (D-021). The locked stack is `qwen3:4b-instruct-2507-q4_K_M` (Ollama, digest pinned in `configs/*.yaml` and checked by `ha.llm.require_digest`; the generator **and** the judge). The thinking-only `qwen3:4b` is on disk but must **not** be used (D-028), `bge-m3` (Ollama; dense embeddings) and `BAAI/bge-reranker-v2-m3` (the local HF weights at `D:\ai-models\local\bge-reranker-v2-m3`). BM25 needs no model. Dataset downloads and compute-heavy runs (⚙ in the plan) also need approval.
- **Never fabricate numbers.** Every reported number comes from `results/registry.jsonl` (written only through `ha.registry.record`) and has a `run_id`.
- **Sealed data stays sealed** (`ha.seal`, D-009): A of the held-out datasets (D2, D4) and `cal-test`. Never set `HA_UNSEAL` yourself; opening a seal is a human gate decision.
- **No tuning on held-out data.** Every design choice is made on D1 (FiQA) and `cal-dev`. After the freeze, the judge prompt, weights and thresholds are fixed; a later change becomes `-v2` and is reported.
- Report negative results. Log every open design choice as a D-entry.
- No PII in stored data. Scraping honours robots.txt and site terms, at most 1 request every 2 s.
- **One model on the GPU at a time** (RTX 3050 Laptop, 6 GB). Call `ha.llm.unload_all()` before torch GPU work (the reranker).
- **No ONNX Runtime.** Use pure PyTorch.
- **Everything on D:** the repo, venv, data, caches and temp files (`ha.paths` sets `TEMP`/`TMP` to `HA_TMP`; D-019). Shell commands that install or build must set `$env:TEMP="D:\humanalign-rag\cache\tmp"`.
- **Windows:** use `pathlib`; open files with `encoding="utf-8"`; spawn-safe multiprocessing; DataLoader `num_workers=0` if workers crash.
- **Seed tiers** (D-002): 5 split seeds for primary analyses, 0–2 for secondary, 0 for robustness. Label reduced analyses in their output.

## Setup and commands
Repo: `D:\humanalign-rag`. The Python 3.11 venv is `.venv` (uv-managed; D-012).
```
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
copy .env.example .env
```
```
ha doctor                      # environment checks; must be all OK before a gate
ha models repair-reranker      # hard-link the cached reranker weights (no download)
ha models verify               # smoke-test the locked stack -> env/models.json
pytest -q                      # full suite; must pass before every gate
pytest tests/test_seal.py::test_scoped_unseal_opens_only_that_scope   # single test
ruff check .
```
Implemented pipeline commands:
```
ha data build <ds> | ha data splits <ds>                    # D1/D2 download, normalise, split
ha data hiaq-scrape | hiaq-chunk | hiaq-sheet | hiaq-check    # HiAQ corpus and scenario tools
ha data hiaq-variants | hiaq-ingest                          # survey forms in, responses out
ha systems encode <ds> | index <ds> | run <ds> --qset auth|smoke
ha run E0                                                    # sanity + latency -> registry
ha explore noise-ceiling                                     # EXPLORATORY only (D-040)
```
Still stubs (they print the WP that implements them): `ha gen`, `ha neigh`, `ha judge`, `ha calib …`, `ha select`, `ha eval`, `ha paper build`.

Commit messages look like `P<n>: <what>`. Results with a dirty `git_sha` don't count (D-015).

## Architecture
**Shared infrastructure** (`src/ha/`):
- `paths.py`: every location, from `.env` or the environment. Tests override `HA_DATA`, `HA_CACHE`, `HA_RESULTS` and `HA_LOGS`.
- `config.py`: canonical-JSON hashing.
- `seal.py`: scoped seals, with an access log.
- `registry.py`: fills in `git_sha`, `config_hash`, `frozen_hash` and `timestamp` itself.
- `cache.py`: a SQLite key-value store for every LLM, embedding or reranker call, keyed by a hash of prompt, model digest and params.
- `llm.py`: the Ollama wrapper (structured JSON, embeddings, digests, `unload_all`).
- `models.py`: verify and repair only; it never downloads.

**Core pipeline** per dataset (D1 FiQA is the dev dataset; D2 CQADupStack-Android and D4 HiAQ-Services are held out):
1. **`data/`**: loaders, the HiAQ scrape and survey ingest, `langid.py` (rules plus a lexicon; CMI; EN / ROM_HI / HINGLISH / DEV_HI), and `splits.py` (R 40% / A 60% over 5 seeds, with CAL-auth always in R). R is the only authentic data any method may see.
2. **`gen/`**: qwen3:4b-instruct-2507 produces P-std and P-persona. Their union is U, about 1,600 per dataset. Each synthetic query's single gold passage is its `source_pid`.
3. **`neighbours/`**: bge-m3 top-30 from `R_seed`, reranked to a top-5. No A queries, no self-neighbours, near-duplicate at 0.97.
4. **`judge/`**: qwen3:4b-instruct-2507 (`think=False`, temperature 0) scores F / H / S / P / C through `JudgeOutput`. It runs 2–3 neighbour permutations, seeded by `hash(query_id, perm_idx)`, and takes the median. The **M1-iso** variant has no neighbours. An F < 4 gate, then NNLS weights fitted on `cal-dev`.
5. **`select/`**: R-derived strata, quotas, greedy selection by score with a 0.90 dedupe. The core sets are **B0** (raw), **B1** (stratified random), **M1-iso** and **M1**.
6. **`systems/`**: 7 configurations (C1–C7: BM25 ×2, BGE-M3 dense, RRF, and three reranked variants), with **no system split** (D-022). Each runs once per dataset over every query set, incrementally, into `results/runs/<ds>/<config>/perquery.parquet`. Set-level metrics aggregate those rows.
7. **`eval/`**: ranx metrics; τ_b (primary), τ_AP, winner, MAE; gap measures; paired bootstrap with shared A-resamples.
8. **`report/`**: reads `results/` only.
