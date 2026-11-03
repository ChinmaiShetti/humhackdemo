# HumanAlign-RAG: Execution Plan v2 (prototype-first)

Plan v2.0, 2026-09-27. It **replaces v1.1** following the team's scope correction (D-020).
The spec ([HUMANALIGN_RAG_BUILD_SPEC.md](HUMANALIGN_RAG_BUILD_SPEC.md)) still defines the methodology; this plan defines **what is built now**. Where they conflict, this plan and [decisions.md](decisions.md) win.

**Research question:** can neighbour-grounded LLM judging and curation make synthetic RAG evaluation queries better proxies for authentic human information needs?

**Primary result:** system-ranking agreement (τ_b) between each query set and **held-out authentic queries**. The judge's own scores are a means to that result, never the headline.

**Hard rule:** **no model is downloaded without asking the team first.** Phase 0 downloads zero models. Dataset downloads are listed with their sizes and wait for approval.

---

## 0. How agents use this plan

**Start of a session**
1. Read `docs/AGENT_CONTEXT.md` (the live snapshot), `CLAUDE.md`, the tracker (§9) and the last entry in `docs/lab_notebook.md`.
2. Take the first `todo` work package (WP) whose "Needs" are all `done`.
3. Never start a WP that is blocked on a human, a download approval or a gate.

**End of a session**
1. Update `docs/PAPER_LOG.md` (paper content, sourced numbers only) and overwrite `docs/AGENT_CONTEXT.md` (snapshot).
1. Run `pytest -q` and `ruff check .`.
2. Update the tracker.
3. Add a lab-notebook entry.
4. Commit as `P<n>: <what>`.

Any choice the spec doesn't fix gets a D-entry in `docs/decisions.md`.

---

## 1. Scope tiers

### CORE PROTOTYPE: required for the paper

**Models.** The stack is locked (D-021). Everything below is already on disk.
| Model | Where | Used for |
|---|---|---|
| `qwen3:4b-instruct-2507-q4_K_M` | Ollama, 2.5 GB, digest `0edcdef34593…` (D-028; the thinking-only `qwen3:4b` is **not** used) | Synthetic query generation **and** the judge (`think=False`) |
| `bge-m3` | Ollama, F16, dense embeddings only | Embedding synthetic and authentic queries and corpora; neighbour retrieval; the dense retrieval configurations |
| `BAAI/bge-reranker-v2-m3` | Hugging Face weights in `D:\ai-models\hf-cache` (2.1 GB, needs a local repair, see P0.2) | Reranking candidate neighbours; the rerank configurations |
| BM25 (`bm25s`) | no model | The sparse configurations |

**Datasets**
| ID | Dataset | Role | Download |
|---|---|---|---|
| D1 | FiQA-2018 (HF `BeIR/fiqa` + `BeIR/fiqa-qrels`) | **Development:** every design decision is made here | ~28 MB |
| D2 | CQADupStack-Android (HF `mteb/cqadupstack-android`) | **Held-out**, English, organic queries | ~15 MB |
| D4 | HiAQ-Services (our survey) | **Held-out**, EN / Romanised Hindi / Hinglish. The only source of the code-mixing analysis. Depends on the survey. | scraped help pages (small) |

**Query sets compared**
- **A**: authentic held-out queries (the reference).
- **B0**: raw synthetic (a random size-matched draw from U, over 20 sampling seeds).
- **M1**: HumanAlign-curated.
- Two zero-cost controls are also core, because without them the claim cannot be attributed to the method:
  - **B1**, stratified random. It shows how much topic matching alone explains.
  - **M1-iso**, the same judge with **no neighbours**. It shows how much the neighbour grounding adds, which is the "neighbour-grounded" part of the research question.

**Retrieval configurations** (7, built only from the models above; D-023)
| ID | Configuration |
|---|---|
| C1 | BM25 (k1 = 0.9, b = 0.4) |
| C2 | BM25 (k1 = 1.2, b = 0.75) |
| C3 | BGE-M3 dense |
| C4 | RRF(C1, C3), k = 60 |
| C5 | C3 top-50 → bge-reranker-v2-m3 |
| C6 | C1 top-50 → bge-reranker-v2-m3 |
| C7 | C4 top-50 → bge-reranker-v2-m3 |

The 7 configurations give 21 system pairs for τ_b, where 5 would give only 10. Each extra configuration is almost free, since the corpus is encoded once and the reranker scores are cached.

**Pipeline**
1. Corpus.
2. Qwen3-4B-Instruct-2507 generates P-std and P-persona queries (U, about 1,600 per dataset from 800 source passages).
3. Dedupe.
4. BGE-M3 embeddings.
5. Top-30 authentic neighbours from **R**, reranked to a top-5.
6. The judge scores F, H, S, P and C, using 2–3 neighbour permutations and the median.
7. F gate plus NNLS-weighted score.
8. Stratified selection with dedupe.
9. Run the 7 configurations on A, B0, B1, M1-iso and M1.
10. Compute agreement with A.

**Research controls kept from the spec (not negotiable)**
- Held-out authentic evaluation sets are sealed (`heldout_A`), and so is `cal-test`.
- R and A are split over 5 seeds. The CAL-auth queries sit in R for every seed, and no query is its own neighbour.
- Near-duplicate guardrail at 0.97; dedupe at 0.95 within a pool and 0.90 during selection.
- Human calibration on 200 items (two annotators, κ, prompt iteration on `cal-dev` only). The judge is then frozen in `FROZEN.json`.
- No tuning on held-out data: every design choice is made on D1 and `cal-dev`.
- Language and code-mixing analysis (LID, CMI, per-class gap) on D4.
- Paired bootstrap with shared A-resamples; everything goes to the registry; negative results are reported.

**Metrics:**
- nDCG@10 (primary), MRR@10, Recall@100;
- τ_b (primary agreement measure), τ_AP, winner agreement, MAE and signed bias;
- system-free gap measures: W1 of length, W1 of leakage, JS of topic, TV of language.

### OPTIONAL ROBUSTNESS: only after the core works

Each item needs team approval; items marked ⬇ need a download.
- **Free re-aggregation ablations** (no new LLM calls): no F gate; single dimension (H, S or P only); equal weights; no stratification; position-bias rate from the permutation logs.
- **Cheap non-LLM baselines:** B2 length-matched, B3 kNN realism, B4 low leakage, B5 discriminator (BGE-M3 embeddings plus logistic regression, so no new model).
- **Neighbour ablations** (new judge calls): top-1 neighbour instead of top-5; no reranker.
- **E7 budget curve:** N ∈ {100, 200, 400}.
- **Alternate-judge robustness** with `qwen3:8b`, which is **already on disk**. It is partly offloaded on 6 GB, so it's slow; seed 0 only.
- **P-fewshot pool with M3 / B6**, and the **M2 rewrite** variant (E4).
- **E11 pooled qrels** for M1 queries. This is high priority because of the sparse-qrel risk: Rahmani et al. SIGIR '24 found τ fell to 0.157 with source-only qrels.
- **E8** blinded human indistinguishability test.
- **Distillation** onto BGE-M3 embeddings with a small regression head (no new model).
- ⬇ HF `BAAI/bge-m3` (~2.3 GB): adds BGE-M3 sparse and dense+sparse configurations.
- ⬇ fastText `lid.176` (126 MB, or `.ftz` at 1 MB) as the LID tiebreaker. The core uses rules plus a lexicon.
- ⬇ D3 MIRACL-Hindi (a Hindi condition; a large corpus to subsample).

### FULL-SCALE: future work, stated as such in the paper

- The 20-system pool: other embedders and rerankers (e5, LaBSE, gte, arctic, mMiniLM).
- Separate generator and judge families (llama3.1:8b / gemma3:4b generators; qwen3.5 judges at 2B, 4B and 9B).
- The docT5query pool; MuRIL / mE5 distillation; bulk runs with vLLM or Kaggle; every analysis at 5 seeds on every dataset.

**A limitation to state in the paper, and our mitigation.** The same model (Qwen3:4B) generates **and** judges, so the judge may favour its own style (self-preference). The mitigation:
- human calibration items include Qwen3-generated queries, so judge–human agreement is measured on exactly this setup;
- the M1-iso control;
- the optional qwen3:8b judge check.

---

## 2. Changes from the spec, and why

| Spec | Prototype | Reason |
|---|---|---|
| About 20 systems, split into dev and held-out halves | 7 configurations; **no system split** (D-022) | With 7 systems, a 3/4 split makes τ meaningless. Tuning protection comes from held-out **datasets** (D2, D4), which stay sealed, plus design choices made only on D1 and `cal-dev`. Selection never looks at system results. |
| 4 datasets | D1 (dev), D2 and D4 (held-out); D3 optional | Compute and download budget; D4 carries the code-mix analysis |
| Generator llama3.1:8b; judge qwen3.5:4b | Both are `qwen3:4b-instruct-2507-q4_K_M` (D-028) | The locked local stack; the risk is disclosed with the mitigations above |
| 1,500 source passages, pool ≈ 3,000 | 800 passages, pool ≈ 1,600 | The judge budget on the 3050 (§5). P0.3 re-checks this against measured throughput. |
| Neighbour and system embeddings from HF BGE-M3 (dense + sparse) | Ollama BGE-M3, dense only | No download (D-024). The sparse configurations become optional. |
| E0: dense models against their model cards | BM25 against the BEIR FiQA figure (0.236 ± 0.02); BGE-M3 dense against the published MTEB FiQA score (± 0.03), treated as a warning, since the GGUF may differ slightly | — |
| Seed tiers from D-002 | Unchanged: 5 seeds for the primary analyses, 0–2 for secondary, 0 for robustness | — |

---

## 3. Critical path, Phase 0 to Phase 4

```
P0 (verify, commit, pre-reg) ──► P1 data D1/D2 ──► P2 retrieval configs + E0 ──► P3 pools D1/D2 ──► P4 neighbours + judge + calibration ──► P4 GATE (freeze candidate)
        │                                                                                     ▲
        └──► P1-H HiAQ track: vetting → scrape → chunk → scenarios (humans) → survey launch ──┘ (the CAL-auth HiAQ half needs the early responses)
```

The **HiAQ survey** is still the long pole. It runs alongside P1–P4 and must launch by about the end of W3 so that D4 exists for P6.

---

## 4. Work packages, Phase 0 to Phase 4

⚙ = needs team approval before it starts (a download, or a compute-heavy run).

### Phase 0: Setup (zero downloads) → HUMAN GATE

| WP | What | Needs | Done when |
|---|---|---|---|
| P0.1 | ✅ Repo, venv (Python 3.11.15, torch 2.11 + cu128, all packages; **no ONNX Runtime**), core infrastructure, CLI, D-001–D-019 | — | done |
| P0.2 | **Verify the local models, no downloads.** `qwen3:4b`: 3 structured-JSON calls with `think=False`. Ollama `bge-m3`: embed 3 texts (EN, Hinglish, Devanagari) and check dim = 1024 and normalisation. **Reranker repair:** build `D:\ai-models\local\bge-reranker-v2-m3\` from **NTFS hard links** to the existing blobs (0 extra bytes, no download, the original cache untouched), then load it and score 2 pairs. Record digests and revisions in `env/models.json`. | P0.1 | `ha models verify` passes all three |
| P0.3 | **Mini throughput benchmark:** 20 judge-shaped calls on qwen3:4b (about 1 minute) and 200 bge-m3 embeddings. These project the P3 and P4 compute (§5) and confirm or adjust the pool size. | P0.2 | numbers in `reports/phase_0.md` |
| P0.4 | `ha doctor` all OK; `pytest -q` and `ruff check` clean; `env/freeze_<date>.txt`; first commit | P0.1 | committed |
| P0.5 | Pre-registration rewritten for the prototype (D-001 and D-017 adapted to D2 + D4); the team approves; tag `prereg-v1` | team | tagged |

### Phase 1: Data → HUMAN GATE (survey launched)

| WP | What | Needs | Done when |
|---|---|---|---|
| P1.1 ⚙ | Download D1 and D2 (~43 MB, datasets only) and convert them to `data/processed/<ds>/{corpus,queries_auth,qrels_auth}.parquet`. Check counts against the dataset cards; apply the 700-query cap. | approval | `ha data build fiqa` and `ha data build cqa_android` are idempotent |
| P1.2 | `splits.py`: R 40% / A 60% over 5 seeds, stratified; CAL-auth forced into R; seal registration | P1.1, P1.3 | tests: deterministic, disjoint, `force_R` honoured |
| P1.3 | `langid.py`: rules plus a romanised-Hindi lexicon (built from open wordlists, **no model**); CMI; the classes, with thresholds in config | H-09 | 30 hand-labelled test cases pass |
| P1-H1 | HiAQ source vetting (robots.txt and terms) | — | D-entries |
| P1-H2 | Polite scraper (at most 1 request every 2 s) and a heading-aware chunker; 1,500–4,000 chunks | P1-H1 | chunk file and log |
| P1-H3 | Scenario sheet (150 chunks) and the leakage checker | P1-H2 | `ha data hiaq check-scenarios` |
| P1-H4 | Form generator (Apps Script for 25 variants, redirect page), `survey_ingest.py` with PII stripping and language-class counts | P1-H3, H-05 | pilot export ingests cleanly |

### Phase 2: Retrieval configurations → AUTO GATE

| WP | What | Needs | Done when |
|---|---|---|---|
| P2.1 | Framework: a config interface; an embedding cache (fp16 `.npy` under `HA_CACHE`); **incremental** runs over query sets; `results/runs/<ds>/<config>/perquery.parquet`; a reranker pair-score cache | P1.1 | tests |
| P2.2 ⚙ | Encode the corpora with Ollama bge-m3: FiQA 57.6k and CQA 23k passages. P0.3 estimates the time. | P0.3, P2.1 | embeddings cached |
| P2.3 | Configurations C1–C7; `metrics.py` (tested against ranx) | P2.2 | tests |
| P2.4 | **E0 sanity:** C1 on FiQA gives 0.236 ± 0.02; C3 against MTEB is a warning only; latency on a 200-query sample | P2.3 | E0 in the registry |

### Phase 3: Synthetic pools → AUTO GATE

| WP | What | Needs | Done when |
|---|---|---|---|
| P3.1 | Generation engine: B.1 and B.2 templates, qwen3:4b-instruct-2507 (digest-checked) with `think=False`, temperature 0.7, top_p 0.9, a seed per call, at most 64 tokens. Cache keyed by (prompt, digest, params). Post-filters; dedupe at 0.95. | P0.2 | tests on the filters and dedupe |
| P3.2 ⚙ | Generate P-std and P-persona for D1 and D2: 800 passages each, about 1,600 queries per dataset. D4 follows when its corpus is final. P0.3 estimates the time. | P3.1, P1.1 | ≥ 90% of target; failures < 5% |

### Phase 4: Neighbours, judge and calibration → HUMAN GATE

| WP | What | Needs | Done when |
|---|---|---|---|
| P4.1 | Neighbours: bge-m3 dense top-30 from `R_seed`, reranked to a top-5 (pair cache across seeds); the features; near-duplicate at 0.97; **no self-neighbours** | P1.2, P3.2 | tests: no A query is ever a neighbour; no query is its own neighbour |
| P4.2 | Judge: the `JudgeOutput` schema; B.4 prompt v0; permutations seeded by `hash(query_id, perm_idx)`; retry once; SQLite cache; the **M1-iso** variant (no neighbour block); aggregation (gate plus NNLS) | P4.1 | tests with a mocked judge |
| P4.3 | Calibration set: 200 items (CAL-auth 30 FiQA + 30 HiAQ; 140 synthetic from qwen3 pools), split 120 `cal-dev` / 80 `cal-test` (sealed); spreadsheet (CSV) export (D-050; replaces Label Studio); `docs/annotation_guidelines.md` | P3.2, H-07 | export file |
| P4.4 | After H-08: κ per dimension; judge v0–v4 on `cal-dev` only; Spearman ρ; NNLS weights with 5-fold CV; FiQA-only weights row | H-08 | report |
| P4.5 | Freeze candidate (`FROZEN.json`: prompt hash, qwen3:4b-instruct-2507 digest, weights, gate); unseal `cal_test` once; report ρ and κ | P4.4 | **P4 HUMAN GATE** |

### Phases 5–8 (outline; detailed at the P4 gate)

- **P5 · Development and freeze.**
  - ⚙ Bulk-judge U for D1 and D2 over 5 seeds, plus the M1-iso pass (`ha judge <ds>`). D2's synthetic pool is judged before the freeze (prereg §3.3). The judge is prompt v0 with equal weights (A3, D-062).
  - Selection (strata, quotas, dedupe), B0, B1, M1-iso, M1.
  - Evaluation code (τ_b, τ_AP, MAE, gap measures, bootstrap).
  - Dev results on D1; choose the final configuration.
  - **FREEZE GATE**, then unseal `heldout_A`.
- **P6 · Held-out evaluation.**
  - D4 finalised: survey closed, LID with a 20% human check, the full qrels protocol (D-005), the R/A split.
  - ⚙ Judge D2 and D4.
  - Run C1–C7 on every set; primary analysis; per-language analysis on D4.
- **P7 · Optional robustness**, from §1, as approved.
- **P8 · Paper.** Tables and figures from the registry only; limitations (self-preference, sparse qrels, 7 configurations, elicited queries); novelty check #2; dashboard if time allows.

---

## 5. Compute budget (all local, RTX 3050 6 GB; P0.3 replaces the estimates with measurements)

| Block | Size | Estimate |
|---|---|---|
| Corpus encoding (Ollama bge-m3) | ~81k passages (D1 + D2) | ~20–60 min, once |
| Generation (qwen3:4b-instruct-2507) | ~1,600 × 3 datasets | **~0.5 h** (measured 0.54 s per call) |
| Judge, M1 (primary) | ~1,500 × 2.2 permutations × 5 seeds × 3 datasets ≈ 50k calls | **~22–28 h** (measured 1.60 s median, 2.03 s mean per call; overnight batches; the cache makes it resumable) |
| Judge, M1-iso | ~1,500 × 3 datasets (no neighbours, so independent of seed) | ~2 h |
| Calibration iterations | 200 × up to 5 versions × 2.2 | ~1.5 h |
| Neighbour reranking | short query–query pairs, cached across seeds | < 1 h |
| Retrieval reranking (C5–C7) | ~2,300 queries × 50 × 3 per dataset, cached | ~1–2 h per dataset |

**Measured (P0.3, D-028): 1.60 s median per judge call, well under the 4 s threshold.** If later prompt versions push it above 4 s, fall back in this order and log each step:
1. Pool of 1,200 instead of 1,600.
2. Seeds 0–2 for D2 and D4 as well; the primary result is then labelled as 3 seeds.
3. Optional Kaggle offload using the same Ollama digest.

**Storage:** no new models. Datasets are ~43 MB (D1 and D2). Embeddings are ~0.3 GB. All of it is on D:.

---

## 6. Human tasks

| ID | Task | When |
|---|---|---|
| H-01 | Add the literature-survey xlsx | P0 |
| H-02 | *(Only if Kaggle is ever needed)* Kaggle account | later |
| H-03 | Private GitHub remote | P0 |
| H-04 | Ethics approval from the faculty advisor, before the survey | P0–P1 |
| H-05 | Write 150 scenarios; pilot the survey with 5 people; approve and host the redirect page | P1 |
| H-06 | Distribute the survey (≥100 respondents, including Hindi speakers) | P1–P5 |
| H-07 | Flag the first ~150 responses (the CAL-auth HiAQ half) | ~P4 |
| H-08 | Calibration annotation of 200 items (~5–6 h each) | P4 |
| H-09 | 30 LID test examples | P1 |
| H-10, H-11 | D4 LID check (20%); full qrels verification (~10–12 h each) | P6 |
| — | Gate sign-offs; download and compute approvals (⚙) | throughout |

---

## 7. Open questions for the team (asked at the P0 gate)

1. **H2 with 2 held-out datasets.** Test the pooled Δτ_b across D2 and D4 (a single one-sided paired bootstrap, per-dataset results reported), or require M1 to win on both?
2. **HiAQ in the core?** Recommended, since it is the only code-mixed authentic data. It depends on the survey and costs the most human hours.
3. **Downloads to approve now:** D1 and D2 datasets only (~43 MB). No models.

---

## 8. Test map

| Area | Tests |
|---|---|
| Seal; registry; cache | P0 ✅ |
| Splits (deterministic, disjoint, `force_R`) | P1.2 |
| LID and CMI (30 cases) | P1.3 |
| PII stripping | P1-H4 |
| Metrics against ranx | P2.3 |
| Generation filters and dedupe | P3.1 |
| Neighbour rules (no A queries, no self-neighbours) | P4.1 |
| Judge schema, permutation protocol, aggregation | P4.2 |
| τ_AP worked example; bootstrap known case; quotas | P5 |

---

## 9. Status tracker

| WP | Status | WP | Status | WP | Status |
|---|---|---|---|---|---|
| P0.1 | done | P1-H1 | done (D-031) | P3.1 | done |
| P0.2 | done | P1-H2 | done (1,572 chunks) | P3.2 ⚙ | done (gate partial fail; D-048) |
| P0.3 | done (D-028 open) | P1-H3 | done; scenarios approved + official sheet (D-052); §6.3 exceptions S090/S091 (D-053) | P4.1 | done D1/D2; D4 blocked |
| P0.4 | done | P1-H4 | variants built (D-053); forms + pilot pending (team) | P4.2 | engine + pilot done; P5.1 bulk runner built |
| P0.5 | done (prereg-v1) | P2.1 | done | P4.3 | 100-item A2 pack exported 2026-10-14; unannotated (A3) |
| P1.1 ⚙ | done | P2.2 ⚙ | done (D1, D2 corpus) | P4.4 | not executed (A3, D-062) |
| P1.2 | done D1/D2; D4 after survey | P2.3 | done | P4.5 | not executed (A3); the freeze is at P5 |
| P1.3 | done; LID audit labels final (D-052) | P2.4 | done: E0 gate passed | P5–P8 | done: freeze signed (D-066), held-out run, exploratory work (D-067, D-068); paper draft in `paper/` |
