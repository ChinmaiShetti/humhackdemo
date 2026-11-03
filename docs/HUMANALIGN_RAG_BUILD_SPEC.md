# HumanAlign-RAG — Build Specification for Claude Code

**Project:** HumanAlign-RAG: Neighbour-Grounded Query Judging for Human-Aligned Synthetic Evaluation of Multilingual and Code-Mixed Retrieval
**Course context:** RVCE, Artificial Intelligence and Machine Learning (IS353IA). Two-member team.
**Goal:** a complete, reproducible research codebase plus a paper draft at the level of a SIGIR / CIKM / ACL-Findings submission.
**Spec version:** 1.0 (2026-09-26)

> **Scope amendment, 2026-09-27 (D-020 to D-026).** This project is built **prototype-first**. The methodology below still applies, but **what is built now** is defined by [PLAN.md](PLAN.md) v2 §1 (CORE PROTOTYPE / OPTIONAL ROBUSTNESS / FULL-SCALE) and by [decisions.md](decisions.md). In short:
> - the model stack is locked to `qwen3:4b` (generator and judge), Ollama `bge-m3` and `bge-reranker-v2-m3`, and **no model is downloaded without team approval**;
> - 7 retrieval configurations replace the 20-system pool, with no system split;
> - the core datasets are D1 (dev), D2 and D4 (held-out); D3 is optional.
>
> Where this document and PLAN.md or decisions.md disagree, the latter win: §4.4, §7 (pool size and generators), §11 (the core methods are B0, B1, M1-iso and M1), §12, §14 (the experiment tiers) and §16 (optional).

---

## 0. How to use this document (read first, Claude Code)

You are the engineering and analysis partner for this project. Read the entire document before writing any code. Then:

1. Create the repository described in §5 and copy **Appendix A** into `CLAUDE.md` at the repo root.
2. Work through the phases in §19 **in order**. Each phase ends with a gate. At a **HUMAN GATE**, stop, summarise what you did in `reports/phase_N.md`, and wait for the team. At an **AUTO GATE**, run the listed checks and continue only if all pass.
3. Every design choice that is not fixed here must be written down in `docs/decisions.md` (date, choice, reason, alternatives).
4. Keep a running `docs/lab_notebook.md`: one entry per work session, recording what was run (with `run_id`s), what was learned and what comes next.

### 0.1 Research-integrity rules (non-negotiable)

- **Never invent or estimate a number** that appears in a table, figure, README or paper. Every reported number must come from `results/registry.jsonl` and carry a `run_id`. The paper build script pulls numbers from the registry; nothing is typed by hand.
- **Held-out data is sealed.** Until the FREEZE GATE (§19, Phase 5) you must not compute any metric on these:
  - the authentic test sets `A` of held-out datasets
  - the held-out system half (§12)
  - the calibration test split (`cal-test`)

  The loader enforces this through `HA_UNSEAL=0` (see §20.3).
- **No tuning after freeze.** After the judge prompt, weights and thresholds are frozen, their hashes are stored in `results/FROZEN.json`. Any later change starts a new version with the suffix `-v2` and must be reported as such.
- **Negative results are reported.** If a simple baseline matches or beats the method, that is a finding; do not hide it or re-engineer until it disappears.
- **No personal data.** The survey collects no names, emails, phone numbers or roll numbers. Strip anything that looks like PII before storing (regex for phone, email, Aadhaar-like 12-digit numbers, PAN pattern).
- **Scraping etiquette.** Honour `robots.txt` and site terms; rate-limit to at most 1 request per 2 s; store the URL and fetch date for every page. If a site forbids automated access, record it in `docs/decisions.md` and skip it.
- **Cite, don't copy.** Paraphrase prior work; quote only short phrases, always with a citation.

### 0.2 When to ask the humans

Ask only for: survey distribution and collection, annotation work, credentials (Kaggle, Hugging Face), approval at HUMAN GATEs, and any irreversible action (deleting data, publishing a dataset). Otherwise decide, log the decision, and continue.

---

## 1. Project summary

RAG and search systems are usually benchmarked on **synthetic queries**: an LLM reads document chunks and writes questions about them. Recent evidence shows these queries differ systematically from what users type, and the difference changes which systems look best:

- **Kucia & Gawlik (CIKM '26).** 322 authentic student queries averaged 6.8 words against 15.7 for 1,851 synthetic ones, and drew on 53 source documents against 165. The hybrid retriever that led on every synthetic metric did not lead on authentic queries, and cost up to 8× the latency. [1]
- **Hussain & Nielbo (2026).** In a production encyclopedia, synthetic queries implied that over 90% of queries needed LLM augmentation, while only 27.8% of real queries did. [2]
- **Rahmani et al. (CIKM '25).** LLM-generated queries are longer than human ones, and LLM judges inflate relevance labels. [3]

**HumanAlign-RAG** curates a synthetic query pool so that the resulting benchmark *behaves* like authentic traffic. For each synthetic query:

1. retrieve the most similar **authentic** queries from a reference pool (BGE-M3, then BGE-reranker-v2-m3);
2. an LLM judge rates the synthetic query against those neighbours on five dimensions: fidelity, human-likeness, specificity leakage, need plausibility and code-mix naturalness;
3. a stratified selector builds a benchmark of fixed size.

The benchmark is validated by **system-ranking agreement** (Kendall τ_b and τ_AP) with a held-out authentic set, across a pool of about 20 retrieval systems, four datasets and three language conditions: English, Hindi, and Romanised Hindi / Hinglish.

We also release **HiAQ-Services**, a new set of scenario-elicited authentic queries (English, Romanised Hindi and Hinglish) over public Indian digital-service help documentation, with relevance labels.

---

## 2. Research questions and pre-registered hypotheses

Write these into `docs/preregistration.md` in Phase 0, commit, and do not edit them afterwards; append amendments only, with dates.

| RQ | Question | Hypothesis | Primary test |
|---|---|---|---|
| RQ1 | How large is the synthetic–authentic gap across domains, generators and languages? | H1: Raw synthetic sets differ significantly from authentic sets in query length, lexical overlap with the gold passage and topic concentration, and give τ_b < 0.8 with authentic system rankings on ≥2 of 4 datasets. | Two-sample KS / Wasserstein with bootstrap CI; τ_b with bootstrap CI |
| RQ2 | Does neighbour-grounded judging select synthetic subsets that rank systems more like authentic queries? | H2: HumanAlign-Filter (M1) achieves higher τ_b than raw synthetic (B0) and random subsets (B1) on ≥3 of 4 held-out datasets, and is at least as good as the best non-LLM baseline (B2–B7) on average. | Paired bootstrap on Δτ_b (1,000 resamples), Holm correction across datasets |
| RQ3 | Which components matter? | H3: Removing the neighbours (isolated judge) or the fidelity gate reduces τ_b; the P (need plausibility) and S (specificity) dimensions contribute more than H alone. | Ablation table, paired bootstrap vs the full M1 |
| RQ4 | Is filtering, rewriting ("humanising") or exemplar-guided generation best at a fixed budget? | H4: Filter + few-shot pool (M3) ≥ Filter (M1) > Rewrite (M2) on τ_b; rewriting improves surface realism but risks intent drift. | Same as RQ2 |
| RQ5 | Does code-mixing change the picture? | H5: The raw synthetic gap is largest for Romanised Hindi / Hinglish queries, and M1 reduces it the most there in absolute terms. | Per-language τ_b and gap G with CIs |
| RQ6 | Can the judge be distilled into a small encoder? | H6: A MuRIL-base or mE5-small regressor trained on judge labels keeps ≥90% of M1's τ_b gain over B0, at <5% of the inference cost. | Relative gain, cost per 1k queries |

**Primary endpoint.** τ_b between system rankings by nDCG@10 on the curated set and on the authentic test set `A`, computed over the **held-out system half**, on **held-out datasets**. Everything else is secondary.

---

## 3. Novelty position and related work map

### 3.1 What exists

| Work | What it does | What it does not do (our gap) |
|---|---|---|
| Kucia & Gawlik, CIKM '26 [1] | Measures synthetic vs authentic divergence for one university chatbot; shows ranking inversion | No method to fix the gap; English/Polish only; 3 configs |
| Hussain & Nielbo, 2026 [2] | Production evidence that synthetic queries mislead routing and augmentation decisions | Diagnosis only |
| van Elburg et al., SynDAiTE @ ECML-PKDD 2025 [4] | Synthetic QA ranks retriever configurations reliably but not generators | No curation; no multilingual; no per-query realism |
| Rahmani et al., SIGIR '24 [5]; CIKM '25 [3] | Fully synthetic test collections; bias of synthetic data | Validates generation; no neighbour-grounded selection |
| He et al., SIGIR '25 [6] | Validates LLM-elicited tip-of-tongue queries by system-rank correlation with CQA queries | Better prompting for one query type; no judge or selection |
| Alaofi et al., SIGIR '23 [7]; ICTIR '25 [8] | LLM query variants vs human variants; persona-based variants expose unequal service | Generation-time; no authentic-grounded filter |
| MAPLE-Synth (Wei et al., 2026) [9] | Retrieval-based in-context generation gives human-level realism | Grounding at generation time; scientific search only; no ranking-agreement curation |
| Chroma Generative Benchmarking, 2025 [10] | Example-steered generation with an aligned judge for document filtering | Filters documents, not queries; no ranking validation |
| Vardasbi et al., SIGIR '26 [11] | Grounds LLM relevance judges in historical behaviour of similar queries | Judges relevance, not query realism; the closest analogue to our neighbour grounding |
| Guiver, Mizzaro & Robertson, TOIS 2009 [12] | Selecting topic subsets that preserve system rankings | Needs system runs on the real topics (oracle-like); not about synthetic realism |
| InPars [13], Promptagator [14], RAGAS [15] | Standard synthetic generation and filtering by answerability or round-trip consistency | Filter for usefulness, not human realism |
| Zeng et al., Findings ACL '26 (CSR-L) [16]; Hindi-BEIR [17]; MiLQ [18] | Code-switched queries degrade retrieval (up to 27%) | Chinese/Japanese/Korean–English or native Hindi; no Hinglish; no synthetic evaluation question |

### 3.2 Our claimed contributions (keep claims exactly this narrow)

- **C1.** A *query-level, neighbour-grounded* realism judge used to **curate** synthetic evaluation sets, validated by **system-ranking agreement with held-out authentic queries** across a pool of about 20 systems. To our knowledge, prior work either diagnoses the gap, improves generation, or judges relevance.
- **C2.** A controlled comparison of *filtering vs rewriting vs exemplar-guided generation* for closing the synthetic–authentic gap, at matched budgets.
- **C3.** HiAQ-Services: the first public scenario-elicited English / Romanised-Hindi / Hinglish authentic query set for RAG evaluation over Indian public-service documentation, with a cross-lingual gap analysis.
- **C4.** A distilled realism scorer that makes curation cheap enough for routine use.

### 3.3 Novelty re-check (mandatory, twice)

Run in Phase 0 and again before writing the paper. Search arXiv, the ACL Anthology, the ACM DL and Semantic Scholar for:

- "synthetic queries" realism judge
- "authentic queries" synthetic RAG evaluation
- query realism LLM-as-a-judge
- synthetic test collection curation "system ranking"
- Hinglish OR code-mixed retrieval evaluation synthetic
- humanize synthetic queries
- behaviour-grounded judge
- benchmark validity synthetic queries

Record every hit from 2025 onwards in `docs/novelty_check.md` with a one-line "overlap / difference" note. If something overlaps C1–C4, stop and raise it at the next HUMAN GATE with a proposal to re-scope.

---

## 4. Environment and hardware

### 4.1 Machine

- **Primary:** Windows 11 laptop/desktop, NVIDIA RTX 3050 (6 GB VRAM). The secondary drive `D:` is used for models and caches.
- **Bulk compute:** Kaggle Notebooks (2× T4 or P100) or Google Colab (T4), both Linux. Use these for:
  - encoding large corpora with the large embedders
  - running the 9B judge
  - any vLLM batch job (vLLM does not run on Windows)
- **Everything is open-weight and local.** No paid APIs.

### 4.2 Paths (set once in `.env`; never hard-code)

```
HA_ROOT=D:\humanalign-rag
HF_HOME=D:\hf_cache
OLLAMA_MODELS=D:\ollama\models          # set as a Windows user env var before starting Ollama
HA_DATA=D:\humanalign-rag\data
HA_CACHE=D:\humanalign-rag\cache
HA_UNSEAL=0
PYTHONUTF8=1                            # needed for Devanagari I/O on Windows
```

If the team already has a different folder layout, ask once at the Phase 0 HUMAN GATE and update `.env`; code must read everything from `.env` or `configs/`.

### 4.3 Software

- **Python and ML stack:**
  - Python 3.11, in a `venv` (not conda, to keep it simple on Windows).
  - PyTorch with CUDA from the official index URL that matches the installed driver. Check it with `torch.cuda.is_available()`.
- **Retrieval and evaluation libraries:**
  - `FlagEmbedding` (BGE-M3, bge-reranker-v2-m3)
  - `sentence-transformers`
  - `faiss-cpu` (the GPU build is not on Windows pip; exact inner-product search is fine at our corpus sizes)
  - `bm25s`
  - `ranx` (IR metrics and fusion)
  - `ir_datasets`
  - `datasets`
  - `scipy`, `scikit-learn`, `pandas`, `pyarrow`, `numpy`
- **LLM and structured output:** `ollama` (Python client), `pydantic` v2 (JSON schemas).
- **Language processing:**
  - `ai4bharat-transliteration` (IndicXlit), or a documented fallback if it fails to install on Windows
  - `fasttext` language ID (`lid.176.bin`)
  - `regex`
- **Tooling and output:**
  - `typer` (CLI), `loguru`, `rich`
  - `pytest`, `ruff`
  - `streamlit`, `plotly`
  - `matplotlib`, `seaborn` (paper figures only)
  - `jinja2` (paper table rendering)
- **Avoid ONNX Runtime.** Its native DLL fails to load on this machine (known issue from a previous project). Use pure PyTorch.
- **Reproducibility:** pin versions in `requirements.txt` after the first working install and record the output of `pip freeze` in `env/freeze_<date>.txt`.

### 4.4 Models (pull or download in Phase 0; verify each loads)

| Role | Model | Where | Approx. size |
|---|---|---|---|
| Judge (primary) | `qwen3.5:4b` (Ollama); call with thinking disabled (`think=False`) | local | 3.4 GB |
| Judge (fallback, if 3.5 misbehaves on JSON) | `qwen3:4b` | local | ~2.6 GB |
| Judge (scale ablation) | `qwen3.5:2b`, `qwen3.5:9b` | local / Kaggle | 2.7 / 6.6 GB |
| Judge (cross-family ablation) | `gemma3:4b` | local | ~3.3 GB |
| Generator G1 | `llama3.1:8b` | local (partial offload) or Kaggle | 4.9 GB |
| Generator G2 | `gemma3:4b` | local | ~3.3 GB |
| Generator G0 | docT5query queries released with BEIR (`BeIR/<name>-generated-queries` on HF) | download | — |
| Embedder / neighbours | `BAAI/bge-m3` | local | 2.3 GB (FP16) |
| Reranker | `BAAI/bge-reranker-v2-m3` | local | 2.3 GB |
| System-pool models | see §12 | local / Kaggle | — |
| Distillation backbones | `google/muril-base-cased`, `intfloat/multilingual-e5-small` | local | — |

**VRAM rule:** never keep Ollama and a PyTorch model on the GPU at the same time on the 3050.

- Set `OLLAMA_KEEP_ALIVE=0` during PyTorch phases.
- Run the judge and the retrieval as separate CLI steps.

Verify every model tag on the day of pulling. If a tag no longer exists, choose the nearest current equivalent and log it in `docs/decisions.md`.

---

## 5. Repository layout

```
humanalign-rag/
  CLAUDE.md                     # Appendix A
  README.md                     # how to reproduce every table/figure
  .env.example
  requirements.txt
  configs/
    datasets/{fiqa,cqa_android,miracl_hi,hiaq}.yaml
    generators.yaml             # G0, G1, G2 × styles
    judge.yaml                  # model, prompt version, dims, gate, weights path
    systems.yaml                # system pool definitions + dev/held-out split
    selection.yaml              # budgets, strata, dedupe thresholds
    experiments/E0.yaml ... E11.yaml
  src/ha/
    __init__.py
    cli.py                      # typer app: `ha <command>`
    paths.py  seal.py  registry.py  cache.py  utils.py
    data/
      public.py                 # ir_datasets / HF loaders, corpus subsampling
      hiaq_scrape.py            # polite scraper for help pages
      hiaq_chunk.py             # heading-aware chunking
      scenarios.py              # scenario authoring helpers + leakage check
      survey_ingest.py          # Google Form CSV → clean queries
      langid.py                 # token-level LID, CMI, language class
      qrels.py                  # pooling + LLM pre-labels + human verification merge
      splits.py                 # R/A splits (5 seeds), cal-dev/cal-test
    gen/
      prompts/                  # versioned .j2 templates (Appendix B)
      generate.py               # G × style × passages → pool
    neighbours/
      index.py  retrieve.py     # BGE-M3 on R pool + rerank → top-5
    judge/
      schema.py                 # pydantic JudgeOutput
      prompt.py                 # render with shuffled neighbours
      run.py                    # batched Ollama calls, caching, permutations
      aggregate.py              # gate + learned weights
      calibrate.py              # human vs judge analysis
    select/
      strata.py  select.py  baselines.py  rewrite.py  oracle.py
    systems/
      sparse.py  dense.py  hybrid.py  rerank.py  pool.py
      run_all.py                # per-query run files for every (system, query pool)
    eval/
      metrics.py                # per-query nDCG@10, MRR@10, Hit@k, R@100
      agreement.py              # tau_b, tau_AP, top-k overlap, winner agreement
      gap.py                    # distributional distances, composite G
      bootstrap.py              # paired bootstrap, Holm correction
    distill/
      train.py  predict.py
    human/
      export_tasks.py  import_labels.py  agreement.py   # Label Studio I/O
    report/
      tables.py  figures.py  paper_numbers.py
  app/dashboard.py              # Streamlit
  paper/
    main.tex  sections/*.tex  figures/  tables/  refs.bib
  data/        (git-ignored; see data cards)
  cache/       (git-ignored)
  results/
    registry.jsonl  FROZEN.json  runs/  tables/  figures/
  docs/
    preregistration.md  decisions.md  lab_notebook.md  novelty_check.md
    annotation_guidelines.md  survey_protocol.md  data_cards/
  reports/phase_*.md
  tests/
```

---

## 6. Data

### 6.1 Datasets

Four datasets span a spectrum of authenticity, from organic to elicited. The spectrum is itself part of the analysis.

| ID | Dataset | Domain | Authentic queries | Authenticity type | Corpus handling | Role |
|---|---|---|---|---|---|---|
| D1 | BEIR FiQA-2018 (`beir/fiqa`, test split) | personal finance Q&A | ~648 test questions from StackExchange / Reddit users | organic | full corpus (~57.6k) | **development dataset** (all design decisions) |
| D2 | BEIR CQADupStack-Android (`beir/cqadupstack/android`) | device support | 699 StackExchange question titles | organic | full corpus (~23k) | held-out |
| D3 | MIRACL-Hindi (dev) | Wikipedia, Hindi | native-speaker questions | prompt-elicited (annotators saw a text prompt) | subsample (see below) | held-out; Hindi condition |
| D4 | **HiAQ-Services** (new) | Indian digital public services help | survey queries: EN, Romanised Hindi, Hinglish | scenario-elicited | full (small) | held-out; code-mix condition |

- **Read counts at load time.** Assert them against the numbers in the dataset cards and log them; do not hard-code.
- **MIRACL-Hindi corpus subsample.** Keep:
  - all passages in the qrels for the chosen queries;
  - for each authentic query, the BM25 top-100 and BGE-M3 top-100 passages;
  - 50,000 random passages;
  - the source passages of all synthetic queries (see §7).

  Record the passage IDs and the exact procedure; report this in the paper's limitations.
- **Query counts.** If a dataset has more than 700 authentic queries, sample 700 with a fixed seed. This keeps compute bounded; log it.

### 6.2 HiAQ-Services: corpus

- **Sources.** Public help, FAQ and user-guide pages of Indian digital services, for example DigiLocker, UPI/BHIM (NPCI), IRCTC, the Income-Tax e-filing portal FAQs and the mAadhaar / UIDAI FAQs. Check each site's terms and `robots.txt` first; log inclusion or exclusion.
- **Chunking.** Chunk by heading: each chunk is one FAQ question + answer, or one section of at most 350 tokens with a 50-token overlap. Keep `doc_id`, `chunk_id`, `url`, `heading_path` and `fetched_at`.
- **Target size.** 1,500–4,000 chunks. Deduplicate near-identical chunks (MinHash or cosine > 0.95).

### 6.3 HiAQ-Services: scenario-elicited authentic queries

This follows the UQV100 backstory method [19]: people read a situation, not the document, and type what they would search.

1. **Scenario authoring (humans; Claude Code assists).**
   - Sample 150 chunks, stratified across services and topics.
   - For each, a team member writes a one- or two-sentence situation in everyday English. Example: *"You try to download your marksheet from DigiLocker but it keeps saying the document isn't available."*
   - **Leakage check (automatic).** A scenario must not contain rare document terms. Reject it if any token with document-frequency rank in the top 5% rarest of the corpus vocabulary appears in both the scenario and the chunk, or if BM25(scenario → chunk) ranks the chunk first with a margin > 2× over the second. Rewrite rejected scenarios by hand.
2. **Survey (Google Form).** The text is in Appendix D.
   - Each respondent gets 6 random scenarios, using a form with 25 variants of 6 scenarios each.
   - Instruction: *type exactly what you would type into the app's help search or a chatbot, in whatever language or script you normally use.*
   - **Do not mention Hinglish.** Mixing must be natural, not prompted.
   - Optional fields: languages comfortable typing in; phone or laptop. No identifiers.
   - Target at least 100 respondents (≥ 600 queries) across RVCE departments and non-student contacts (family, staff), to widen age range and typing habits.
3. **Cleaning.**
   - Strip PII. Drop empty or off-topic answers; record reasons.
   - Keep typos. Only normalise whitespace.
4. **Language labelling** (§6.5): token-level LID, then a Code-Mixing Index, then the class `EN`, `ROM_HI` (Romanised Hindi, mostly Hindi tokens), `HINGLISH` (mixed), `DEV_HI` (Devanagari) or `OTHER`. A human verifies a 20% sample; report the agreement.
5. **Relevance labels (qrels).**
   - The scenario's source chunk is relevance grade 2 by construction.
   - **Pooling:** take the top-10 from four diverse systems (BM25, BGE-M3 dense, mE5-base, hybrid) for every authentic query. Get LLM pre-labels (`qwen3.5:4b`, graded 0/1/2, with the scenario as context), then human verification of all LLM-positive items and a 10% sample of LLM-negative items. Report LLM–human agreement.
   - If the time budget does not allow this, pool-label a random 30% of queries and report results on both "source-only" and "pooled" qrels for D4.

### 6.4 Splits (`src/ha/data/splits.py`)

- **Authentic reference pool and test set.** For every dataset, split authentic queries into a reference pool **R (40%)** and a test set **A (60%)**, stratified by language class (and by topic cluster, if more than 30 queries). Use **5 seeds**; every experiment reports mean ± 95% CI over seeds *and* bootstrap.
  - R is the only authentic data the judge, the neighbours, the few-shot generator, the kNN baseline, the discriminator baseline and the strata may see.
  - A is used only for evaluation.
- **Calibration set.** 200 items, drawn from FiQA and HiAQ R pools plus synthetic pools, and made up of 140 synthetic and 60 authentic queries. Split it 120 `cal-dev` / 80 `cal-test` (sealed).
- **System pool.** Split into dev systems and held-out systems (§12).

### 6.5 Language processing (`src/ha/data/langid.py`)

- **Token LID.**
  - Devanagari script → `hi`.
  - Latin script → a rule-based / dictionary classifier (English wordlist + romanised-Hindi lexicon). Build the lexicon from IndicXlit back-transliteration of a Hindi frequency list plus common chat spellings: *nahi, nahin, kya, kaise, hai, ho, gaya, karu, mera, kyun, …*
  - fastText LID is a tiebreaker only. Keep named entities and service names (DigiLocker, UPI, IRCTC, OTP) as `ne`, excluded from CMI.
- **CMI** (Das & Gambäck style): CMI = 100 × (1 − max_lang_tokens / (N − ne_tokens)), with 0 when there are no language tokens.
- **Classes:**
  - `EN` if the share of Hindi tokens is below 0.1
  - `ROM_HI` if the Hindi share is above 0.8 and the text is in Latin script
  - `HINGLISH` if the Hindi share is between 0.1 and 0.8
  - `DEV_HI` if the text is in Devanagari script

  Thresholds go in the config; do a sensitivity check at ±0.05.
- **Unit tests** with 30 hand-labelled examples (write these in Phase 1).

### 6.6 Data cards

Write `docs/data_cards/<dataset>.md` for each dataset, covering: source, licence, size, how the queries were created, known biases, the subsampling applied and the intended use. The HiAQ card must follow the Datasheets for Datasets headings.

---

## 7. Synthetic query pools (`src/ha/gen/`)

For every dataset, generate one pool with the following design.

- **Source passages.** Sample 1,500 passages uniformly at random from the corpus with a fixed seed. This mirrors standard practice: generators do not know which passages are popular. For D3, sample from the subsampled corpus. Store `source_pid` as the single relevant passage (grade 1) of each synthetic query, as RAGAS-style pipelines do; this sparse-qrel property is part of the "raw synthetic" condition and is analysed in E11.
- **Generator × style grid:**

| Pool | Generator | Style | Per passage | Approx. queries |
|---|---|---|---|---|
| P-std | G1 llama3.1:8b | S-std (RAGAS-like, Appendix B.1) | 1 | 1,500 |
| P-persona | G1 | S-persona (Appendix B.2; 5 personas, with the Hinglish persona only for D3/D4) | 1 random persona | 1,500 |
| P-fewshot | G1 | S-fewshot, 5 exemplars sampled from **R** (Appendix B.3) | 1 | 1,500 |
| P-std-G2 | G2 gemma3:4b | S-std | 1 | 1,500 (generator-sensitivity check) |
| P-docT5 | G0 docT5query | released | as released, 1 per passage | ≤1,500 (D1, D2 only) |

- **Decoding.** Temperature 0.7, top_p 0.9, fixed seed per call, max 64 new tokens. Strip quotes and numbering. Drop queries shorter than 2 tokens or with more than 60% token overlap with the passage's first sentence (a pure copy).
- **Deduplication.** Within a pool, drop pairs with cosine > 0.95 (BGE-M3).
- **The main experiments use the union pool `U = P-std ∪ P-persona`** (≈3,000 queries), which represents a realistic "generate a lot" pipeline. P-fewshot is used for M3 and B6. P-std-G2 and P-docT5 are used for the generator-sensitivity analysis (E1, E9).
- **Cache** every generation call by a hash of (prompt, model, params). Record the model digest from `ollama show`.

---

## 8. Neighbour retrieval (`src/ha/neighbours/`)

1. Encode R with BGE-M3 dense (FP16, normalised).
2. For each synthetic query q, take the top-30 cosine neighbours in R, rerank them with bge-reranker-v2-m3 and keep the top-5.
3. Store the neighbour IDs, the dense and rerank scores, and these features:
   - `nn_max_cos`: highest neighbour cosine
   - `nn_mean5_cos`: mean cosine of the top-5 neighbours
   - `nn_lang_mix`: language classes of the neighbours
4. **Guardrail.** If `nn_max_cos` > 0.97, flag q as a near-duplicate of a real query. It is excluded from selection, to avoid any path by which authentic text enters the benchmark.

---

## 9. The HumanAlign judge (`src/ha/judge/`)

### 9.1 Dimensions (1–5 scale; anchors in Appendix B.4)

| Code | Dimension | Question the judge answers |
|---|---|---|
| F | Fidelity (gate) | Does the source passage answer this query? |
| H | Human-likeness | Do the length, wording, register, abbreviations, typos and detail level resemble the real neighbours? |
| S | Specificity leakage (reverse-scored; 5 = natural) | Does the query use only knowledge a user would have *before* reading the passage? |
| P | Need plausibility | Would many real users of this service have this need, or is it an obscure detail only a document reader would ask? |
| C | Code-mix naturalness | Only when the query or its neighbours are not pure English: does the mixing and spelling look like natural Indian typing? (null otherwise) |

P targets the **topic-concentration** finding in [1]: generators spread evenly over the corpus, while users concentrate on a few needs.

### 9.2 Prompt and structured output

- The prompt template is in Appendix B.4. The output schema (pydantic, passed to Ollama `format=`) is:

```python
class JudgeOutput(BaseModel):
    F: conint(ge=1, le=5)
    H: conint(ge=1, le=5)
    S: conint(ge=1, le=5)
    P: conint(ge=1, le=5)
    C: Optional[conint(ge=1, le=5)]
    keep: bool
    reason: Literal["ok","not_answerable","too_document_like","over_specific",
                    "implausible_need","unnatural_code_mix","near_duplicate_of_real"]
```

- **Decoding.** Temperature 0, fixed seed, `think=False`, `num_ctx` 4096. On a JSON validation failure, retry once with the error message appended, then mark the call as failed. Report the failure rate.
- **Position-bias control** [20]:
  1. Run 2 random permutations of the neighbour order.
  2. If any dimension differs by 2 or more between them, run a third permutation.
  3. Take the median per dimension.
  4. Log the disagreement rate.
- **Caching.** Key each call by a hash of (rendered prompt, model digest, decoding params), stored in SQLite `cache/judge.sqlite`.
- **Throughput.** Target 3–6 s per call for qwen3.5:4b on the 3050. Benchmark on 50 calls in Phase 4. If the total estimated time for all pools exceeds 40 GPU-hours, move bulk judging to Kaggle (Ollama or vLLM on Linux) with the identical prompt and model.

### 9.3 Aggregation (`aggregate.py`)

- **Gate:** if F < 4, the score s(q) = −∞ (rejected).
- **Score:** s(q) = w_H·H + w_S·S + w_P·P + w_C·C, where C is replaced by the mean of the other three when it is null.
- **Weights** are learned on **cal-dev** by non-negative least squares against the human "overall realism" rating (Appendix C), with 5-fold CV on cal-dev. Ablation: equal weights.
- `keep` from the judge is logged but **not** used for selection (the scores are). Report agreement between `keep` and the thresholded score.

### 9.4 Calibration (Phase 4, HUMAN GATE)

1. Export the 200 calibration items to Label Studio. The annotation guidelines are Appendix C. Both members rate all items on F, H, S, P, C and "overall realism" (1–5), blind to source (synthetic or authentic) and to the judge's output.
2. **Inter-annotator agreement.** Report quadratic-weighted Cohen's κ per dimension. If κ < 0.5 on any dimension, revise the guidelines and re-annotate 40 items; record the change.
3. **Adjudication.** Average the two ratings; discuss disagreements of 2 or more and settle them.
4. **Prompt iteration.** Use cal-dev only, with at most 5 prompt versions (v0–v4). The metric is mean Spearman ρ between the judge and the human average across dimensions. Keep the best version.
5. **Freeze.** Record the prompt hash, model digest, weights and gate threshold in `results/FROZEN.json`, then unseal cal-test once and report ρ and κ on it.

---

## 10. Selection (`src/ha/select/`)

Given a pool U, a budget N and the reference pool R:

1. **Strata.**
   - Cluster the R queries' BGE-M3 embeddings with k-means, k = min(10, |R|/15), fixed seed. The cluster proportions π_k are the target topic distribution.
   - For D3/D4, cross the clusters with the language class (EN / ROM_HI / HINGLISH / DEV_HI) and use the joint proportions from R.
   - Assign each synthetic query to its nearest centroid. For the language class, use its detected class.
2. **Quota.** n_k = round(N·π_k). If a stratum has too few candidates, redistribute its quota proportionally.
3. **Within a stratum,** rank candidates by s(q) and take them greedily. Skip a candidate if its cosine with an already-selected query exceeds 0.90 (MMR-style dedupe).
4. **Output:** a list of query IDs plus a selection log (score, stratum, reason for any skip).

**Budgets.**
- The main runs use N = |A| (size-matched to the authentic test set for each seed).
- The budget curve (E7) uses N ∈ {100, 200, 400, 800}, capped by what is available.

---

## 11. Methods, baselines and bounds

All methods produce size-matched sets from the same pools and see only R.

| ID | Name | Description |
|---|---|---|
| B0 | Raw | Random N from U without stratification (a random draw of the raw pool, to keep the size matched; repeated over 20 seeds) |
| B1 | Stratified-random | Strata quotas as §10, random within each stratum (isolates the effect of topic matching) |
| B2 | Length-matched | Sample from U to match R's query-length histogram (1-token bins), via rejection sampling |
| B3 | kNN-realism | Rank U by `nn_mean5_cos` (closeness to real queries); no LLM |
| B4 | Low-leakage | Rank U by −(BM25 score of q against its source passage, normalised by query length); no LLM |
| B5 | Discriminator | Train a classifier to tell R from U (mE5-small + logistic regression head, 5-fold; out-of-fold probabilities); select the queries it finds most human |
| B6 | Few-shot generation | P-fewshot pool, random N (exemplar-guided generation, as in Chroma / Promptagator) |
| B7 | Persona generation | P-persona pool, random N (Alaofi et al. 2025 style) |
| M1 | **HumanAlign-Filter** | Judge scores + gate + stratified selection over U |
| M2 | **HumanAlign-Rewrite** | Rewrite every query in U with the neighbours (Appendix B.5), re-judge, apply the gate, then select as M1. Reject a rewrite with cosine > 0.95 to any R query |
| M3 | **HumanAlign-Filter on few-shot pool** | M1 applied to U ∪ P-fewshot |
| O | Oracle (upper bound) | Greedy selection of N from U that maximises τ_b with **A** using system results. This peeks at A and exists only to show headroom; never compare it as a method |
| M1-d | Distilled filter | M1 with s(q) predicted by the distilled model (§16) |

Every stratified method (B1, M1, M2, M3) should also be reported **without** stratification in the ablation table, so the topic-matching effect is separated from the judge effect.

---

## 12. System pool (`configs/systems.yaml`, `src/ha/systems/`)

About 20 retrieval systems give the ranking-agreement statistics enough resolution.

| ID | Family | System |
|---|---|---|
| S1 | sparse | BM25 (bm25s, k1=0.9, b=0.4) |
| S2 | sparse | BM25 (k1=1.2, b=0.75) |
| S3 | sparse | BGE-M3 lexical weights (sparse mode) |
| V1 | dense | BAAI/bge-m3 (dense) |
| V2 | dense | intfloat/multilingual-e5-small (`query:` / `passage:` prefixes) |
| V3 | dense | intfloat/multilingual-e5-base |
| V4 | dense | intfloat/multilingual-e5-large |
| V5 | dense | intfloat/multilingual-e5-large-instruct (with a task instruction) |
| V6 | dense | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| V7 | dense | sentence-transformers/LaBSE |
| V8 | dense | Alibaba-NLP/gte-multilingual-base |
| V9 | dense | Snowflake/snowflake-arctic-embed-m-v2.0 |
| H1 | hybrid | RRF(S1, V1), k=60 |
| H2 | hybrid | RRF(S1, V3) |
| H3 | hybrid | BGE-M3 dense+sparse weighted (0.7 / 0.3) |
| R1 | rerank | V1 top-50 → bge-reranker-v2-m3 |
| R2 | rerank | S1 top-50 → bge-reranker-v2-m3 |
| R3 | rerank | S1 top-50 → cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 |
| R4 | rerank | V2 top-50 → cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 |
| R5 | rerank | H1 top-50 → bge-reranker-v2-m3 |

- **Dev / held-out split.** Deterministic and balanced by family:
  - **dev** = {S1, S3, V1, V3, V6, V8, H1, R1, R3, H3}
  - **held-out** = {S2, V2, V4, V5, V7, V9, H2, R2, R4, R5}
- **Efficiency.** Every curated set is a *subset* of a pool, or a rewrite of pool queries. So:
  - run every system **once** over (A ∪ R ∪ all pools ∪ M2 rewrites) per dataset;
  - store per-query metrics in `results/runs/<dataset>/<system>/perquery.parquet`;
  - compute every set-level metric by aggregating per-query rows. This makes the bootstrap cheap.
- **Sanity reproduction (E0).** BM25 on FiQA must reproduce the BEIR-reported nDCG@10 of 0.236 within ±0.02. Where a model card reports an nDCG@10 for FiQA, the dense model must match it within ±0.03. If not, fix preprocessing before going further.
- **Compute estimates** go in `reports/phase_2.md`: encoding time per model per corpus on the 3050 vs Kaggle. Encode V4 (e5-large) and V5 (e5-large-instruct) on Kaggle when the local ETA exceeds 2 h.

---

## 13. Metrics and statistics (`src/ha/eval/`)

### 13.1 Per-query retrieval metrics

nDCG@10 (primary), MRR@10, Hit@1, Hit@5 and Recall@100, computed with `ranx` against the qrels of each query set: official qrels for authentic sets, `source_pid` for synthetic sets. Also record latency per query for each system, measured on the 3050, as the median of 3 runs.

### 13.2 Set-level metric

M_s(X) is the mean per-query metric of system s over query set X.

### 13.3 Agreement between set X and authentic test set A (over a system set 𝒮)

- **τ_b:** Kendall τ_b between (M_s(X))_{s∈𝒮} and (M_s(A))_{s∈𝒮}. This is the **primary** measure.
- **τ_AP:** AP rank correlation [21], which weights disagreements near the top more heavily. Implement it from the paper and unit-test it against a worked example; A is the reference ranking.
- **Winner agreement:** 1 if argmax_s M_s(X) = argmax_s M_s(A).
- **Top-3 overlap:** |top3(X) ∩ top3(A)| / 3.
- **Absolute gap:** MAE_𝒮 = mean_s |M_s(X) − M_s(A)|, plus its signed mean (the optimism bias).

### 13.4 Distributional gap (system-free)

- **Length:** W1 distance between the token-length distributions of X and A.
- **Leakage:** W1 between distributions of normalised lexical overlap of each query with its gold passage (the share of query tokens found in the gold passage).
- **Difficulty:** W1 between per-query difficulty distributions, where difficulty(q) = 1 − mean_s nDCG@10_s(q) over dev systems.
- **Topic concentration:** Jensen–Shannon divergence between X's and A's distributions over R-derived clusters. Also report the Gini coefficient of gold-passage frequency.
- **Language:** total-variation distance between language-class distributions (D3/D4).

### 13.5 Composite gap G (secondary; report the components too)

G(X) = mean of:

- (1 − τ_b)/2
- MAE_𝒮 / mean_s M_s(A)
- W1_len / sd_len(A)
- W1_leak / sd_leak(A)
- JS_topic

### 13.6 Uncertainty

- **Paired bootstrap** (B = 1,000): resample queries of A and of X independently with replacement, recompute all set-level metrics and agreements, and take 95% percentile CIs.
- **Comparing two methods:** use the **same** A-resamples for both and report the CI of Δτ_b and the one-sided bootstrap p-value.
- **Seeds:** report mean ± sd over the 5 R/A split seeds. B0 and B1 additionally average over 20 sampling seeds.
- **Multiple comparisons:** Holm correction across the 4 datasets for the primary endpoint.
- **Small-sample check:** for |A| < 200, also report τ_b CIs; call a difference "reliable" only if the CI excludes 0.

---

## 14. Experiments

Each experiment has a config in `configs/experiments/` and one command, `ha run E<k>`, which writes to the registry and regenerates its tables and figures.

| ID | Purpose (RQ) | Datasets | Output |
|---|---|---|---|
| E0 | Sanity: reproduce BEIR numbers; check qrels and ID mapping | D1, D2 | `tab_sanity` |
| E1 | Characterise the gap (RQ1): distributions and τ_b for every raw pool vs A | all | `fig_gap_distributions`, `tab_raw_gap` (per generator/style) |
| E2 | Main result (RQ2): B0–B7, M1, M3, O on held-out systems | D2, D3, D4 (+ D1 marked as dev) | `tab_main`, `fig_tau_ci` |
| E3 | Ablations (RQ3): no neighbours (isolated judge); no F gate; single dimension (H only, S only, P only); equal weights; no stratification; top-1 vs top-5 neighbours; no reranker for neighbours | D1 (dev) → confirm on D2, D4 | `tab_ablation` |
| E4 | Filter vs rewrite vs few-shot (RQ4): M1, M2, M3, B6; also measure intent drift (share of rewrites that fail the F gate; BGE-M3 cosine to the original) | all | `tab_rewrite`, qualitative examples |
| E5 | Language slices (RQ5): per-class τ_b and G for EN / ROM_HI / HINGLISH (D4) and DEV_HI (D3) | D3, D4 | `tab_language`, `fig_language_gap` |
| E6 | Distillation (RQ6): M1-d vs M1; cost per 1k queries | all | `tab_distill` |
| E7 | Budget curve: τ_b vs N for B0, B3, B5, M1 | D1, D4 | `fig_budget` |
| E8 | Human indistinguishability: blinded pairs (§15.2) | D1, D4 | `tab_turing` |
| E9 | Judge robustness: position-bias rate; judge scale (2B / 4B / 9B); cross-family (gemma3:4b); self-preference (judge = llama3.1:8b, the generator family) | D1, D4 | `tab_judge_robustness` |
| E10 | Generalisation: judge frozen on D1 + cal-dev, applied unchanged to D2–D4; dev vs held-out systems | all | rows of `tab_main` |
| E11 | (Optional) Sparse-qrel check: pool-label the top-10 of 4 systems for 150 M1 queries (LLM pre-labels + human check); recompute τ_b | D4 | `tab_qrels_check` |

**Reading guide for results.** If B1 (stratified-random) closes most of the gap, the contribution shifts towards "topic matching matters most". Report that honestly and position the judge's added value relative to B1.

---

## 15. Human studies

### 15.1 Calibration annotation

Covered in §9.4, with guidelines in Appendix C. Budget about 5–6 hours per annotator.

### 15.2 Blinded indistinguishability test (E8)

- **Items.** For D1 and D4, build 75 pairs each: (authentic from A, synthetic from B0) and (authentic from A, synthetic from M1), with 150 pairs in total, mixed and with the order randomised.
- **Raters.** 4–6 volunteer raters (not the two team members). For each pair: "Which one did a real person type?"
- **Metrics.**
  - Detection accuracy per condition (50% = indistinguishable). Two-proportion test between B0 and M1.
  - Fleiss' κ across raters.

### 15.3 Ethics

- A short consent paragraph opens the survey and the rating form.
- No identifiers are collected. The data is used for research and released under CC BY 4.0 after a PII check.
- Record faculty-advisor approval in `docs/decisions.md` before launch.

---

## 16. Distillation (`src/ha/distill/`)

- **Training data.** Judge scores s(q) and the gate decision for every query judged across D1 and the R-side pools (never A). Target: the continuous score, with a separate binary head for the gate.
- **Backbones.**
  - `google/muril-base-cased`, chosen because it was pretrained with transliterated data
  - `intfloat/multilingual-e5-small`, chosen for speed

  Input: `[query] [SEP] [top-3 neighbours]`; also report a query-only variant.
- **Training.** 3 epochs, lr 2e-5, batch 16, FP16, max_len 128, on the 3050. Model selection by 5-fold CV on the training data.
- **Evaluation.**
  - Spearman ρ with judge scores on held-out datasets
  - E6's τ_b when used for selection
  - throughput (queries/s) and cost per 1k queries versus the LLM judge

---

## 17. Dashboard (`app/dashboard.py`, Streamlit)

1. **Overview:** pool size, number kept and rejected, reasons, and the gap components before and after curation.
2. **Query browser:** each synthetic query with its neighbours, judge scores, decision, stratum and nearest authentic queries. Filter by reason, language and score.
3. **Systems view:** system rankings on A vs B0 vs M1 as a slope chart, with τ_b and CI.
4. **Language view:** per-class gaps (D3/D4).
5. **Try it:** paste a query and see the judge output live. Uses Ollama; disable the tab if Ollama is not running.

The dashboard reads only from `results/` and `data/processed/`. It never recomputes experiments.

---

## 18. Paper plan (`paper/`)

- **Format.** ACM `acmart` sigconf, which fits SIGIR / CIKM. Keep an 8-page body in mind with an optional appendix; keep an ACL-format port possible. The team will pick the venue; check deadlines and page limits on the day.
- **Build.** `ha paper build` renders tables from the registry through Jinja templates into `paper/tables/*.tex`, and figures into `paper/figures/*.pdf`. It then runs `latexmk` if available; otherwise the team compiles on Overleaf.

### 18.1 Outline

1. **Introduction.** The synthetic–authentic gap; why ranking agreement matters; the contributions C1–C4.
2. **Related work.** Synthetic test collections and RAG evaluation; query variants and user simulation; LLM-as-a-judge and behaviour grounding; topic-subset selection; code-mixed IR.
3. **Problem formulation.** Pools, R/A, systems, agreement measures, and the curation objective.
4. **HumanAlign.** Neighbour retrieval; judge dimensions; aggregation; stratified selection; rewrite variant; distillation.
5. **Data.** The four datasets; HiAQ-Services construction; language labelling; qrels.
6. **Experimental setup.** Generators, system pool, baselines, statistics, the sealed-data protocol and pre-registration.
7. **Results.** RQ1–RQ6 in order.
8. **Analysis.** Qualitative error cases; when curation fails; cost.
9. **Limitations and ethics.** MIRACL subsampling; elicited vs organic authenticity; sparse synthetic qrels; small |A|; judge biases; the survey population (mostly students).
10. **Conclusion.**

### 18.2 Required figures and tables

- **Figures:**
  - Fig. 1, pipeline (redraw the synopsis diagram in vector form)
  - Fig. 2, length / leakage / topic distributions (E1)
  - Fig. 3, τ_b with CIs per method and dataset (E2)
  - Fig. 4, slope chart of system rankings (A vs B0 vs M1) on D4
  - Fig. 5, budget curve (E7)
  - Fig. 6, per-language gap (E5)
- **Tables:**
  - Tab. 1, datasets
  - Tab. 2, main results
  - Tab. 3, ablations
  - Tab. 4, filter vs rewrite
  - Tab. 5, judge robustness
  - Tab. 6, human study
  - Tab. 7, distillation / cost

### 18.3 Artifact release

- **Repo:** the code, configs, frozen prompts, and the HiAQ-Services queries, qrels, corpus chunk IDs and URLs.
- **Corpus text:** only if licences permit; otherwise provide a re-scrape script.
- **Also include:** per-query run files, the judge outputs cache (as JSONL), and a `REPRODUCE.md`.

---

## 19. Phases, gates and timeline (about 14 weeks)

| Phase | Weeks | Work | Gate |
|---|---|---|---|
| P0 Setup | 1 | Repo, `.env`, venv, CUDA check, pull/download all models (§4.4), novelty re-check (§3.3), `preregistration.md` committed | **HUMAN GATE**: confirm paths, models, novelty report, pre-registration |
| P1 Data | 1–3 | Public loaders and corpus subsampling; HiAQ scraping and chunking; scenario authoring tools and leakage check; survey form generated from scenarios; language-ID module and tests; data cards | **HUMAN GATE**: team finalises scenarios and launches the survey (runs in parallel with P2–P4) |
| P2 Systems | 2–3 | All 20 systems; per-query run infrastructure; E0 sanity; latency; compute report | **AUTO GATE**: E0 within tolerance; unit tests pass |
| P3 Pools | 3–4 | Generate P-std, P-persona, P-fewshot, P-std-G2, P-docT5 for D1–D3 (D4 pools once the corpus is final; P-fewshot for D4 once R exists); dedupe; stats | **AUTO GATE**: pool sizes ≥90% of target; failure rate < 5% |
| P4 Judge | 4–6 | Neighbour index; judge v0; throughput benchmark; calibration export; after annotation, κ, prompt iterations v1–v4 on cal-dev; weights | **HUMAN GATE**: annotation done; review κ and ρ; approve the freeze candidate |
| P5 Dev and freeze | 6–8 | Selection and baselines; E1, E3, E7 on D1 with dev systems; decide the final configuration; write `FROZEN.json`; set `HA_UNSEAL=1` | **FREEZE GATE (HUMAN)**: both members sign off in `docs/decisions.md` |
| P6 Held-out evaluation | 8–9 | HiAQ qrels finalised; D4 pools and R/A; E2, E5 and E10 on all datasets and held-out systems; bootstrap | **AUTO GATE**: all runs in the registry; no sealed-data violations logged |
| P7 Extensions | 9–11 | E4 rewrite, E6 distillation, E8 human test, E9 robustness, E11 (optional) | **HUMAN GATE**: review results; decide the paper narrative |
| P8 Write-up | 11–14 | Dashboard; paper build; novelty re-check #2; limitations; artifact packaging; internal review (Claude Code writes a critical review as a sceptical reviewer in `reports/self_review.md`) | **HUMAN GATE**: submission decision |

If the survey collects fewer than 300 HiAQ queries by the end of week 6, fall back to D1–D3 as the main datasets and report HiAQ as a pilot with wider CIs. Log this in the decisions file.

---

## 20. Engineering standards

### 20.1 CLI

A single entry point, `ha`, with sub-commands:

- `ha data build <ds>`
- `ha gen <ds> <pool>`
- `ha neigh <ds>`
- `ha judge <ds> <pool>`
- `ha calib export|import|report`
- `ha select <ds> <method>`
- `ha systems run <ds> [--systems ...]`
- `ha eval <ds> <exp>`
- `ha run E<k>`
- `ha paper build`
- `ha dashboard`

Every command is idempotent and resumable (it skips work already in the cache).

### 20.2 Results registry

Every result row in `results/registry.jsonl` carries:

```json
{"run_id": "...", "exp": "E2", "dataset": "D4", "split_seed": 0, "method": "M1",
 "system_set": "heldout", "metric": "tau_b", "value": 0.0, "ci_low": 0.0, "ci_high": 0.0,
 "n_queries": 0, "git_sha": "...", "config_hash": "...", "frozen_hash": "...",
 "timestamp": "..."}
```

### 20.3 Sealing (`seal.py`)

- Loaders for A (held-out datasets), cal-test and held-out systems raise `SealedDataError` unless `HA_UNSEAL=1`.
- Every access after unsealing is appended to `results/unseal_log.jsonl`.
- Tests confirm the seal works.

### 20.4 Determinism

Global seed utility (Python, NumPy, torch); `torch.backends.cudnn.deterministic=True` for encoders. Record the Ollama model digest and version.

### 20.5 Tests (pytest)

Required tests cover:

- metric functions against `ranx` on toy data
- τ_AP on a worked example
- the bootstrap on a known case
- stratified-selection quotas
- the dedupe threshold
- language ID and CMI on 30 hand examples
- the seal
- JSON schema parsing of the judge output

`ruff` must be clean. CI is not required, but `pytest -q` must pass before every gate.

### 20.6 Logging

`loguru` to `logs/<date>.log`; progress bars with `rich`. Long jobs checkpoint every 200 items.

### 20.7 Windows notes

- Use `pathlib` everywhere.
- Avoid `multiprocessing` fork assumptions (use `spawn`-safe code with `if __name__ == "__main__":`).
- Set `num_workers=0` in DataLoaders if workers crash.
- UTF-8 everywhere (`PYTHONUTF8=1`).

### 20.8 Kaggle / Colab

Put notebooks in `notebooks/kaggle_*.ipynb`. Each one clones the repo, installs pinned requirements, runs exactly one CLI command and writes outputs to a dataset or zip that is then copied back to `results/` or `cache/`. It uses the same configs; no logic lives only in a notebook.

---

## 21. Risks and fallbacks

| Risk | Signal | Fallback |
|---|---|---|
| Too few survey responses | < 300 by week 6 | D1–D3 become the main datasets; D4 is reported as a pilot; extend collection for the camera-ready version |
| The judge's added value over B1/B3 is small | E3/E2 on dev | Reframe around what drives realism (topic matching vs style); keep M1 but make the analysis the contribution; strengthen the P dimension |
| Low κ on human calibration | κ < 0.5 after revision | Merge H and S into one "naturalness" rating; keep F binary |
| Judge JSON failures on qwen3.5 | > 5% failures | Switch to `qwen3:4b`; log it |
| VRAM out-of-memory | CUDA OOM | Smaller batch; FP16; move the model to Kaggle; never co-load with Ollama |
| Slow encoding of the MIRACL subsample | ETA > 3 h locally | Kaggle; reduce the random filler to 30k passages (log it) |
| τ_b noisy with ~20 systems | Wide CIs | Add systems (more e5 / arctic sizes, chunked variants); report τ_AP and winner agreement alongside |
| Rewrite drifts intent | High F-gate failure | Report it as a finding; restrict rewrites to style-only edits via the prompt |
| Scraping blocked | robots / ToS | Drop that source; add another public FAQ source; log it |

---

## Appendix A — `CLAUDE.md` (copy to the repo root)

```markdown
# CLAUDE.md — HumanAlign-RAG

Research codebase. Spec: docs/HUMANALIGN_RAG_BUILD_SPEC.md (read it before any change).

## Hard rules
- Never fabricate numbers. Every reported number comes from results/registry.jsonl with a run_id.
- Sealed data (A of held-out datasets, cal-test, held-out systems) stays sealed until FREEZE GATE (HA_UNSEAL=0).
- After freeze, do not change judge prompt/weights/thresholds; new versions get "-v2" and are reported.
- Report negative results. Log every decision in docs/decisions.md.
- No PII in data. Respect robots.txt and site terms.
- One model on the GPU at a time (RTX 3050, 6 GB). OLLAMA_KEEP_ALIVE=0 during torch jobs.
- No ONNX Runtime (DLL fails on this machine). Pure PyTorch.
- Windows: pathlib, UTF-8 (PYTHONUTF8=1), spawn-safe multiprocessing.

## Workflow
- Work phase by phase (spec §19). Stop at HUMAN GATEs and write reports/phase_N.md.
- Before each gate: pytest -q and ruff check must pass.
- Update docs/lab_notebook.md after each session.
- Prefer small, reviewed commits: "<phase>: <what>".

## Commands
ha data build <ds> | ha gen <ds> <pool> | ha neigh <ds> | ha judge <ds> <pool>
ha calib export|import|report | ha select <ds> <method> | ha systems run <ds>
ha run E<k> | ha paper build | ha dashboard
```

---

## Appendix B — Prompt templates (store as versioned `.j2` files)

### B.1 S-std (standard synthetic generation; deliberately mirrors common practice)

```
You are creating an evaluation set for a question-answering system.
Read the passage and write ONE clear, self-contained question that can be answered using only this passage.
Return only the question.

Passage:
"""{{ passage }}"""
```

### B.2 S-persona

Personas (sample one per passage):

1. a busy office worker typing quickly on a phone
2. a college student
3. an older person who is not very comfortable with apps
4. a first-time smartphone user in a small town
5. (D3/D4 only) someone who usually types in Hinglish (Hindi written in English letters, mixed with English words)

```
Imagine {{ persona_description }} who needs help with something covered in the passage below.
Write the exact text this person would type into the help-centre search box or a support chatbot.
Keep their natural length, spelling and style. Return only that text.

Passage:
"""{{ passage }}"""
```

### B.3 S-fewshot (exemplars from R only)

```
Here are examples of what real users of this service typed into its help search:
{% for e in exemplars %}- {{ e }}
{% endfor %}
Write ONE new query, in the same style as these real users, for a need that the passage below answers.
Do not copy any example. Return only the query.

Passage:
"""{{ passage }}"""
```

### B.4 Judge (v0; iterate only on cal-dev)

**System:**

```
You are an expert in how people search for help online, including how Indian users type in English, Romanised Hindi and Hinglish. You assess whether a machine-generated TEST QUERY is a realistic stand-in for what real users of this service type. You judge the query, not the passage.
```

**User:**

```
DOMAIN: {{ domain_description }}

SOURCE PASSAGE (the TEST QUERY was generated from it):
"""{{ passage }}"""

REAL USER QUERIES FOR SIMILAR NEEDS (typed by real users; order is random):
{% for n in neighbours %}{{ loop.index }}. {{ n }}
{% endfor %}
TEST QUERY: "{{ query }}"
DETECTED LANGUAGE: {{ lang_class }}

Rate the TEST QUERY from 1 to 5 on each criterion.

F  Fidelity: 5 = the passage fully answers it; 3 = partly; 1 = it does not.
H  Human-likeness: 5 = matches the real queries in length, wording, register, abbreviations, typos and level of detail; 3 = plausible but noticeably more formal or complete; 1 = reads like an exam question, documentation heading or chatbot prompt.
S  Natural specificity: 5 = uses only words and details a user would know BEFORE reading the passage; 3 = one passage-specific term or detail; 1 = relies on passage-only terminology, exact figures, conditions or steps.
P  Need plausibility: 5 = a common need among users of this service; 3 = occasional; 1 = an obscure detail only someone reading the document would think to ask.
C  Code-mix naturalness: only if the TEST QUERY or the real queries are not pure English, else null. 5 = mixing and spelling look like natural Indian typing; 3 = understandable but slightly stiff; 1 = mechanically swapped words, formal/Sanskritised transliteration, or unnatural grammar.

Do not reward copying a real query. Similar length and tone are good; identical wording is not.
Then give keep (true/false) and one reason from:
ok, not_answerable, too_document_like, over_specific, implausible_need, unnatural_code_mix, near_duplicate_of_real.

Return only JSON matching the schema.
```

**Anchor examples.** From v1 onwards, include 3 anchor examples drawn from cal-dev only, with the human ratings. Choose one high, one mid and one low example; for D4 include one Hinglish example.

### B.5 Rewrite (M2)

```
Real users of this service typed these queries for similar needs:
{% for n in neighbours %}- {{ n }}
{% endfor %}
Rewrite the TEST QUERY the way a real user of this service would type it, keeping exactly the same information need.
Match the real users' typical length, tone and level of detail. Use {{ target_language }} (chosen to match how these users write).
Do not copy any real query. Return only the rewritten query.

TEST QUERY: "{{ query }}"
```

`target_language` is sampled from R's language-class distribution: EN, ROM_HI or HINGLISH (D4); DEV_HI or EN (D3); EN (D1/D2).

---

## Appendix C — Annotation guidelines (summary; full version in `docs/annotation_guidelines.md`)

**Rating the query.**

- Rate each query on F, H, S, P, C and Overall realism (1–5), using the same anchors as B.4.
- You will not be told whether a query is real or generated. Rate what you see.
- Imagine the service's help-search box on a phone. "Overall realism" means: how likely is it that a real user typed exactly this?
- **Typos and abbreviations** are neither good nor bad by themselves; judge them against how real users type.
- **Hinglish:** judge spelling variation (*nahi / nahin / nai*) as natural. Penalise pure dictionary transliteration of formal Hindi that nobody types in chats.
- **Fidelity:** read the passage. A query is answerable if the passage contains what the person needs to act on.

**Process.**

- Do 20 practice items together, then rate the rest independently.
- Flag unclear items; do not discuss them until both have finished.

---

## Appendix D — Survey text (Google Form)

**Title:** How do you search for help? (2 minutes)

**Intro:**

> We are two RVCE students studying how people type questions into help pages and chatbots. You'll see 6 short situations. For each, type exactly what you would type into the app's help search or a support chatbot: short or long, any language or script you normally use, typos are fine. There are no right answers. We don't collect your name, email or phone number. Your answers will be used for research and may be published as an anonymous dataset. By continuing you agree to this.

**Per situation:**

> *"{{ scenario }}"*
> What would you type? (short text)

**Optional at the end:**

- Which languages do you usually type in on your phone? (English / Hindi in English letters / Hindi in Devanagari / Kannada / Other)
- Did you imagine using a phone or a computer?

---

## Appendix E — Key references

1. F. J. Kucia and B. M. Gawlik. Beyond Benchmark Scores: How Synthetic and Authentic Query Distributions Diverge in RAG Evaluation. CIKM '26. arXiv:2609.14579.
2. Z. Hussain and K. Nielbo. The Coverage Illusion: From Pre-retrieval Routing Failure to Post-retrieval Cascades in a Production RAG System. arXiv:2605.27220, 2026.
3. H. A. Rahmani, V. Ramineni, E. Yilmaz, N. Craswell, B. Mitra. Towards Understanding Bias in Synthetic Data for Evaluation. CIKM '25. arXiv:2506.10301.
4. J. van Elburg, P. van der Putten, M. Marx. Can we Evaluate RAGs with Synthetic Data? SynDAiTE workshop @ ECML-PKDD 2025. arXiv:2508.11758.
5. H. A. Rahmani, N. Craswell, E. Yilmaz, B. Mitra, D. Campos. Synthetic Test Collections for Retrieval Evaluation. SIGIR '24. arXiv:2405.07767.
6. He et al. Tip of the Tongue Query Elicitation for Simulated Evaluation. SIGIR '25. arXiv:2502.17776.
7. M. Alaofi, L. Gallagher, M. Sanderson, F. Scholer, P. Thomas. Can Generative LLMs Create Query Variants for Test Collections? SIGIR '23.
8. M. Alaofi, N. Ferro, P. Thomas, F. Scholer, M. Sanderson. Demographically-Inspired Query Variants Using an LLM. ICTIR '25.
9. Y. Wei et al. Can Retrievers Find the Same Paper from Different Aspects? (MAPLE). arXiv:2608.15624, 2026.
10. K. Hong, A. Troynikov, J. Huber. Generative Benchmarking. Chroma Technical Report, 2025.
11. A. Vardasbi, G. Penha, E. Palumbo, C. Hauff, H. Bouchard, M. Lalmas. As It Was: Aligning LLM Search Evaluation with Historical User Preferences. SIGIR '26. arXiv:2607.01040.
12. J. Guiver, S. Mizzaro, S. Robertson. A Few Good Topics: Experiments in Topic Set Reduction for Retrieval Evaluation. ACM TOIS 27(4), 2009.
13. L. Bonifacio et al. InPars: Data Augmentation for Information Retrieval using Large Language Models. SIGIR '22.
14. Z. Dai et al. Promptagator: Few-shot Dense Retrieval From 8 Examples. ICLR 2023.
15. S. Es et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL 2024 Demos.
16. Q. Zeng et al. Code-Switching Information Retrieval: Benchmarks, Analysis, and the Limits of Current Retrievers. Findings of ACL 2026. arXiv:2604.17632.
17. A. Acharya, R. Murthy, V. Kumar, J. Sen. Benchmarking and Building Zero-Shot Hindi Retrieval Model with Hindi-BEIR and NLLB-E5. NAACL 2025.
18. J. Kim et al. MiLQ: Benchmarking IR Models for Bilingual Web Search with Mixed Language Queries. arXiv:2505.16631, 2025.
19. P. Bailey, A. Moffat, F. Scholer, P. Thomas. UQV100: A Test Collection with Query Variability. SIGIR '16.
20. P. Wang et al. Large Language Models are not Fair Evaluators. ACL 2024.
21. E. Yilmaz, J. A. Aslam, S. Robertson. A New Rank Correlation Coefficient for Information Retrieval. SIGIR '08.
22. J. Chen et al. BGE M3-Embedding. Findings of ACL 2024.
23. N. Thakur et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS 2021 D&B.
24. X. Zhang et al. MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages. TACL 2023.
25. S. Khanuja et al. MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730, 2021.
26. G. V. Cormack, C. L. A. Clarke, S. Büttcher. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR '09.
27. L. Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023 D&B.
28. S. Shankar et al. Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences. UIST '24.
29. A. Das and B. Gambäck. Identifying Languages at the Word Level in Code-Mixed Indian Social Media Text. ICON 2014 (Code-Mixing Index).
30. T. Gebru et al. Datasheets for Datasets. Communications of the ACM 64(12), 2021.

The full 42-paper survey is in `HumanAlign-RAG_Literature_Survey.xlsx`; add the references above that are missing from it (4, 6, 11, 12, 21, 24, 25, 29, 30) to `refs.bib`.
