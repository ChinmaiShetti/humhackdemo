# HumanAlign-RAG: full project report (as of 2026-09-30)

**For:** the two-member team. **Scope:** everything done from project start (2026-09-27) to now.
**Sources:** git history, `docs/decisions.md` (D-001 … D-055), `docs/PAPER_LOG.md`, `docs/lab_notebook.md`, `reports/phase_{0..4}.md` and the files named inline. Numbers are copied from those records; nothing here is new analysis.

> **Note added 2026-10-10:** this is a dated snapshot, left as written. Two statements are superseded:
> - Its statement that option 3 (the collaborative sheet) is **not implemented** (§0, §9.1) is superseded by **D-056**: the template, separate ingest, instructions and tests were built on 2026-10-10.
> - Its use is now set by **D-057**: the sheet is exploratory only and not yet distributed; the Google Forms collect the initial authentic tranche.

---

## 0. Status at a glance

| Area | State |
|---|---|
| **P0 setup** | ✅ Done. Pre-registration frozen as `prereg-v1` (commit `b1c18e0`, 2026-09-27). |
| **P1 data (engineering)** | ✅ Done: D1/D2 data, splits, model-free LID, HiAQ corpus (1,572 chunks), survey tooling. |
| **P1 human gate** | ✅ Scenarios 150/150 approved; LID labels 100/100 final; survey variants built; 25 Google Forms created and verified (closed). Ethics metadata **waived** by the team lead (D-055). ⏳ Survey **not collected**. |
| **P2 retrieval** | ✅ Done. 7 configurations C1–C7; E0 sanity gate passed. |
| **Exploratory noise ceiling** | ✅ Done (labelled exploratory; does not feed H2). |
| **P3 synthetic pools** | ✅ Closed. Auto gate **partial fail** on pool size (6/16 pools ≥ 90%); pools kept as generated (D-048). |
| **P4 judge + calibration** | 🔶 Engineering done: neighbours D1/D2, judge engine and pilot, calibration design and export tooling, practice pack. **Blocked:** the 200-item calibration pack (needs the HiAQ inputs). **Human calibration not started.** |
| **P5–P8** | Not started. **No bulk judging has run.** |
| **HiAQ collection method** | **Changed on 2026-09-29:** the team chose a shared collaborative sheet ("option 3"). **Not implemented yet** (§9.1). |
| **Tests / lint** | 157 tests pass; `ruff check .` clean (last run 2026-09-29). |
| **Git** | Branch `backup`. HEAD `7368887`, one commit ahead of `origin/backup` (not pushed). **10 files uncommitted** (§10). |

---

## 1. The study in one page

- **Question:** can a neighbour-grounded LLM judge pick synthetic RAG test queries that rank retrieval systems the same way real user queries do?
- **Primary result (prereg H2):** system-ranking agreement, Kendall τ_b over **7 retrieval configurations**, with held-out authentic queries A. It is pooled over **D2 (CQADupStack-Android)** and **D4 (HiAQ-Services)**.
  - H2 holds if τ_b(M1, A) > τ_b(B0, A) **and** > τ_b(B1, A).
  - Test: one-sided paired bootstrap, B = 1,000, seed 20260927, intersection-union test at α = 0.05 (D-027).
- **Methods compared:**
  - B0 = random draw from the synthetic pool U;
  - B1 = stratified random;
  - M1-iso = judge without neighbours;
  - **M1 = neighbour-grounded judge** (the method under test).
- **Datasets:**
  - D1 FiQA: development only;
  - D2 CQADupStack-Android: held-out;
  - D4 HiAQ-Services: held-out, our own survey data on Indian government e-services (Income Tax e-filing and EPFO), with English, romanised Hindi and code-mixed queries.
- **Locked local model stack** (no new downloads without team approval, D-021):

| Role | Model | Identity |
|---|---|---|
| Generator + judge | Ollama `qwen3:4b-instruct-2507-q4_K_M` | digest `0edcdef34593eac1…` |
| Embeddings / neighbours | Ollama `bge-m3` (dense) | digest `7907646426070047…` |
| Reranker | `BAAI/bge-reranker-v2-m3` | revision `953dc6f6…`, local at `D:\ai-models\local\bge-reranker-v2-m3` |
| Lexical | BM25 (`bm25s`) | no model |

- **Integrity rules in force throughout:**
  - The frozen prereg is never edited above "## Amendments"; `tests/test_prereg_frozen.py` guards it.
  - **Sealed:** A of D2 and D4, and `cal-test`. `HA_UNSEAL` has never been set, and there is no `results/unseal_log.jsonl`.
  - Every reported number goes in `results/registry.jsonl` with a clean git SHA.
  - Exploratory analyses are labelled and never feed H2.
  - Everything lives on D: (D-008).

---

## 2. Timeline (git)

| Date | Commit | What |
|---|---|---|
| 09-27 | `56c3273` | P0: repo, core infrastructure, model verification, D1/D2 data |
| 09-27 | `3deb1ee` | P0: pin `qwen3:4b-instruct-2507-q4_K_M` (D-028) |
| 09-27 | `c8ef87a` | P0: line-ending-independent prompt hashes |
| 09-27 | `9102806` | P0: prereg draft r3, 10 review criteria locked (D-029) |
| 09-27 | **`b1c18e0`** | **P0: prereg frozen, tag `prereg-v1` (D-030)** |
| 09-27 | `f0504d2` | P0: `results/PREREG.json` (artefact hashes) + frozen-body guard test |
| 09-27 | `7de8b56` | P1: HiAQ source vetting, polite scraper, chunker, model-free LID, D1/D2 splits |
| 09-27 | `c442096` | P1: survey tooling (variants, forms, redirect, PII ingest), scenario sheet + leakage WARN |
| 09-27 | `7c81958` | P1: scenario sheet regenerated (empty) with 75/75 quota (D-038) |
| 09-27 | `53896ec` | P2: retrieval configs C1–C7, metrics, sealed query sets, E0 + latency (D-039) |
| 09-28 | `efb6c16` | P2: E0 passed; corpus encodings; latency; phase 2 report |
| 09-28 | `a21744b`, `0e7f9f2` | P2 (exploratory): τ_b + noise-ceiling analysis (D-040) |
| 09-28 | `178832b` | Docs: PAPER_LOG + AGENT_CONTEXT; session protocol |
| 09-28 | `d163086` | P1: human-side intake (scenario draft, LID audit, ethics reported) (D-041–D-043) |
| 09-28 | `1cda5f2`, `3e45616` | P1: review tools and review artefacts (D-044); nothing approved |
| 09-28 | **`809ceec`** | **P3: pools retained, partial auto-gate failure recorded (D-048)** |
| 09-28 | `4d5f2f0` | Phase 1: review artefacts finalised (D-045) |
| 09-28 | `4b8d224` | "phase 123, +4ing": the team's own commit of the P4 engineering work (D-049) |
| 09-29 | **`7368887`** | **Phase 1: review finalised and survey-variant gate (D-050–D-053)** |
| — | *(uncommitted)* | D-054 forms builder, D-055 ethics waiver, pilot ingest, docs (§10) |

---

## 3. Phase 0: setup and pre-registration

**Environment:** repo `D:\humanalign-rag`; Python 3.11.15; torch 2.11.0 + cu128 on an RTX 3050 (6 GB); temp files on D: (D-019); `ha doctor` all OK.

**Data (BEIR counts matched exactly):**
- **D1 FiQA:** 57,638 passages, 648 test queries, 1,706 qrels.
- **D2 CQADupStack-Android:** 22,998 passages, 699 queries, 1,696 qrels.

**Model decision (D-028):** the local `qwen3:4b` turned out to be **thinking-only**. The team approved downloading the instruct build `qwen3:4b-instruct-2507-q4_K_M` (2.33 GB, on D:). The tag without `-q4_K_M` does not exist.

**Verification (engineering, not results):**
- Generation: 0.54 s median per call.
- Judge: 1.60 s median per call; JSON valid 20/20; Fidelity separates matched (4.5) from mismatched (1.0) passages.
- Weaknesses in judge prompt v0, left for calibration to fix:
  - it is too generous to exam-style queries;
  - `reason` and `keep` disagree;
  - there is a halo effect on mismatched passages.

**Pre-registration (`docs/preregistration.md`):**
- Reviewed against the team's 10 locking criteria (D-029).
- Frozen 2026-09-27 as `prereg-v1` (D-030).
- `results/PREREG.json` holds the frozen-body hash and SHA-256 hashes of 12 artefacts.
- **Key fixed parts:**
  - H2 pooled test;
  - calibration procedure (§4: 200 items, 120 cal-dev / 80 cal-test, κ, prompt v0–v4 on cal-dev only, NNLS weights);
  - isolation table (§3.3);
  - model digests;
  - retrieval configurations;
  - generation settings (800 source passages per dataset, seed 7);
  - the fallback "fewer than 300 HiAQ queries → H2 on D2 alone" (§8).
- **Scope decisions:**
  - prototype-first, with CORE / OPTIONAL / FULL-SCALE tiers (D-020);
  - locked model stack (D-021);
  - no dev/held-out system split (D-022);
  - 7 configurations (D-023);
  - Ollama BGE-M3 dense only (D-024);
  - reranker repaired by hard links (D-025).

---

## 4. Phase 1: data

### 4.1 Engineering (done)
| Work package | Result |
|---|---|
| **Splits (P1.2)** | 5 split seeds, R 40% / A 60%, stratified by language × 10 topic clusters (D-034). FiQA R = 260 / A = 388, with 30 CAL-auth always in R (D-003). CQA R = 279 / A = 420. A of D2 is sealed. D4 is split after its data exist. |
| **LID (P1.3)** | Model-free rules plus a romanised-Hindi lexicon (D-032); classes EN / ROM_HI / HINGLISH / DEV_HI / OTHER; CMI. All 1,347 English D1/D2 queries classed EN. |
| **Source vetting (D-031)** | **Included:** Income Tax e-filing help, EPFO. **Excluded:** NPCI/BHIM (robots), DigiLocker (403), IRCTC (terms), ABDM (no reproduction), Parivahan (permission needed; future extension), india.gov.in (403). **Parked:** UIDAI (JavaScript-only; no headless browser approved). |
| **Crawl + chunks (D-033, D-037)** | 1 request every 2 s, robots-checked, every request logged; 301 + 6 pages. **1,572 chunks** (725 FAQ pairs, 847 sections; 1,418 Income Tax, 154 EPFO). |
| **Scenario sheet (D-035, D-038)** | 150 chunks from individual-user topics only; team quota 75 Income Tax / 75 EPFO; seed 17. |
| **Leakage checker (spec §6.3, D-036)** | REJECT on a shared rare document term or a BM25 give-away (> 2× margin); WARN at ≥ 60% word overlap. |
| **Survey tooling** | 25 × 6 variants, Apps Script form builder, random-redirect page, ingest with PII stripping. |

### 4.2 Human gate: scenarios (done)
1. **D-042:** a ChatGPT-assisted draft of 150 scenarios was received, pending team-lead approval.
2. **D-044, D-045:** review-safe tools were added (checker on a review copy, deterministic replacement-chunk draw with `--exclude`, LID adjudication table). The review artefacts were processed. **Nothing was approved by the agent.**
3. **D-051 (09-28):** the team approved 9 decisions, applied verbatim:
   - **replacement chunks and texts:** S017 → `incometax/1fa8efc9b1cd/002`, S025 → `incometax/6cebf3b156cc/003` (re-draw; the first pick `…/013` was professional-role), S136 → `incometax/306ec57bb9dd/004`, S147 → `incometax/d51b96a53e6a/010`;
   - **rewrites:** S005, S007, S067, S096, S137.
4. **D-052 (09-29):**
   - The remaining 137 were approved as drafted; S060 and S066 were kept as WARN.
   - S090 and S091 were rewritten by the team to keep their information needs ("someone **outside the family**"; "covered by **both** military pension and EPS").
   - **150/150 approved**, and copied to the official `data/hiaq/scenario_sheet.csv`.
     - That copy keeps the official conventions: `author = team`, `notes` empty; the ChatGPT provenance is recorded in the review artefact and in D-042/D-051/D-052.
     - Before writing, every row was verified against `chunks.parquet`.
     - **sha256 `4968e202f87efd5734d456c300603e742beca93859a528f602735fb947d35596`.**
5. **Checker result on the official sheet:** **146 PASS / 2 WARN (S060, S066) / 2 REJECT (S090 "unmarried", S091 "military")**. Both REJECTs are words the source chunk itself uses and the need requires.
6. **D-053, approved spec §6.3 exceptions:**
   - `configs/scenario_check_exceptions.yaml` is an explicit allowlist of **only S090 and S091**. Each entry is pinned to its decision (D-052), the exact checker reason, and the sha256 of the approved text.
   - `survey.gate_check` fails closed. An unlisted REJECT, EMPTY, an edited text, a changed reason, or a stale or mismatched check file all block.
   - The checker and the scenarios are unchanged. No prereg amendment (this is a build-spec rule).

### 4.3 Human gate: language ID (done)
- The audit grew from 30 to 100 examples (D-043). ChatGPT recommendations were kept separate and never used as labels.
- **D-052:** the team gave final labels for the 10 flagged rows (L14, L29, L53, L54, L55, L65, L88, L89, L99, L100). The other 90 take their provisional label.
- **100/100 final. Only L54 and L55 differ from provisional** (ROM_HI → HINGLISH).
- Final counts: DEV_HI 21, EN 20, HINGLISH 22, OTHER 19, ROM_HI 18.
- The 30 original test cases agree with the final labels, so `tests/test_langid.py` is unchanged.

### 4.4 Survey variants and Google Forms (done, closed)
- **Variants (D-053):** `ha data hiaq-variants` → `data/hiaq/survey_variants.json`, **sha256 `e125c5c2…`**.
  - 25 forms × 6 situations; each of the 150 exactly once; every form mixes both services; seed 17; deterministic on rebuild.
  - The payload records the two applied exceptions.
- **Forms builder fixed (D-054)** before use:
  - **Paste bug:** the documented paste into a quoted string would have broken on the file's 757 newlines and apostrophes. The fix pastes it as an object.
  - **Wrong or partial paste:** a pre-check (25 × 6, 150 IDs, fingerprint `0b3bb2e7…`) now stops the script before it creates anything.
  - **Language cue:** the optional questions moved to page 2, so their language list can't cue answers.
  - **Personal accounts:** the sign-in setting is tolerated there.
  - **Created closed:** forms are made not accepting responses.
  - **Record:** each form writes one manifest line to the log.
  - **Dry run:** Node with a mocked Forms API, the real JSON, no Google calls.
- **Created by the team on 2026-09-29, 18:51–18:54** (personal Google account). The log was saved as `data/hiaq/forms_creation_log.txt.txt` (Windows double extension).
- **Verified:** `ha data hiaq-forms-manifest --log …` → `data/hiaq/forms_manifest.json`, **sha256 `b5775eeb…`**.
  - Forms 1–25 each once; unique IDs and URLs.
  - 6 items per form; all 150 IDs once, in variant order; every item text matches the official sheet word for word (**S090 in form 05, S091 in form 11**).
  - Every form **not accepting responses**.
- **Participant-facing content:** the Appendix D text only. No email, sign-in, name or demographics; 2 optional questions (languages usually typed; phone or computer). No mention of the study's hypothesis, LLMs or Hinglish.
- Also written: `data/hiaq/survey_questionnaire_all_forms.md` (all 25 forms, readable).

### 4.5 Ethics
- **D-041:** faculty-advisor approval was reported as obtained ("DONE", commit `d163086`).
- **D-055 (09-29):** the team lead decided the approval was informal and **not to record** the approver, role, date, scope or conditions.
  - The five placeholders are marked "not recorded: waived (D-055)". **No values were invented.**
  - The protocol gate is updated accordingly.
  - **Paper consequence:** the ethics statement may say only "informal approval by the faculty advisor". Rating-study (E8) coverage is not documented.

---

## 5. Phase 2: retrieval configurations

**Configurations (frozen):**
- C1 BM25 (0.9/0.4);
- C2 BM25 (1.2/0.75);
- C3 BGE-M3 dense;
- C4 RRF(C1, C3);
- C5 C3 → rerank;
- C6 C1 → rerank;
- C7 C4 → rerank.

**E0 sanity gate, D1 FiQA, all 648 queries** (run `E0-20260928T003743-14d0b0`, git `53896ec`):

| Config | nDCG@10 | Reference |
|---|---|---|
| C1 | 0.2380 | 0.236 ± 0.02 (BEIR) ✅ |
| C2 | 0.2384 | — |
| C3 | 0.4128 | 0.413 ± 0.03 (DRAMA) ✅ |
| C4 | 0.3578 | — |
| C5 | 0.4354 | — |
| C6 | 0.3582 | — |
| C7 | 0.4328 | — |

- **Near-ties on authentic FiQA:** C1–C2 Δ 0.0004, C4–C6 Δ 0.0004, C5–C7 Δ 0.0027. These limit how high τ_b can go for any query set.
- **Runtime:**
  - FiQA corpus encoding took 28.6 min; CQA took 10.1 min.
  - Latency per query: 1 ms (BM25), 16 ms (dense), 0.65–0.88 s (reranked).
- **D2 before the freeze:** only corpus-side work and a 10-made-up-query smoke run. Its real queries are sealed (D-039).

### Exploratory authentic noise ceiling (D-040; context only)
Run `EXP-noise-ceiling-20260928T011917-5bc429`, D1 only:
- **Complementary halves:** mean τ_b 0.779.
- **Independent resamples:** mean 0.857, median 0.905.
- **Frozen R vs A per seed:** 0.905 / 0.810 / 0.714 / 0.810 / 0.524.
- **Finding:** complementary halves understate agreement, because their means are anti-correlated. Even two independent real-query sets reach a median τ_b of about 0.9, which gives a scale for the synthetic-vs-authentic numbers. This does not affect H2.

---

## 6. Phase 3: synthetic pools

**Settings (frozen):**
- qwen3:4b-instruct, T 0.7, top_p 0.9, max 64 tokens, seed 1000 + i;
- 800 source passages per dataset (seed 7);
- templates `gen_std_v1`, `gen_persona_v1`, `gen_fewshot_v1`;
- drop < 2 tokens and > 60% first-sentence overlap;
- dedupe at cosine 0.95.

**Reproducibility fix (D-046):** Ollama's KV-prefix reuse made repeated prompts sample differently, so every run now starts from a fresh model state, in fixed order.

**Final unique queries per pool (target 800):**

| Dataset | P-std | P-persona | P-fewshot s0–s4 |
|---|---|---|---|
| D1 FiQA | 727 | 716 | 780 / 772 / 779 / 773 / 783 |
| D2 CQA | 665 | 630 | 688 / 668 / 658 / 666 / 676 |
| D4 HiAQ | 586 | 645 | deferred (needs D4 R) |

- **Auto gate: partial FAIL.**
  - Pool size ≥ 90%: **6/16** pass. The first count said 7/16; it was corrected in D-047.
  - Failure rate < 5%: **16/16** pass (maximum 0.13%).
- **Cause:** the frozen first-sentence copy filter on question-shaped corpora. CQADupStack passages are user questions, and HiAQ chunks open with their own FAQ question. The one failure per D2 pool is a single 7,885-token passage that exceeds Ollama's 4,096 context.
- **D-048 (team):** pools retained exactly as generated. No filter, top-up or context change; no prereg amendment.
- **U = P-std ∪ P-persona:** D1 1,443, **D2 1,295**, D4 1,231.
- Not generated: D4 P-fewshot (needs R), P-std-G2 (no gemma3:4b), P-docT5 (download not approved).

---

## 7. Phase 4: neighbours, judge, calibration

### 7.1 Neighbours (P4.1, D-049)
- **Method:** bge-m3 dense top-30 against **R_s only** → reranker (max_length 128) → top-5.
- **Features:** `nn_max_cos`, `nn_mean5_cos`, `nn_lang_mix`; near-duplicate if > 0.97.
- **Done for D1 and D2, seeds 0–4:** 0 near-duplicates. Median `nn_max_cos` is 0.566–0.570 (D1) and 0.638–0.645 (D2). About 100 s of reranking per seed.
- **D4:** waits for its R.

### 7.2 Judge engine and pilot (P4.2)
- **Prompt v0:** spec B.4 verbatim.
- **Decoding:** temperature 0, seed 0, `num_ctx` 4096, JSON-schema output.
- **Robustness:** retry once with the validation error; 2 permutations plus a third on Δ ≥ 2, median per dimension.
- **M1-iso:** no neighbour block.
- **Aggregation:** gate F ≥ 4; NNLS weights with 5-fold CV.
- **Weights are `null`** until calibration.
- **Pilot** `P4-judge-pilot-20260928T230859-1f27d5` (engineering only): 12 D1 queries, 38 calls; **38/38 valid JSON**, 1.50 s per call, deterministic on 4/4 uncached repeats.
- **Bulk projection:** about 14.5 h locally for D1 + D2. This is Phase 5 and **has not been run**.

### 7.3 Calibration design (D-050; team decisions)
1. **Authentic items get a real qrels passage.**
   - Rule: highest grade, pids sorted, index = sha256("17|dataset|qid") mod n.
   - FiQA: 30/30 resolved (19 have several candidates).
   - HiAQ: pending its qrels. Nothing is substituted.
2. **Annotators see 5 neighbours taken from R_s0 minus all 60 CAL-auth items**, so no authentic item is revealed as "real". This is a display rule only; the judge side is unchanged.
   - Dry run: 0 leaks (the judge side would have shown them in 42/70 blocks).
3. **HiAQ domain text** is in `configs/domains.yaml`; the dataset configs are hash-locked.
4. **CSV spreadsheet workflow**, not Label Studio.
   - Blind raw sheets per annotator; a separate key; adjudication (flag at |Δ| ≥ 2) and final labels as separate files.
   - Raw sheets are never overwritten once anything has been entered.

**Tools and files:**
- Tooling: `src/ha/human/calibration.py`; `ha calib status | export [--practice] | import | finalize`.
- **Practice pack written:** `data/calibration/sheets/practice.csv` (20 FiQA synthetic items outside the 200).
- **Guidelines v1:** `docs/annotation_guidelines.md`. The team confirmation date is blank.
- **The 200-item pack is blocked:** it needs the 30 HiAQ CAL-auth items, their qrels, and HiAQ R.

---

## 8. Decisions register (D-001 … D-055)

| ID | Summary | By |
|---|---|---|
| D-001 | H2 wording, ≥ 2 of 3 held-out (superseded by D-027) | team |
| D-002 | Seed tiers: 5 / 3 / 1 seeds by analysis tier | team |
| D-003 | CAL-auth forced into R; no self-neighbours | team |
| D-004 | Ollama judge backend everywhere | team |
| D-005 | Full HiAQ relevance labelling (30% fallback only if infeasible) | team |
| D-006 | Code to Kaggle as a zip; no GitHub token | team |
| D-007 | Phase-based timeline; no assumed exam dates | team |
| D-008 | Everything on D: | team |
| D-009 | Scoped seals (cal_test, heldout_A, heldout_systems) | agent |
| D-010 | E0 sanity checks for held-out systems | agent |
| D-011 | Reuse existing model caches | agent |
| D-012 | Python 3.11 and awkward packages | agent |
| D-013 | E10 disclosure for D4 | agent |
| D-014 | Survey: 25 variants + random redirect | agent |
| D-015 | Registry provenance | agent |
| D-016 | How the prereg is frozen | agent |
| D-017 | Operationalising H2 clauses (partly superseded by D-027) | agent |
| D-018 | PYTHONUTF8 user environment variable | agent |
| D-019 | Temp files on D: | team |
| D-020 | Prototype-first scope (CORE / OPTIONAL / FULL-SCALE) | team |
| D-021 | Locked local model stack; no downloads without asking | team |
| D-022 | No dev/held-out system split | agent |
| D-023 | Seven retrieval configurations | agent |
| D-024 | Ollama BGE-M3, dense only | agent |
| D-025 | Reranker repair by hard links | agent |
| D-026 | Core datasets | agent |
| D-027 | **H2 = pooled agreement test over D2 + D4; HiAQ core** | team |
| D-028 | Local qwen3:4b is thinking-only → instruct-2507-q4_K_M | team |
| D-029 | Prereg reviewed against 10 locking criteria | agent |
| D-030 | **Prereg frozen as `prereg-v1`** | team |
| D-031 | HiAQ source vetting | agent |
| D-032 | Language ID without a model | agent |
| D-033 | HiAQ chunking | agent |
| D-034 | Split stratification clusters | agent |
| D-035 | Scenario chunks from individual-user topics only | agent |
| D-036 | Scenario word-overlap WARN | agent |
| D-037 | Scraper safety stop | agent |
| D-038 | Scenario sheet 75/75; hosting on GitHub Pages; optional sources | team |
| D-039 | Phase 2 scope under the seal | agent |
| D-040 | Authentic noise ceiling: exploratory only | team |
| D-041 | Ethics approval reported | team |
| D-042 | ChatGPT-assisted scenario draft received | team |
| D-043 | LID audit expanded to 100 | team |
| D-044 | Review artefacts processed; nothing approved | agent |
| D-045 | ChatGPT recommendations applied to the review artefact only | agent |
| D-046 | P3 engine details; fresh-state runs | agent |
| D-047 | P3 auto gate measured: partial FAIL (count corrected 6/16) | agent |
| D-048 | **P3 pools retained as generated** | team |
| D-049 | P4 started: neighbours, judge engine, pilot | agent |
| D-050 | Calibration display decisions (qrels passage, filtered neighbours, domain text, CSV) | team |
| D-051 | 9 scenario decisions approved | team |
| D-052 | **LID final 100/100; scenarios final 150/150; official sheet written** | team |
| D-053 | §6.3 exception allowlist (S090, S091); variants built | team |
| D-054 | Forms builder fixed; forms created closed | agent |
| D-055 | Ethics metadata not recorded; pilot data kept separate | team lead |

---

## 9. Open items and decisions needed

### 9.1 HiAQ collection by shared sheet ("option 3", chosen 2026-09-29): NOT implemented
> *Superseded 2026-10-10 by D-056 (implemented) and D-057 (exploratory only; not distributed). See the note at the top.*
- **Team choice:** real people type answers into one shared collaborative sheet, instead of 100 people filling Google Forms. This is kept as a **separate, clearly labelled set**: outside HiAQ's authentic data, the splits and the main test, and disclosed.
- **Designed but not written:** a tool-safety service outage blocked every write. Nothing exists yet; `data/hiaq/collab/` does not exist.
  - **Template:** 100 blocks × 6 = 600 rows; each situation 4 times.
  - **Columns:** `block, round, form, item, scenario_id, situation, answer, contributor_code, languages, device`.
  - **Separate ingest:** refuses edited situations, answers without a contributor code, two contributors in a block, or one person answering a situation twice. It writes only `collab/collab_queries.parquet` (`source = collab`).
  - **Contributor rules and message:** see the chat of 2026-09-29.
- **To do:** save the code, template, tests, D-056 and the protocol section.
- **Warning:** in a shared sheet people can see each other's answers, which weakens independence. Use per-person copies or filters.

### 9.2 Consequences of option 3 (team decisions needed)
1. **D4 authentic set would be empty.** Prereg §8: with fewer than 300 HiAQ queries, H2 and H3 are evaluated on **D2 alone** and D4 is reported as a pilot. This must be recorded as a dated **amendment before `heldout_A` is unsealed**.
2. **Calibration HiAQ half is blocked.** Prereg §4 needs 30 authentic HiAQ items from the first survey tranche. Options:
   - still collect some real Forms responses (even about 30 people);
   - formally decide to use collaborative-sheet answers for these items, which reverses option 3 for them and must be disclosed;
   - amend the calibration design.
3. **The 25 Google Forms** stay closed and unused unless the team decides otherwise. The 5-person pilot (protocol §3.4) matters only if Forms are used.

### 9.3 Other open items
- **Guidelines v1:** confirmation date blank (team).
- **Engineering not built** (needed once any HiAQ data exists):
  - a builder turning collected queries into `data/processed/hiaq/{queries_auth, qrels_auth, splits}` (off-topic drops, source-chunk grade-2 qrels, CAL-auth draw);
  - `configs/datasets/hiaq.yaml`.
- **Hosting:** the redirect page was planned for GitHub Pages, which requires a push. So far the instruction has been "no push".
- **Stale lines in `docs/AGENT_CONTEXT.md`:** §2 still says "ethics metadata is missing" and "Not yet approved by the team". Both are superseded by D-055 and the forms step. They should be refreshed with the next update.

---

## 10. Uncommitted work (since `7368887`)

| File | Change |
|---|---|
| `tools/make_forms.gs` | D-054 fixes (object paste, pre-check, page break, closed forms, manifest log) |
| `src/ha/data/survey.py` | forms-manifest verifier (`record_forms_manifest` etc.); `ingest(pilot=True)` → `pilot_queries.parquet` |
| `src/ha/cli.py` | `ha data hiaq-forms-manifest`; `ha data hiaq-ingest --pilot` |
| `tests/test_survey.py` | +9 tests (manifest, script checks, pilot ingest) |
| `docs/decisions.md` | D-054, D-055 |
| `docs/survey_protocol.md` | forms steps (D-054); ethics §0 and pilot steps (D-055) |
| `docs/AGENT_CONTEXT.md`, `docs/PAPER_LOG.md`, `docs/lab_notebook.md` | status, ethics line, notebook entries |
| `data/humanside/humanalign_phase1_review.md` | ethics placeholders marked waived |

Git-ignored files (not in git; tracked by the hashes above):
- `data/hiaq/scenario_sheet.csv`, `scenario_check.csv`, `survey_variants.json`;
- `forms_creation_log.txt.txt`, `forms_manifest.json`, `survey_questionnaire_all_forms.md`;
- `data/calibration/`.

---

## 11. Problems found and fixed along the way (for the team's awareness)

| Problem | Fix |
|---|---|
| Local `qwen3:4b` was thinking-only | instruct build downloaded after team approval (D-028) |
| Ollama nondeterminism from KV-prefix reuse | fresh model state per run, fixed order (D-046) |
| P3 gate totals mis-added (7/16) | corrected to 6/16 (D-047); per-pool rows were right |
| Regex bugs in the checker ("ere" matched "where"; firm/trust boundaries); nav links parsed as FAQ questions | fixed with tests |
| Scraper safety stop skipped logging its last request | fixed; every request now logged |
| Pilot answers would have been written into the real dataset file | `--pilot` writes a separate file (D-055) |
| Forms builder paste bug; language cue on the answer page; forms opening immediately | fixed and dry-run verified (D-054) |
| Official-sheet copy stopped on empty `heading_path` | 66 rows legitimately have no heading; rule changed to "equals the chunk store" (D-052) |
| S090's approved text also REJECTs ("unmarried"), not only S091 | found by the checker; exception approved by the team (D-053) |
| **Agent errors caught:** a fabricated qwen3:8b digest tail (then verified: it matched); review files wrongly reported missing (team corrected); the collaborative-sheet file described as existing before it was created | each corrected in the conversation; nothing downstream used the wrong values |

---

## 12. Where things are

| What | Where |
|---|---|
| Live status for agents | `docs/AGENT_CONTEXT.md` |
| Decisions | `docs/decisions.md` |
| Paper material | `docs/PAPER_LOG.md` |
| Session history | `docs/lab_notebook.md` |
| Plan and tracker | `docs/PLAN.md` |
| Pre-registration + hashes | `docs/preregistration.md`, `results/PREREG.json` |
| Phase reports | `reports/phase_{0,1,2,3,4}.md` (+ checklists, stats JSON) |
| Survey protocol | `docs/survey_protocol.md` |
| Annotation guidelines | `docs/annotation_guidelines.md` |
| Human review artefacts | `data/humanside/` |
| HiAQ corpus, sheet, variants, forms | `data/hiaq/` |
| Synthetic pools | `data/synthetic/<ds>/<pool>/` (hashes in `reports/phase_3_pool_files.sha256`) |
| Neighbours | `data/neighbours/<ds>/s<seed>/` |
| Calibration packs | `data/calibration/` |
| Registry | `results/registry.jsonl` |

## 13. Suggested next steps (in order)
1. **Decide 9.2 (1)–(2):** accept the D2-only fallback for H2, and choose how to fill the 30 HiAQ calibration items.
2. **Commit** the current uncommitted work (§10), after tests and lint.
3. **Implement option 3** (§9.1): template, separate ingest, D-056, protocol section. Then pilot the sheet with 2–3 people.
4. Build the HiAQ processing step and config once any HiAQ data exist.
5. Confirm guidelines v1 → calibration annotation (H-08) → P4.4/P4.5.
6. Phase 5 (bulk judging, about 14.5 h) only after the calibration freeze and an explicit go-ahead.
