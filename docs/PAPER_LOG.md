# Paper log: HumanAlign-RAG

A cumulative, paper-oriented record of **everything we do**, organised by paper section. It is the source for writing the paper; the code, registry and decision log are the evidence.

**Rules (agents and humans)**
1. **Update this file after every piece of work**: new data, runs, findings, deviations, limitations.
2. Every **result** number cites its `run_id` from `results/registry.jsonl`, or its artefact file. Descriptive data facts cite the file they come from. **Never type an unsourced number.**
3. Mark every exploratory item **[EXPLORATORY]** and every engineering check **[ENGINEERING]** (never a paper result).
4. Record negative and null findings and deviations. Never delete an entry; add a correction below it.

Last updated: 2026-09-28 (D-045: review recommendations applied to the artifact; nothing approved).

---

## 1. Framing and contributions (for the Introduction)
- **Research question:** can neighbour-grounded LLM judging and curation make synthetic RAG evaluation queries better proxies for authentic human information needs?
- **Primary outcome:** system-ranking agreement (Kendall τ_b over 7 retrieval configurations) between a query set and held-out authentic queries. The judge's own scores are secondary.
- **Scope:** a reproducible research prototype on a single laptop GPU (RTX 3050 6 GB), prototype-first (D-020). Full-scale variants (a 20-system pool, other generator and judge families) are future work.
- **Claimed contributions** (spec §3.2, scoped by D-020):
  - C1: a neighbour-grounded realism judge for curating synthetic queries, validated by ranking agreement.
  - C2: the filter vs no-neighbour control (M1-iso) as the core ablation; the filter vs rewrite comparison is optional.
  - C3: HiAQ-Services, scenario-elicited Indian EN / Romanised-Hindi / Hinglish queries.
  - C4: distillation, optional.
- **Novelty check #1** (2026-09-27, `docs/novelty_check.md`): no overlap with C1–C4. The closest works are Kucia & Gawlik (CIKM '26; diagnosis only), Vardasbi et al. (SIGIR '26; behaviour-grounded **relevance** judge) and MAPLE (grounding at generation time). Rahmani et al. (SIGIR '24) report τ = 0.157 with sparse source-only qrels, which is a design risk for us (see §9).

## 2. Pre-registration (for Methods and Appendix)
- Frozen **2026-09-27** as git tag **`prereg-v1`** (commit `b1c18e0540f9f5943e1019c6a5e94de03b15aeff`). The hash of the frozen text and SHA-256 hashes of 12 artefacts are in `results/PREREG.json`, and a test guards the frozen body.
- **Single primary test (H2):** pooled over D2 and D4, τ_b(M1, A) > τ_b(B0, A) **and** > τ_b(B1, A). One-sided paired bootstrap, B = 1,000, seed 20260927, intersection-union test at α = 0.05. Per-dataset results are secondary breakdowns.
- **Secondary:** H3, M1 > M1-iso (neighbour grounding); H1, raw synthetic differs from authentic (descriptive); H4, the per-language gap (D4; classes with at least 50 held-out queries).
- **Locked before any result:** the calibration procedure (§4), the isolation table (§3.3), full model digests, retrieval configurations (§5.3), the D4 pooling configurations (C1, C3, C4, C5), N = min(|A|, number passing each judge's gate), linear-gain nDCG (ranx), and seeds (calibration 17; bootstrap 20260927).
- It was reviewed against 10 locking criteria before freezing (D-029). Two further gaps were found and fixed: size matching under gate shortfalls, and D4 qrels pooling.

## 3. Models and compute (Methods, Reproducibility)
| Role | Model | Identity |
|---|---|---|
| Generator and judge | Ollama `qwen3:4b-instruct-2507-q4_K_M` | digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` |
| Dense embeddings | Ollama `bge-m3` (F16) | digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` |
| Reranker | `BAAI/bge-reranker-v2-m3` (fp16) | HF revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

- **Environment:** Windows 11; RTX 3050 Laptop 6 GB; 15.2 GB RAM; Python 3.11.15; torch 2.11.0+cu128; Ollama 0.34.4. Packages are pinned in `env/freeze_2026-09-27.txt`. The code refuses to run on any digest mismatch.
- **Methods note: the generator/judge choice (D-028).** The default Ollama tag `qwen3:4b` is the **thinking-only** Qwen3-4B-2507 build (it shares ID `359d7dd4bcda` with `qwen3:4b-thinking-2507-q4_K_M`).
  - Four non-thinking strategies all failed to stop reasoning text from appearing in generated queries: `think=False`, `/no_think`, an empty-think prefill, and JSON-constrained output.
  - With `think=True`, one call spent 2,048 tokens (49.4 s) without producing an answer.
  - Source: `reports/phase_0_benchmark_qwen3-4b-thinking.json`, `tools/diag_gen_embed.py`.
  - We therefore use the non-thinking **instruct** sibling at the same quantisation (Q4_K_M). Worth a sentence in the paper, as a reproducibility hazard for others.
- **[ENGINEERING] instruct-model verification** (`reports/phase_0_llm_verification.json`):
  - generation: 20/20 genuine queries, median 0.54 s per call;
  - judge: 20/20 valid JSON, median 1.60 s per call; mean F 4.5 with matched passages vs 1.0 with mismatched controls; 0/8 permutation pairs differed by ≥ 2;
  - VRAM 2.96 GiB.
  - Weaknesses of the v0 prompt, noted for calibration and **not** tuned: H is lenient toward exam-style queries; `reason`/`keep` are inconsistent; halo effect under mismatch.
- **Reranker weights** were reused from an existing local cache through NTFS hard links (D-025). No model was downloaded apart from the team-approved `qwen3:4b-instruct-2507-q4_K_M`.

## 4. Data (Data section; data cards)
### 4.1 Public datasets (source: `data/processed/<ds>/meta.json`)
| ID | Dataset | Role | Passages | Eval queries | Qrels | HF revision |
|---|---|---|---|---|---|---|
| D1 | FiQA-2018 (BEIR test) | development | 57,638 | 648 | 1,706 (2.63 per query) | `BeIR/fiqa` 979c07a7…, qrels 252958f2… |
| D2 | CQADupStack-Android | held-out | 22,998 | 699 | 1,696 (2.43 per query) | `mteb/cqadupstack-android` e03f271e… |

Counts match the BEIR paper exactly, and the build checks them automatically.

### 4.2 Splits (`data/processed/<ds>/splits.parquet`; D-034)
- R 40% / A 60% for split seeds 0–4, stratified by language class × topic cluster. Clusters: k-means on bge-m3 query embeddings, k = min(10, ⌊n/30⌋), used for splitting only.
- FiQA: R = 260 / A = 388 for every seed, 10 clusters, 30 CAL-auth queries (seed 17) forced into R for all seeds.
- CQA-Android: R = 279 / A = 420, 10 clusters. **A is sealed** until the freeze.

### 4.3 Language identification (D-032)
- Model-free token LID using:
  - Devanagari script;
  - a hand-curated romanised-Hindi lexicon (chat spellings included);
  - an English vocabulary with counts from the FiQA corpus (D1 only; count ≥ 5);
  - a named-entity / service list (tagged `ne`, excluded from CMI).
- Hindi words that also occur at least 50 times in FiQA (to, main, me, is, …) are ambiguous and resolved by the majority language of their text.
- CMI follows Das & Gambäck (2014). Classes: EN (< 0.10 Hindi share), ROM_HI (> 0.80), HINGLISH (in between), DEV_HI and OTHER by script.
- **Validation:** 30/30 hand-labelled examples pass (**labels provisional until the team reviews them, H-09**). All 1,347 English queries of D1 and D2 are classed EN, so there are no false code-mix positives. D4 gets a 20% human check.

### 4.4 HiAQ-Services corpus (D-031, D-033, D-035, D-037)
**Source vetting.** Every source was checked against robots.txt **and its live website policy**. Search-engine summaries of policies were stale in two cases (Parivahan, EPFO); the live page decided.

| Source | Decision | Reason |
|---|---|---|
| Income Tax e-filing help | included | no robots.txt, no automated-access clause; reproduction free with acknowledgement; linking needs permission (the release will cite, not link) |
| EPFO | included | robots allows; the live policy permits reproduction "without requiring specific permission" |
| NPCI / BHIM UPI | excluded | robots `Disallow: /` for all agents (ClaudeBot named) |
| DigiLocker | excluded | robots.txt returns 403 |
| IRCTC | excluded | terms forbid copying or reproduction |
| ABDM | excluded | reproduction requires permission |
| Parivahan | excluded | live policy: reproduction only after emailed permission (possible future extension) |
| india.gov.in | excluded | CDN blocks crawlers (403) |
| UIDAI | not used | FAQ content is JavaScript-rendered; the team declined a headless browser |

**Crawl.**
- At most 1 request every 2 s, identifying User-Agent, robots checked per host, every request logged with URL and time (`data/hiaq/raw/<source>/pages.jsonl`).
- Income Tax was seeded from its sitemap: 382 URLs, of which 288 are English help pages not reachable through static links.
- 399 requests: 301 pages returned 200, 93 returned 404, and 5 returned page-specific 403s. The safety stop (3 consecutive 403/429) fired and the crawl was not resumed. EPFO: 6 pages.

**Chunking.**
- Heading-aware and FAQ-aware: `<details>` accordions or questions of 3–40 words ending in "?" become question–answer chunks; other text becomes sections of at most 350 words with 50-word overlap.
- Navigation and menu link-only blocks are dropped. Boilerplate (appearing on at least 30% of a source's pages) is removed.
- Exact dedupe, then MinHash (Jaccard ≥ 0.9).

**Result (`data/hiaq/chunk_stats.json`):**
- 1,778 raw → 1,587 after exact dedupe → **1,572 chunks**;
- 725 question–answer and 847 section chunks;
- 1,418 Income Tax and 154 EPFO;
- median 89 words (90th percentile 350).

This is within the spec's 1,500–4,000 target and above the 800 source passages the pre-registration needs. Professional topics (statutory forms, companies, firms, CAs, deductors) stay in the corpus as distractors but are excluded from scenario sampling.

### 4.5 HiAQ survey design (spec §6.3; `docs/survey_protocol.md`)
- **Scenario sheet:** 150 chunks, **75 Income Tax + 75 EPFO** (team quota, D-038; 81 topics; seed 17), from individual-user topics only (925 Income Tax and 118 EPFO chunks eligible). **All scenarios are written by the team lead.** The agent never writes or substitutes them.
- **Leakage check** (spec §6.3: shared rare document terms, or a BM25 self-retrieval margin above 2×), plus a **WARN signal we added (D-036)**, prompted by a negative finding about the spec's check:
  - Using each chunk's own FAQ question as a pseudo-scenario, the spec's rules rejected only **6/20**.
  - Word overlap with the chunk of at least 60% flags **58/58** such pseudo-scenarios (median overlap 1.0).
  - Three hand-written natural scenarios scored 0.00–0.50.
  - WARN means human review; it is not an automatic reject.
- **Forms:** 25 Google Forms × 6 scenarios (each scenario used once, both services in every form), with the Appendix D text verbatim. No Hinglish prompting; no identifiers; one GitHub Pages link redirects at random.
- **Ingest:** PII stripped (phone, email, 12-digit Aadhaar-like numbers and PAN become placeholders); whitespace normalised; typos kept; language class and CMI attached. Exclusions (empty, not a query, off-topic) are made by two team members blind to method output.
- **Status:** not launched. It waits for **ethics approval** (H-04) and the team's scenarios (H-05).

### 4.6 Human-side inputs received 2026-09-28 (D-041 to D-043)
- **Ethics:** faculty-advisor approval is complete, but its metadata is **still needed** for the ethics statement: approver name and role, date, and scope (survey only, or survey + rating study; the rating study is not assumed covered). Placeholder: `data/humanside/humanalign_phase1_review.md`.
- **Scenarios: LLM-assisted drafting (disclose in the paper).** The 150 scenarios were drafted with ChatGPT assistance row by row from each chunk, in the second person, as pre-document situations. **Each row requires team-lead review and approval before use.** The authentic queries are still typed by human respondents.
  - Agent leakage check on the draft: 141 PASS / 2 WARN / 7 REJECT. All rejects are EPFO rows, triggered by everyday words that are rare in this corpus (job, daughter, unmarried …).
  - 4 rows ask respondents to role-play professionals (S017, S025, S136, S147). A filter gap was found and fixed.
  - Suggested paper wording: *"Scenarios were drafted with LLM assistance from each help-page chunk, then reviewed, edited and approved by a team member, and screened automatically for document-term leakage (§6.3) and document echo (word-overlap review flag)."*
- **Scenario review status (D-044):** not approved.
  - 4 professional rows are marked skip, with deterministic replacement chunks drawn; the S025 replacement is itself professional and flagged for re-draw.
  - The 7 REJECT rows have ChatGPT rewrite candidates, which still fail the rare-term rule.
  - 2 WARN rows await review.
  - The official sheet is untouched.
- **Scenario review state (D-045):** the ChatGPT review recommendations were applied to the review artifact only, and **nothing is approved**.
  - 7 rewrites proposed. 4 still fail the rare-term rule on everyday words; 3 carry fidelity/role flags (S090, S091, S096).
  - 4 professional rows are replaced by deterministic chunk draws; S025 was re-drawn after a professional first draw. The replacement scenarios are agent (Claude Code) drafts.
  - Review-artifact check: 144 PASS / 4 REJECT / 2 WARN. The official sheet is byte-identical.
  - Paper note: the scenario pipeline had **two LLM assistants** (ChatGPT drafts and rewrites; the Claude Code agent drafted 4 replacements). Disclose both.
- **LID audit:** expanded from 30 to 100 hand-labelled examples (EN 20, ROM_HI 20, HINGLISH 20, DEV_HI 21, OTHER 19). OTHER is a non-target-script control (Kannada, Bengali, Tamil, Malayalam, Telugu, Gurmukhi, Gujarati, Urdu, mixed Kannada–Latin). Human final labels are pending. Agreement will be reported after they are frozen, with the review kept blind to system output.

## 5. Retrieval configurations and sanity check (Experimental setup)
**Configurations** (prereg §5.3; `configs/systems.yaml`; frozen):

| ID | Configuration |
|---|---|
| C1 | BM25 (k1 = 0.9, b = 0.4) |
| C2 | BM25 (k1 = 1.2, b = 0.75) |
| C3 | BGE-M3 dense (Ollama, `num_ctx` 8192, exact cosine) |
| C4 | RRF(C1, C3), k = 60 |
| C5 | C3 top-50, reranked |
| C6 | C1 top-50, reranked |
| C7 | C4 top-50, reranked |

BM25 is `bm25s` (lucene), lowercased, with English stopwords and no stemming. Passage text is title plus text; reranker max_length 512. There is no dev/held-out system split (D-022).

**E0 on D1 FiQA, all 648 queries** (run **`E0-20260928T003743-14d0b0`**, git `53896ec`):

| Config | nDCG@10 | MRR@10 | Hit@1 | Hit@5 | R@100 |
|---|---|---|---|---|---|
| C1 | 0.2380 | 0.2959 | 0.2238 | 0.3951 | 0.4946 |
| C2 | 0.2384 | 0.2924 | 0.2176 | 0.3889 | 0.5113 |
| C3 | 0.4128 | 0.5032 | 0.4182 | 0.6142 | 0.7125 |
| C4 | 0.3578 | 0.4344 | 0.3426 | 0.5509 | 0.6932 |
| C5 | 0.4354 | 0.5181 | 0.4352 | 0.6327 | 0.7125 |
| C6 | 0.3582 | 0.4548 | 0.3827 | 0.5679 | 0.4946 |
| C7 | 0.4328 | 0.5192 | 0.4352 | 0.6404 | 0.6932 |

- **Reference checks:** C1 0.2380 against BEIR's BM25 0.236 ± 0.02 ✅ (gate). C3 0.4128 against **0.413** (DRAMA, arXiv 2502.18460, Table 7, M3-BGE-Dense) ✅. So Ollama's F16 bge-m3 reproduces the published dense model, and no HF download is needed.
- **Latency** (same run; ms per query; median of 3 passes over a fixed 200-query sample; caches off): C1 0.97, C2 0.96, C3 16.21, C4 17.48, C5 649.83, C6 875.81, C7 783.91.
- **Compute** (`results/runs/<ds>/resources.jsonl`):
  - bge-m3 encoding: FiQA 57,638 passages in 1,715 s (33.6/s); CQA-Android 22,998 in 606 s (37.9/s); peak 0.62 GiB VRAM.
  - Reranking runs for C5/C6/C7 on 648 queries: 430 / 565 / 496 s, with a pair cache shared across the three; peak 1.42 GiB.
  - BM25 indexing: 3–11 s.
- **Held-out isolation:** CQA-Android received only corpus-side work (encoding, indexes, and a smoke run on 10 made-up queries). None of its real queries were retrieved or scored before the freeze (D-039). No seal was ever opened (`results/unseal_log.jsonl` does not exist).

## 5b. Synthetic pools, P3 (Methods; D-046 to D-048; `reports/phase_3.md`)
P3 generation proceeded in parallel with the remaining P1 human gate (**not passed**).
- **Settings:** `qwen3:4b-instruct-2507-q4_K_M` (digest `0edcdef3…`), temperature 0.7, top_p 0.9, max 64 tokens, seed 1000 + i. 800 source passages per dataset (seed 7). Templates B.1, B.2 and B.3 (verbatim). Filters: < 2 tokens; > 60% first-sentence overlap. Exact dedupe, then within-pool BGE-M3 cosine > 0.95. Every pool starts from a fresh model state (reproducibility, D-046).
- **Pools (final unique / 800):**
  - D1: P-std 727, P-persona 716, P-fewshot s0–s4 = 780 / 772 / 779 / 773 / 783;
  - D2: P-std 665, P-persona 630, P-fewshot s0–s4 = 688 / 668 / 658 / 666 / 676;
  - D4: P-std 586, P-persona 645.
  - Generation failures ≤ 0.13% per pool. The only failure is D2 pid 50091, a 7,885-token prompt against a 4,096 context; it was rejected, not truncated.
- **Resulting U = P-std ∪ P-persona:** D1 1,443; D2 1,295; D4 1,231.
- **P3 AUTO GATE, partial failure:**
  - **FAIL** on size ≥ 90%: 6/16 pools meet it (totals first mis-added as 7/16; corrected, D-047).
  - **PASS** on generation failures < 5%: 16/16; maximum 0.13%.
  - **Team decision (D-048):** the pools are retained as generated. The filter was not changed; no top-up, no context change, **no pre-registration amendment**. The retained pools are sufficient for the planned selection (the prereg sets no pool-size minimum).
  - This is a gate outcome, not a termination of the study.
- **Finding (descriptive; Discussion material):** plain LLM generation frequently restates the passage's first sentence, and the rate depends on corpus shape: 9–10% for FiQA answers, 14–21% for CQADupStack questions, 16–25% for HiAQ FAQ chunks. Exemplar-guided prompting with real R queries cuts it on FiQA (2–3.5%) but less on CQADupStack (14–18%). This echoes the spec's motivation (synthetic queries lean on document wording).
- **Deferred or stopped:**
  - D4 P-fewshot: deferred (no R yet).
  - P-std-G2: stopped (gemma3:4b not available).
  - P-docT5: stopped (a download that needs approval; full-scale tier).
  - D3: stopped (not downloaded).

## 5b-bis. Phase 1 human artefacts (D-051, D-052; Data section and Limitations)
- **LID audit:** 100 examples (30 original test cases plus 70 added), human final labels by the team. They differ from the audit draft on 2/100 (L54, L55: ROM_HI → HINGLISH). Final: DEV_HI 21, EN 20, HINGLISH 22, OTHER 19, ROM_HI 18. Labelled blind to classifier output.
- **Scenarios:**
  - 150 drafted with ChatGPT assistance and approved by the team (137 as drafted; 13 revised or replaced by team decision; 4 replacement chunks drawn deterministically).
  - **Deviation from spec §6.3** (not the prereg): S090 ("unmarried") and S091 ("military") are kept despite a rare-term REJECT, because the source-supported need requires the word (D-052, D-053). They are allowlisted explicitly; any other REJECT still blocks the survey build.
  - Disclose in the paper: "two scenarios retained despite the automatic rare-term rule, by documented human decision."

## 5b-ter. HiAQ collection method (D-056; Data section and Limitations)
- **Collaborative-sheet set (D-056):** real people typed answers to the 150 situations in a shared sheet (100 blocks × 6; each situation up to 4 times). It is kept as a separate set labelled `collab`, outside HiAQ's authentic survey data and the H2 test.
- **Limitation to disclose:** contributors could see others' answers in the shared sheet.
- **Amendments recorded 2026-10-10 (D-060):** A1 (§8 fallback: H2/H3 on D2 alone; H1 keeps Holm m = 9 with the D4 tests not rejected; H4 not estimable) and A2 (FiQA-only calibration, 100 items). The analysis is reported as **prereg-v2 = prereg-v1 + A1 + A2**; prereg-v1's primary analysis was not executable because its calibration set could not be constructed. A2 was recorded before any calibration annotation or result.
- **Amendment A3 recorded 2026-10-14 (D-062): no human calibration.**
  - Prompt v0 (spec B.4 verbatim) is final; M1 and M1-iso score with equal weights on (H, S, P, C*); the gate stays F ≥ 4.
  - κ, ρ, the NNLS weights and every `cal-test` statistic are not executed.
  - The analysis is **prereg-v3 = prereg-v1 + A1 + A2 + A3**; prereg-v2's primary analysis is not executable.
  - **The paper must state** that the judge was not human-calibrated, and must make no claim about judge–human agreement.
  - Recorded before any bulk judging output, selection or held-out result.
- **Reduced scope (D-059):** authentic HiAQ collection is closed at zero.
  - The planned confirmatory test is H2/H3 on D2 alone (amendment A1 rev 2, draft).
  - Calibration is FiQA-only, 100 items (amendment A2; *superseded by A3: no annotation is run*): smaller `cal-dev` (60), English-only, and the C weight is not separately identifiable.
  - HiAQ is reported as a corpus, scenario set and synthetic-pool pilot. `X-team150` is external synthetic data and exploratory only.
  - Disclose that v1's calibration was infeasible and A2 was recorded before any calibration data existed.
- *(Superseded by D-059:)* - **Authentic HiAQ (D-057):** collected with the Google Forms, one form per respondent. The initial tranche targets 25 completed forms (up to 150 answers) and supplies the 30 HiAQ calibration items. The collaborative sheet is exploratory only.
- If HiAQ's authentic set stays below 300 eligible queries after cleaning, prereg §8 applies: H2/H3 are evaluated on D2 alone and D4 is reported as a pilot (*superseded:* triggered, because collection was closed at zero (D-059); recorded as amendment A1 on 2026-10-10, D-060).

## 5c. Neighbours and judge, P4 (Methods; D-049; `reports/phase_4.md`; in progress, uncommitted)
- **Neighbours (P4.1):** bge-m3 dense cosine against R_s only → top-30 → bge-reranker-v2-m3 (max_length 128) → top-5. D1 and D2, split seeds 0–4. **0 near-duplicates** (cos > 0.97) in every seed. Median nn_max_cos: D1 0.57, D2 0.64. D4 is pending its R (after the survey).
- **Judge (P4.2):** v0 = spec B.4 verbatim; temperature 0, seed 0, num_ctx 4096; 2 permutations, plus a 3rd on Δ ≥ 2, then the median; M1-iso without neighbours.
- **[ENGINEERING] pilot** `P4-judge-pilot-20260928T230859-1f27d5`: 38 calls on 12 D1 queries. 100% valid JSON first try; 1.50 s per call; 3rd-permutation rate 2/12; deterministic 4/4.
- **Bulk judging (P5.1; `ha judge <ds>`, `src/ha/judge/bulk.py`):**
  - M1-iso once per U query; M1 for each split seed with the R_s neighbours; near-duplicates are skipped (§5.2).
  - Every stage starts from a fresh model state (D-046) and visits queries in a fixed order.
  - Failure, gate-pass and third-permutation rates go to the registry.
  - **D1 FiQA** (run `P5.1-judge-fiqa-20261015T064926-d3c37d`; 1,443 U queries): 0 failures, 0 retries.
    - Gate pass: M1-iso 76.2%; M1 66.1–67.8% across seeds. Third-permutation rate about 13%.
    - With real neighbours, human-likeness H falls from 4.52 to about 3.75; S and P also fall.
    - Seeds 0–3: 848 queries pass in every seed, 356 in none, 239 mixed.
    - Persona queries pass less often than standard ones (M1 s0: 61% vs 73%).
    - N = |A| = 388 binds in every seed (about 970 M1 passes).
    - About 30% of gate passes tie at the maximum equal-weight score, so selection needs a seeded tie-break, fixed before the freeze.
  - **Selection (P5.2; `src/ha/select/sets.py`; D-063):**
    - N = min(|A|, M1 passes, M1-iso passes); near-duplicates excluded.
    - B0 is a random draw and B1 is stratified random, 20 draws each. M1 and M1-iso use the gate, then greedy selection by equal-weight score within k-means strata of R (k = 10).
    - Quotas by largest remainder; a seeded tie-break; dedupe at cosine 0.90.
    - FiQA: N = 388 = |A| in every split.
  - **Evaluation (P5.3; `src/ha/eval/sets.py`):** τ_b over C1–C7 from per-query nDCG@10, with B0/B1 averaged over draws; τ_AP, winner agreement, top-3 overlap, MAE, bias; the §7 paired bootstrap (B = 1,000, seed 20260927), with Δ̄ for M1 vs B0, B1 and M1-iso and seed-averaged τ CIs per method.
  - **H1 (`src/ha/eval/h1.py`; D-064):** KS + W1 for length and gold-passage overlap, JS permutation test for topic, max p over seeds, Holm m = 9.
  - **Freeze mechanics:** `FROZEN.json` records the judge, weights, data hashes per dataset and seed, the code sha and the bootstrap settings. The lead's signature hashes the record. `heldout_A` opens only while that record is signed and unchanged.
  - **[DESCRIPTIVE] Synthetic queries are easy for retrieval:** mean nDCG@10 of U against its source passage (C1/C2/C3/C4) is FiQA 0.675/0.712/0.701/0.755 and Android 0.934/0.931/0.969/0.969, vs real FiQA 0.238/0.238/0.413/0.358 (E0). Android's near-ceiling scores leave small system gaps, a risk for τ on synthetic sets. Source: `results/runs/<ds>/C*/U.run.parquet`.
  - **Real vs synthetic style (D1, descriptive):** median 10 vs 14 words; starts with a question word 57% vs 83%; lowercase start 2% vs 49% (persona pool).
- Observation (n = 12, uncalibrated; **not** a result): the no-neighbour judge rated exam-style synthetic queries H = 5; with real neighbours, H and S dropped for document-echo queries. Calibration will test this properly.
- **Blocked:** P4.3–P4.5 (HiAQ CAL-auth tranche; human annotation).
- **Calibration display (D-050; team decisions):**
  - Authentic calibration items show a real qrels passage: highest grade, then a fixed sha256 tie-break. No length criterion; nothing chosen by hand.
  - Annotators see 5 neighbours retrieved by the frozen procedure from **R_s0 minus all 60 CAL-auth items**. This preserves the §4.3 blinding: otherwise 42/70 FiQA synthetic items would show an authentic calibration query as a "real user query". The judge side is unchanged.
  - D4 domain text is in `configs/domains.yaml`.
  - Spreadsheet (CSV) instead of Label Studio, with raw, adjudication and final labels as separate files.
  - The practice pack is exported; the 200-item pack is blocked on the HiAQ CAL-auth half and its qrels. *(Superseded: under A2, recorded 2026-10-10 (D-060), the pack is 100 FiQA items.)*
- **Paper, Methods (annotation):** "Authentic items were shown a relevance-judged passage (highest qrels grade; deterministic hash tie-break); annotators saw five reference queries retrieved from the reference pool after removing all authentic calibration items, so that no item's origin was revealed."

## 6. Results (to be filled after the freeze)

### 6.1 CONFIRMATORY: H2 and H3 on D2 CQADupStack-Android (prereg-v3; freeze D-066)
- **Run** `P5-eval-heldout-cqa_android-20261016T110347-f9f1b6`, at commit `4bebf60`. The code is identical to the frozen analysis commit `12c8111`. `heldout_A` was unsealed after the signed freeze (`results/unseal_log.jsonl`).
- **Seed-averaged τ_b with A,** with 95% bootstrap CIs:

  | Set | τ_b | 95% CI |
  |---|---|---|
  | M1 | 0.779 | 0.478–0.791 |
  | M1-iso | 0.543 | 0.326–0.657 |
  | B0 | 0.619 | 0.537–0.628 |
  | B1 | 0.581 | 0.506–0.589 |

  - Per seed, M1 vs B0: 0.81/0.63, 0.71/0.58, 0.78/0.59, 0.81/0.70, 0.78/0.60. M1's point estimate is higher in all 5 seeds.
- **Bootstrap (§7; B = 1,000):**
  - Δ̄(M1 − B0) = +0.160 (−0.122, 0.219), p = 0.211;
  - Δ̄(M1 − B1) = +0.198 (−0.085, 0.255), p = 0.106;
  - Δ̄(M1 − M1-iso) = +0.236 (−0.070, 0.370), p = 0.079.
- **Decision:**
  - **H2 not supported** (needs p_B0 < 0.05 and p_B1 < 0.05).
  - **H3 not supported** at α = 0.05.
  - Report as: positive point estimates, not statistically significant.
- **Other measures:**
  - τ_AP: M1 0.66, B0 0.51, B1 0.45, M1-iso 0.38.
  - Top-3 overlap: M1 0.93, B0 0.76.
  - MAE is about 0.52–0.54 for every method: absolute scores are inflated about 2× (synthetic nDCG@10 about 0.93–0.99 vs authentic 0.37–0.49).
- **Caveat for the Discussion:**
  - Synthetic scores sit near the ceiling on D2, so system gaps are tiny.
  - Bootstrap resampling of such sets often flips near-tied pairs, which pulls τ down. M1's point estimate sits near the top of its own CI.
  - This, together with τ_b's coarseness over 7 systems (§7.6), limits power.

### 6.2 H1 (D1, D2; D4 at p = 1 under A1), run `H1-20261016T110407-169410`
- **Holm-rejected (m = 9):**
  - length: D1 and D2;
  - gold-passage overlap: D1 and D2;
  - topic: D1 only (D2 topic p = 0.047 is not rejected after Holm).
- **Effect sizes (seed-mean):**
  - length W1: D1 3.7 words (14.6 vs 11.1), D2 7.8 words (16.9 vs 9.0);
  - overlap W1: D1 0.10, D2 0.23;
  - topic JS: D1 0.047, D2 0.022.
- **τ_b(B0, A):** D1 0.534 (0.44–0.59), D2 0.619 (0.54–0.63). Both are below 0.8, so the criterion is met on 2 of 3 datasets (D4 counted as not met, A1).
- **H1:** supported for length, overlap and the τ criterion; topic concentration differs only on D1.

### 6.3 [EXPLORATORY] D4 HiAQ pilot (collected batch "group a", D-065, D-067)
- **Run** `P5-eval-exploratory-hiaq-20261017T064411-f305b8`. 146 collected queries, split per seed into R 57 and A 89; N = 89 (binding: A); k = 3 strata. Judging had 0 failures (1,231 U queries × 6 stages).
- **Seed-averaged τ_b with A:** B0 0.810; B1 0.761; M1-iso 0.691; **M1 0.460**.
  - Per seed, M1 vs B0: 0.59/0.72, 0.14/0.85, 0.33/0.86, 0.52/0.76, 0.71/0.86.
- **Bootstrap:** Δ̄(M1 − B0) = −0.350 (−0.629, −0.131); Δ̄(M1 − B1) = −0.301 (−0.569, −0.060). **The curated set agrees worse than random on HiAQ.**
- **Pattern (split seed 0):**
  - A ranks dense above BM25 (C3 0.541 vs C1 0.398), and so does B0 (C3 0.763 vs C1 0.642).
  - M1 reverses it (C1 0.774 vs C3 0.741): the judge's picks favour lexical retrieval.
  - *(Withdrawn 2026-10-17:* the guess that official terms drive this is not supported. See §6.4; collected queries have low passage overlap.)
- **Caveats:** small A (89) and single-chunk qrels; provenance is the team's statement (D-065); exploratory only.

### 6.4 [EXPLORATORY] Mechanism: gold-passage overlap (run EXP-overlap-20261017T084222-48fccf, after the freeze)
| Seed mean | D1 FiQA | D2 Android | D4 HiAQ |
|---|---|---|---|
| Overlap, authentic A | 0.54 | 0.61 | 0.30 |
| Overlap, B0 / B1 | 0.64 / 0.65 | 0.84 / 0.84 | 0.80 / 0.77 |
| Overlap, M1-iso / M1 | 0.67 / 0.70 | 0.85 / 0.86 | 0.80 / 0.73 |
| nDCG@10 C1 − C3 (BM25 − dense), A | −0.17 | −0.11 | −0.12 |
| C1 − C3, B0 / M1 | −0.03 / +0.01 | −0.04 / −0.03 | −0.08 / +0.03 |
| Spearman ρ, M1 score vs overlap (gate passes) | 0.11 | 0.14 | −0.03 |
- Synthetic queries share far more vocabulary with their gold passage than authentic queries do (+0.10, +0.23, +0.50). That removes dense retrieval's advantage over BM25, which is the main ranking difference.
- On D1 and D2 the realism judge does not penalise overlap (weak positive ρ), so M1 selects the highest-overlap queries.
- On HiAQ, M1's lexical tilt is **not** explained by overlap (its picks overlap less than B0's). This is an open question.

### 6.5 [EXPLORATORY] Judge-free selectors (D-068, specified before any run)
- **Runs:** `EXP-v2-fiqa-20261017T121652-79580b`, `EXP-v2-cqa_android-20261017T121722-9a1e31`, `EXP-v2-hiaq-20261017T121741-cb5244`.
- **Seed-mean τ_b** (OVL = overlap-matched random; NN = nearest-to-real):

  | Dataset | B0 | B1 | M1 | OVL | NN |
  |---|---|---|---|---|---|
  | D1 FiQA | 0.534 | 0.504 | 0.448 | **0.636** | 0.467 |
  | D2 Android | 0.619 | 0.581 | 0.779 | **0.802** | 0.562 |
  | D4 HiAQ | 0.810 | 0.761 | 0.460 | **0.923** | 0.733 |

- **OVL vs B0:** +0.102 (0.094, 0.130), +0.183 (0.120, 0.228) and +0.113 (0.070, 0.170); p < 0.001 on all three. OVL vs B1 is also positive on all three, and OVL vs M1 is +0.19, +0.02 (p 0.09) and +0.46.
- **NN** is not better than random on any dataset (Δ vs B0 −0.07, −0.06, −0.08).
- **Manipulation check:** OVL matched R's overlap on D1 (0.54–0.56 vs 0.54–0.55). On D2 and D4 it could not, because too few low-overlap synthetic queries exist (D2: 7 and about 60–96 in the two lowest bands vs quotas of about 90–110 and about 70–80). OVL's overlap was 0.74–0.76 vs R 0.61–0.64 on D2, and 0.43–0.47 vs 0.28–0.30 on D4.
- **Caveat:** the OVL idea came from the post-freeze overlap diagnosis, which looked at A on all datasets. OVL itself uses only R at selection time, but the result is hypothesis-generating and needs a fresh dataset to confirm.

### 6.6 CONFIRMATORY REPLICATION R1 (D-069, D-070; `docs/replication_prereg.md`)
- **Sequence:** protocol (`061524a`) → code (`72c289c`) → data built, with counts matching BEIR (D5 17,405/506; D6 3,633/323) → replication freeze (`d489fe5`) → unseal and authentic runs (`06d46c3`) → evaluation, once.
- **Run** `R1-20261017T194854-d1b74e`:
  - Pooled Δ̄(OVL − B0) = +0.062 (0.016, 0.091), p = 0.005.
  - Pooled Δ̄(OVL − B1) = +0.081 (0.015, 0.102), p = 0.004.
  - **R1 supported.**
- **Per dataset:**
  - D5 Webmasters: OVL 0.529 vs B0 0.381 and B1 0.332, higher in all 5 seeds.
  - D6 NFCorpus: OVL 0.594 vs B0 0.619 and B1 0.628, no gain.
- **Context, not tested:** NFCorpus authentic queries have 38.2 relevant passages per query vs 2.8 on D5; synthetic queries have 1.
- **Manipulation check:**
  - D5: OVL overlap 0.68–0.69 vs R 0.54–0.59 vs random 0.80.
  - D6: OVL 0.63–0.66 vs R 0.54–0.64 vs random 0.72–0.75.

### 6.7 [EXPLORATORY] D-071 analyses (specified before running)
- **E-density** (`EXP-density-20261017T230906-1f1a17`): authentic qrels sparsified to one passage per query.
  - D6 NFCorpus: OVL 0.756 vs B0 0.781 vs B1 0.835; Δ(OVL − B0) = −0.026 (−0.048, 0.099), p = 0.28.
  - **The pre-stated prediction failed: relevance density does not explain D6's null.**
  - D5 Webmasters is unchanged (Δ = +0.148).
- **E-metrics** (`EXP-metrics-20261017T231816-deeada`):
  - OVL > B0 on D1, D2 and D4 under MRR@10, Recall@100 and Hit@5 (Δ +0.03 to +0.19; p ≤ 0.003).
  - On D5, MRR and Hit@5 are positive (+0.14, +0.09) and Recall@100 is about 0.
  - On D6, MRR and Hit@5 are negative (−0.06, −0.03) and Recall@100 is positive (+0.08).
  - M1 − B0 is n.s. under MRR and Hit@5 everywhere. On D2 it is positive under Recall@100 (+0.155, p = 0.027; one of 3 metrics, no correction).
  - τ undefined for D4 Recall@100 M1/M1-iso (all systems tied).
  - Caveat: some p-values sit on the opposite side of near-zero Δ (e.g. D5 Recall@100: Δ = −0.004, p = 0.011). This reflects the bootstrap's downward bias for near-tied systems (§9).
- **E-dims** (`EXP-dims-20261017T231909-6571a9`), selecting by a single judge dimension. τ_b:

  | Dataset | SEL-F | SEL-H | SEL-S | SEL-P | M1 | B0 |
  |---|---|---|---|---|---|---|
  | D1 | 0.467 | 0.486 | 0.352 | 0.486 | 0.448 | 0.534 |
  | D2 | 0.581 | 0.669 | 0.676 | 0.581 | 0.779 | 0.619 |
  | D4 | 0.622 | **0.219** | 0.542 | 0.634 | 0.460 | 0.810 |

  - No single dimension beats random.
  - On D2 the combined score (M1) is best.
  - **On HiAQ, human-likeness alone is the worst selector (τ_b 0.22 vs 0.81 random):** the H judgement drives M1's HiAQ failure.

### 6.8 CONFIRMATORY REPLICATION R2 (D-072; `docs/replication2_prereg.md`)
- **Sequence:** protocol (`f86df31`) → setup (`dee5440`) → data built, with counts matching (D7 5,183/300; D8 16,705/804→700; D9 32,176/876→700) → replication freeze (`5306510`) → unseal and authentic runs (`4f190ba`) → evaluation, once.
- **Run** `R2-20261019T001603-e7e628`:
  - Pooled Δ̄(OVL − B0) = +0.057 (0.025, 0.085), p = 0.001.
  - Pooled Δ̄(OVL − B1) = +0.054 (0.032, 0.082), p < 0.001.
  - **R2 supported.**
- **Per dataset** (τ_b OVL / B0 / B1):
  - D7 SciFact: 0.536 / 0.677 / 0.663. OVL is worse in all 5 seeds.
  - D8 Mathematica: 0.871 / 0.731 / 0.773. Better in all 5.
  - D9 Programmers: 0.628 / 0.457 / 0.437. Better in all 5.
- **The a priori expectation failed:** I expected SciFact to favour OVL, since it has sparse qrels.
- **Across R1 and R2:** OVL helps on all three CQADupStack forums and not on NFCorpus or SciFact. Untested: their authentic queries are titles or claims, not user questions.

### 6.0 [DEVELOPMENT] D1 FiQA (before the freeze; no hypothesis decision rests on these)
- **Agreement with A** (run `P5-eval-dev-fiqa-20261016T104019-aed308`, commit `d6b0a9a`). Seed-averaged τ_b over C1–C7, with 95% bootstrap CIs:

  | Set | τ_b | 95% CI |
  |---|---|---|
  | B0 | 0.534 | 0.44–0.59 |
  | B1 | 0.504 | 0.41–0.57 |
  | M1-iso | 0.467 | 0.33–0.54 |
  | M1 | 0.448 | 0.35–0.56 |

  - Δ̄(M1 − B0) = −0.087 (−0.145, 0.011), p = 0.96; Δ̄(M1 − B1) = −0.056, p = 0.86; Δ̄(M1 − M1-iso) = −0.019, p = 0.47.
  - Winner agreement is 0.2 and top-3 overlap 0.67 for every method.
- **Mechanism (split seed 0 score vectors):**
  - A ranks dense above BM25: C3 0.415 vs C1 0.224.
  - The synthetic sets rate them about equal: M1 C1 0.730 vs C3 0.726. Absolute scores are about 0.4 higher (bias about +0.40).
  - Synthetic queries reuse their source passage's words, which favours lexical retrieval. Realism curation does not remove this.
- **H1 preview** (run `H1-…`, `results/eval/H1-fiqa.json`): length 14.6 vs 11.1 words, gold overlap 0.64 vs 0.55, topic JS 0.071. All p < 0.001 in every seed. τ_b(B0, A) = 0.53 < 0.8.
- **Handling:** nothing in §5 was changed because of these results (prereg §3.3). The design is frozen as built.
_H1–H4 results go here, from the registry only, after the freeze gate._

## 7. [EXPLORATORY] Authentic noise ceiling for τ_b (D-040; context only, not a hypothesis)
Data: D1 FiQA only, reusing the E0 per-query nDCG@10. Configurations C1–C7. Run **`EXP-noise-ceiling-20260928T011917-5bc429`**; the first run, `…011823-654ce0`, covered (a)–(c) only.

| Variant | Mean τ_b | Median | 2.5–97.5% | Share with τ_b = 1 |
|---|---|---|---|---|
| (a) Complementary random halves (324/324; 1,000 partitions; seed 20260928) | 0.779 | 0.810 | 0.714–0.905 | 0.0% |
| (d) Independent resamples (324 each, with replacement; 1,000; seed 20260929) | 0.857 | 0.905 | 0.619–1.000 | 14.6% |
| (b) Frozen splits R_s vs A_s, seeds 0–4 | 0.905 / 0.810 / 0.714 / 0.810 / 0.524 | — | — | — |

- **Why τ_b falls below 1:** 3 of the 21 pairs are near-ties on authentic FiQA (E0 Δ nDCG@10: C1–C2 0.0004, C4–C6 0.0004, C5–C7 0.0027). Every other pair almost never flips.
- **Methodological caveat, a finding in itself:** complementary halves are anti-correlated on every pair difference, because their means sum to the full-set mean. So near-tie pairs flip in about 95% of random halves (C4–C6 0.959, C1–C2 0.941), against about 50% for independent samples (C1–C2 0.529, C4–C6 0.475). Variant (a) therefore **understates** agreement. Variant (d) is the matching reference when an independent synthetic set is compared with A.
- **Interpretation for the paper:** even two independent sets of *authentic* FiQA queries of this size reach a median τ_b of only about 0.9 over C1–C7. This gives a scale for the synthetic-vs-authentic τ_b values. It does not change H2, the configurations, tuning, selection or the freeze.

## 8. Deviations from the original spec (all logged; for Methods and Limitations)
| Spec | What we did | Why |
|---|---|---|
| About 20 retrieval systems, dev/held-out split | 7 configurations from the locked stack, no system split (D-022, D-023) | Prototype scope; a split of 7 is meaningless; isolation comes from held-out datasets |
| llama3.1:8b generator; qwen3.5:4b judge | qwen3:4b-instruct-2507 for both (D-021, D-028) | Locked local stack; self-preference disclosed |
| 4 datasets | D1 (dev), D2 and D4 (held-out); D3 MIRACL-Hindi optional (D-026) | Compute and scope |
| HF BGE-M3 (dense + sparse) | Ollama bge-m3, dense only (D-024) | No download; validated by E0 |
| 1,500 source passages | 800 per dataset (prereg §5.2) | Judge budget on the 3050 |
| HiAQ sources: DigiLocker, UPI, IRCTC, UIDAI, Income Tax | Income Tax + EPFO (D-031) | Robots and policy compliance; JS rendering |
| Spec leakage check only | Plus a WARN signal (D-036) | The spec's check missed 14 of 20 obvious document echoes |
| Human-calibrated judge (κ, prompt iteration v0–v4, NNLS weights) | Uncalibrated prompt v0 with equal weights (amendment A3, D-062) | Project lead's decision: no human annotation |

## 9. Limitations (running list)
- The same model generates and judges (possible self-preference). Mitigation: the M1-iso control. No human calibration was run (A3).
- **The judge is not human-calibrated (A3).** Its realism scores are not validated against human ratings, and the equal weights are fixed a priori.
- Synthetic qrels are sparse (source passage only). Rahmani et al. found τ can collapse under sparse qrels; E11 (pooled qrels) is high-priority optional.
- τ_b over 7 configurations is coarse. 3 near-tie pairs cap agreement even for authentic-vs-authentic comparisons (§7).
- The HiAQ corpus is tax-dominated: 90% of chunks are Income Tax. Scenarios are balanced 75/75.
- HiAQ queries are scenario-elicited, not organic; the respondent pool is student-heavy.
- D4 pooled qrels use C1, C3, C4 and C5, which favours those configurations; disclosed.
- LID is rule-based and its test labels are provisional (H-09); D4 gets a human check.
- Scenarios were LLM-assisted (ChatGPT) drafts, human-approved. Possible priming of respondents' phrasing by LLM-written situations.
- Single laptop GPU; everything is runnable locally; latency was measured on this hardware.

## 10. Ethics and data governance
- No personal data is collected (no emails, sign-in or names). PII is stripped at ingest.
- **Ethics:** the faculty advisor gave informal approval, as reported by the team (D-041). The approver, date, scope and conditions were not recorded (team-lead decision, D-055). The paper states only this; rating-study (E8) coverage is not documented.
- Scraping followed robots.txt and the live website policies, at 1 request every 2 s with every request logged. Sources that forbid automated access or reproduction were excluded.
- Release plan: HiAQ queries, qrels, chunk IDs and **source citations**. Income Tax restricts linking without permission, so links are not published unless permission is obtained. Chunk text is released only where the policy permits (EPFO and Income Tax allow reproduction with acknowledgement).

## 11. Chronology
- **2026-09-27:**
  - Plan v1 → team decisions → prototype scope (plan v2).
  - Environment set up on D:.
  - Qwen3-4B thinking-model finding → instruct model approved.
  - Pre-registration reviewed (10 criteria) → frozen as `prereg-v1`.
  - HiAQ vetting, crawl and chunking; LID; splits; survey tooling.
- **2026-09-28:**
  - P3: engine, smoke test, pilot, full run of 16 pools. Gate: FAIL on size (6/16), PASS on failures (16/16); pools retained (D-046 to D-048).
  - Scenario sheet set to 75/75.
  - Phase 2: corpus encodings, E0 (gate passed), latency.
  - Exploratory noise ceiling (with the complementary-halves caveat).
- **2026-10-10:** authentic HiAQ collection closed at zero (D-059); amendments A1 and A2 recorded (D-060).
- **2026-10-14:**
  - guidelines finalised (D-061); A2 pack exported;
  - amendment A3 recorded: no human calibration (D-062);
  - P5.1 bulk-judging runner built.

## 12. Still needed for the paper
- Ethics approval; the 150 scenarios; the reviewed LID labels.
- Bulk judging (P5.1), selection, dev results on D1, then the freeze. There is no calibration κ or ρ (A3).
- Held-out results H1–H4, with bootstrap CIs.
- Figures: pipeline diagram, τ_b with CIs, the per-language gap.
- Novelty check #2 before submission.
