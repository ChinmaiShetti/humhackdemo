# Pre-registration: HumanAlign-RAG (prototype)

**Status: FROZEN on 2026-09-27 (git tag `prereg-v1`; D-016, D-030).** From here the file is append-only: any change goes in a dated amendment at the bottom. A methodological change is released as a separate **v2** analysis, and **v1 stays the primary pre-registered analysis**.

- **Drafted:** 2026-09-27. Revision 3 (same day) tightens the locking rules after the team's 10-point review (D-029). **Frozen:** 2026-09-27, after the team approved revision 3 and all six D-029 choices.
- **Team:** two-member RVCE team (AIML, IS353IA).

## 0. What this document locks

| # | Commitment | Where |
|---|---|---|
| 1 | One primary test: H2, pooled over the two held-out datasets, framed as ranking **agreement** | §1, §2, §7 |
| 2 | Per-dataset H2 results are secondary breakdowns with no significance claim | §2, §7.5 |
| 3 | Authentic held-out queries never touch calibration, prompts, anchors, weights or settings | §3.3 |
| 4 | M1-iso is the pre-specified control for the neighbour-grounding contribution | §2 (H3), §5.2 |
| 5 | The calibration procedure is fixed before any calibration result is seen | §4 |
| 6 | Prompt and anchor changes may use `cal-dev` only | §3.3, §4.5 |
| 7 | The whole judge and pipeline configuration is frozen before held-out evaluation | §6 |
| 8 | The primary metric and statistical procedure cannot change after results are seen | §7, §7.7 |
| 9 | Null and negative results are kept and reported | §8 |
| 10 | Model identities (full digests) and retrieval configurations are fixed | §5.1, §5.3 |

---

## 1. Question and primary endpoint

**Question.** Can neighbour-grounded LLM judging and curation make synthetic RAG evaluation queries better proxies for authentic human information needs?

**Primary endpoint (the single primary test; everything else is secondary or exploratory).**
- **Metric per query:** nDCG@10 computed with `ranx` (metric `ndcg@10`, linear gain, graded qrels).
- **Set-level metric:** the mean over the queries in the set.
- **Agreement:** Kendall τ_b (`scipy.stats.kendalltau`, variant `b`) between the 7-configuration ranking (C1–C7) on a query set X and on the held-out authentic test set A of the same dataset and split seed.
- **Pooling:** averaged over split seeds 0–4, then over the two held-out datasets D2 (CQADupStack-Android) and D4 (HiAQ-Services).
- **Test:** one-sided, α = 0.05, with the decision rule in §7.4.

The judge's own scores and its agreement with humans are secondary outcomes.

## 2. Hypotheses

| | Hypothesis | Status |
|---|---|---|
| **H2** | Pooled over D2 and D4, the configuration ranking produced by the curated set **M1** agrees more closely with the ranking produced by the authentic held-out queries A than the rankings produced by **B0** do, and than those produced by **B1** do (τ_b(X, A) over C1–C7). D2 and D4 are also reported separately, as secondary breakdowns. | **Confirmatory; the single primary test** (5 seeds) |
| **H3** | Neighbour grounding matters: pooled over D2 and D4, τ_b(M1, A) > τ_b(M1-iso, A). **M1-iso is the pre-specified control** for the neighbour-grounding contribution: the same judge and pipeline, without neighbours (§5.2). | Key secondary (5 seeds) |
| **H1** | Raw synthetic queries (B0) differ from authentic A in query length, lexical overlap with the gold passage and topic concentration, and give τ_b(B0, A) < 0.8 on at least 2 of the 3 datasets (D1, D2, D4). | Secondary, descriptive (5 seeds) |
| **H4** | On D4, the gap between B0 and A is largest for the ROM_HI and HINGLISH classes, and M1 reduces it most there in absolute terms. | Secondary (5 seeds; a re-slice of the H2 runs) |

**Tests for the secondary hypotheses**
- **H3:** the §7 procedure with M1-iso as the comparator, pooled over D2 and D4, one-sided α = 0.05. Reported as a secondary result; it does not affect the H2 decision.
- **H1:**
  - Length and gold-passage overlap: two-sample KS test, plus W1 distance with a bootstrap CI.
  - Topic concentration: JS divergence over R-derived clusters against a 1,000-permutation null.
  - Holm correction across the 3 datasets × 3 measures.
  - τ_b < 0.8 is judged on the seed-averaged point estimate and reported with its CI.
- **H4:** per-class gap = 1 − τ_b(X_c, A_c) and the MAE of configuration scores, where both sets are restricted to class c.
  - A class is analysed only if A_c has at least 50 queries averaged over seeds; otherwise it is reported descriptively.
  - H4 is descriptive, with bootstrap CIs and no significance claim.

**Optional robustness analyses** are not pre-registered hypotheses. They are listed in `PLAN.md` §1 (ablations; baselines B2–B5; budget curve; `qwen3:8b` judge; few-shot and rewrite variants; E11 pooled qrels; E8 human test). **Every one that is run is reported in full, whatever its result**, labelled exploratory and with its seed count.

## 3. Data, splits and isolation

### 3.1 Datasets
| ID | Dataset | Role | Qrels |
|---|---|---|---|
| D1 | FiQA-2018 test (648 queries, 57,638 passages; HF `BeIR/fiqa` rev `979c07a7…`, `BeIR/fiqa-qrels` rev `252958f2…`) | development | official |
| D2 | CQADupStack-Android (699 queries, 22,998 passages; HF `mteb/cqadupstack-android` rev `e03f271e…`) | held-out | official |
| D4 | HiAQ-Services (our survey; EN / ROM_HI / HINGLISH) | held-out | §5.4 |

If a dataset has more than 700 authentic queries, 700 are sampled with seed 13.

**HiAQ inclusion rules, fixed now and applied before any split:**
- strip PII (phone, email, 12-digit Aadhaar-like numbers, PAN);
- drop empty answers and answers that are not queries or are off-topic for their scenario. Two team members decide, blind to any method output, and record the reason;
- keep typos; normalise whitespace only.

### 3.2 Splits
- R 40% / A 60% for split seeds 0–4, stratified by language class and by k-means topic cluster. The clusters are computed on all authentic queries of the dataset and are used only for splitting.
- `CAL-auth` (§4.1) is placed in R for every seed and never enters A.

### 3.3 Isolation rules: what each part of the data may be used for

| Data | Allowed use | Never used for |
|---|---|---|
| **A of D2 and D4** | Evaluation only, after the freeze gate (§6), through the sealed loader (`heldout_A`) | calibration items, anchors, prompt edits, weights, gate or settings, pool generation, neighbours, strata, qrels decisions (except as queries for D4 pooling, §5.4, where no metric is computed) |
| R of any dataset | neighbours; strata; `CAL-auth` items (FiQA and HiAQ only) | — |
| **`cal-dev`** (120 items) | **the only data allowed to drive** prompt versions, anchor examples and judge weights (§4) | — |
| `cal-test` (80 items) | unsealed once, after the prompt version and weights are chosen; reporting only | any choice |
| D1, all of it | pipeline development and sanity checks (E0; dev runs of E1). Any change to a §5 setting made because of D1 results must be recorded as a dated amendment **before** the freeze. | prompt, anchor or weight changes (those are `cal-dev` only) |
| Synthetic pools of D2 and D4 | generated and judged before the freeze; they use only the corpus and R | — |

D2 contributes no calibration items, so it is fully held out from the judge. D4 contributes `CAL-auth` items (in R) and synthetic calibration items, and this is disclosed (D-013).

## 4. Calibration procedure (fixed before any calibration result is seen)

1. **Items (200).**
   - 60 authentic `CAL-auth`: 30 FiQA and 30 HiAQ. The HiAQ half comes from the first survey tranche, stratified by language class. Both halves are drawn with seed 17, from R.
   - 140 synthetic: 70 FiQA and 70 HiAQ, each half split equally between P-std and P-persona, drawn with seed 17 from the seed-0 pools.
   - Neighbours come from R of seed 0. An authentic item never has itself, or anything with cosine > 0.97 to it, as a neighbour.
2. **Split.** 120 `cal-dev` and 80 `cal-test`, stratified by dataset × authentic/synthetic, seed 17. `cal-test` is sealed (`cal_test`).
3. **Annotation.**
   - `docs/annotation_guidelines.md` (spec Appendix C, expanded) is finalised before annotation starts.
   - The two annotators first rate 20 practice items together. These are extra FiQA synthetic items, not among the 200.
   - Both then rate all 200 items independently on F, H, S, P, C and overall realism (1–5). They are blind to whether an item is synthetic or authentic and to the judge's output, and see the items in random order.
4. **Agreement and the human reference.**
   - Quadratic-weighted Cohen's κ is computed per dimension.
   - If κ < 0.5 on any dimension, the guidelines are revised **once** and 40 `cal-dev` items are re-annotated. Both κ values are reported.
   - The human reference is the mean of the two ratings. Disagreements of 2 or more are discussed, and the settled value replaces the mean.
5. **Prompt versions.**
   - v0 is spec Appendix B.4 verbatim. At most v1–v4 follow.
   - **Allowed changes:** instruction wording; rubric anchor wording; and from v1 onwards, 3 anchor examples (one high, one mid, one low; for D4 prompts, one of them Hinglish) drawn **from `cal-dev` only**.
   - **Fixed:** the output schema, the five dimensions, the model and the decoding settings.
6. **Selecting a version.**
   - Criterion: the mean Spearman ρ between judge and human reference over F, H, S, P, plus C where not null, on `cal-dev`, **excluding the items used as anchors**. Ties go to the earlier version.
7. **Weights.**
   - Non-negative least squares of human overall realism on the judge's H, S, P and C (C imputed as the mean of H, S and P when null), fitted on all `cal-dev` items. The 5-fold CV ρ is reported.
   - Fitted separately, with the same procedure, for M1 and for M1-iso (M1-iso on its own no-neighbour judgements).
8. **Fixed a priori, not tuned:** the gate F ≥ 4. Permutations: 2, plus a third if any dimension differs by 2 or more; the median is taken per dimension.
9. **No calibration outcome stops the study.** Whatever ρ and κ turn out to be, the chosen version is frozen. ρ and κ on `cal-test` are reported once.

## 5. Methods and fixed settings

### 5.1 Models (full identities)
| Role | Model | Identity |
|---|---|---|
| Generator and judge | Ollama `qwen3:4b-instruct-2507-q4_K_M` | digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` |
| Dense embeddings | Ollama `bge-m3` | digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` |
| Reranker | `BAAI/bge-reranker-v2-m3` (fp16) | HF revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

The code refuses to run on a digest mismatch. The thinking-only `qwen3:4b` is not used (D-028). No model changes are made (team, 2026-09-27).

### 5.2 Query sets
Each set has size N_{d,s} = min(|A_{d,s}|, number of U queries passing the M1 gate, number passing the M1-iso gate), applied to **every** method for that dataset and seed. Every time the minimum binds, it is reported.

| ID | Definition |
|---|---|
| B0 | A uniform random draw of N from U (20 sampling seeds) |
| B1 | Stratified random: R-derived strata and quotas, random within each stratum (20 sampling seeds) |
| M1-iso | The **same frozen prompt version with the neighbour block omitted**, 1 call per query, its own NNLS weights (§4.7), the same gate and the same stratified selection |
| M1 | Neighbour-grounded judge scores (§4), then the gate, then stratified selection |

- **Pool:** U = P-std ∪ P-persona.
  - Templates `gen_std_v1` and `gen_persona_v1` (spec Appendix B.1 and B.2) are fixed and never tuned.
  - Decoding: temperature 0.7, top_p 0.9, at most 64 tokens, per-call seed 1000 + i.
  - 800 source passages per dataset (seed 7).
  - Drop queries under 2 tokens, and queries with more than 60% overlap with the passage's first sentence.
  - Dedupe within the pool at BGE-M3 cosine 0.95.
- **Neighbours:** BGE-M3 dense top-30 from `R_seed`, reranked to a top-5 (reranker max_length 128). Neighbour order is permuted with the seed `hash(query_id, perm_idx)`. Queries with `nn_max_cos` > 0.97 are excluded from every set.
- **Judge decoding:** temperature 0, seed 0, `num_ctx` 4096, `think=False`. A failed call is retried once, with the validation error appended. If it still fails, the query cannot be selected, and the failure rate is reported.
- **Strata:** k-means on the BGE-M3 embeddings of R, with k = min(10, ⌊|R|/15⌋), seed 0. Crossed with language class for D4. Quotas n_k = round(N·π_k), with proportional redistribution when a stratum runs short.
- **Selection:** greedy by score within each stratum; a candidate is skipped if its cosine with an already selected query exceeds 0.90.

### 5.3 Retrieval configurations
| ID | Configuration |
|---|---|
| C1 | BM25, k1 = 0.9, b = 0.4 |
| C2 | BM25, k1 = 1.2, b = 0.75 |
| C3 | BGE-M3 dense (Ollama), normalised, exact cosine |
| C4 | RRF(C1, C3), k = 60 |
| C5 | C3 top-50, reranked |
| C6 | C1 top-50, reranked |
| C7 | C4 top-50, reranked |

- **BM25:** `bm25s`, lowercased, English stopword list, no stemming, the same for every dataset.
- **Passage text:** "title + text", sent to C3 untruncated up to the model limit; `num_ctx` is fixed in `configs/systems.yaml` during P2 on D1 and frozen (§6).
- **Reranking:** bge-reranker-v2-m3, max_length 512.

### 5.4 Qrels
- **D1 and D2:** official qrels.
- **Synthetic queries:** their source passage, grade 1.
- **D4:**
  - The source chunk of each scenario is grade 2.
  - Pooling takes the top-10 from C1, C3, C4 and C5 for every authentic D4 query.
  - LLM pre-labels are 0/1/2, from the core model with the scenario as context.
  - Humans verify every LLM-positive item and a random 10% of the LLM-negative items.
  - The final label is the human label where one exists, and 0 otherwise for LLM-negatives.
  - D4 qrels are finished **before** the freeze, and no metric is computed while they are built.
  - Pool bias toward C1, C3, C4 and C5 is disclosed.

## 6. Freeze (before any held-out evaluation)

Before `heldout_A` is unsealed, `results/FROZEN.json` records:
- the judge: template hashes, prompt version, anchor item IDs, model digest, decoding settings and permutation rule;
- the M1 and M1-iso weights, and the gate;
- hashes of the generation config and the pools, per dataset and seed;
- the selection, strata and dedupe settings;
- the retrieval configuration file (§5.3);
- the git SHA of the analysis code, and the bootstrap seed.

Both team members sign in `decisions.md`, and only then is `heldout_A` unsealed. Any later change to a frozen item produces a separately reported `-v2`. **The v1 result stays the primary result.**

## 7. Statistical analysis

1. **Unit of resampling:** queries. B = 1,000 bootstrap iterations, with the numpy seed 20260927.
2. **In each iteration,** for each dataset d ∈ {D2, D4} and each seed s = 0…4:
   - resample A_{d,s} with replacement, once, and use that resample for every method;
   - resample each method's set independently, with replacement, at its own size. B0 and B1 resample each of their 20 draws and average τ_b over them;
   - compute τ_b(X, A) over C1–C7 from the per-query nDCG@10 table.
3. **Statistic:** Δ̄_B = mean_d mean_s [τ_b(M1, A) − τ_b(B, A)], for B ∈ {B0, B1}.
4. **H2 decision:** the one-sided p-value p_B is the share of iterations with Δ̄_B ≤ 0. **H2 is supported if and only if p_B0 < 0.05 and p_B1 < 0.05.** This is an intersection-union test, so no multiplicity correction is needed.
5. **Also reported:**
   - Δ̄ with its 95% percentile CI;
   - the per-dataset Δ with CIs, as secondary breakdowns with no significance claim;
   - τ_AP, winner agreement, top-3 overlap, MAE and signed bias;
   - W1 of length and leakage, JS of topic, TV of language;
   - mean ± sd over seeds;
   - D1 rows, marked as development.
6. **Caveat.** With 7 configurations (21 pairs), τ_b is coarse, and the paper says so.
7. **Lock.** After the freeze, none of the following may change: the primary endpoint (§1), the datasets, the configurations, B, α, the resampling scheme or the decision rule. Any other analysis is labelled exploratory. Before the freeze, unit tests for τ_b, the bootstrap (on a known case) and nDCG against `ranx` must pass.

## 8. Reporting, fallbacks and deviations

- **Every pre-registered hypothesis (H1–H4) is reported whatever its outcome**, including a null or negative H2, on every seed and both held-out datasets. No seed, dataset or configuration is dropped after results are seen.
- Null and negative results are kept. If B1 matches M1, the finding is "topic matching explains most of the gap". If M1-iso matches M1, the neighbour-grounding claim is not supported, and the paper says so.
- **Too few survey responses.** If HiAQ has fewer than 300 queries after the §3.1 rules, H2 and H3 are evaluated on D2 alone and D4 is reported as a pilot. This is recorded as an amendment **before** `heldout_A` is unsealed.
- **Qrels protocol.** If the full D4 protocol (§5.4) becomes infeasible, the 30% pooled-qrels fallback is used. This is recorded as a deviation before the freeze.
- **Throughput.** Any reduction in seeds or pool size is decided and amended **before** the freeze.
- **Stated limitations:**
  - the same model generates and judges (possible self-preference);
  - synthetic qrels are sparse (source passage only);
  - HiAQ queries are elicited from scenarios rather than organic;
  - τ_b over 7 configurations is coarse.

---

## Amendments

_None yet._

### A1 · 2026-10-10 · HiAQ-Services has no eligible authentic queries (§8 "Too few survey responses")
1. **Condition and timing.**
   - On 2026-10-10 the team closed authentic HiAQ collection for this study with zero eligible authentic queries (D-059); no survey responses were collected.
   - Zero is fewer than 300, so the §8 contingency applies.
   - This amendment is recorded before `heldout_A` is unsealed, and together with A2, before calibration annotation starts.
2. **H2 and H3** are evaluated on D2 (CQADupStack-Android) alone. The §7 procedure is unchanged with d ∈ {D2}:
   - Δ̄_B = mean over seeds s = 0…4 of [τ_b(M1, A) − τ_b(B, A)];
   - B = 1,000 query-level bootstrap iterations, seed 20260927;
   - H2 is supported if and only if p_B0 < 0.05 and p_B1 < 0.05;
   - H3 uses M1-iso as the comparator, one-sided α = 0.05.

   D4 is reported as a pilot. With no authentic D4 queries, no D4 agreement statistic can be computed, so the pilot report is limited to the HiAQ corpus, scenarios, synthetic pools and their descriptive statistics. This invokes a pre-registered contingency and is not a methodological change.
3. **H1.** The wording is unchanged.
   - D4's three tests (length KS, gold-passage overlap KS, topic JS permutation) cannot be computed. They stay in the Holm family of 3 datasets × 3 measures (m = 9) as not rejected (p = 1), so the family size and the Holm thresholds are those of §2.
   - For the τ_b criterion, D4 counts as not meeting τ_b(B0, A) < 0.8. The criterion is met only if both D1 and D2 have seed-averaged τ_b(B0, A) < 0.8, each reported with its CI.
4. **H4** (D4 only) is not estimable and is reported as such.
5. **D4 qrels.** No D4 qrels are built, since there are no authentic D4 queries to score, and the §8 qrels fallback does not apply. D4 synthetic queries keep their source-passage qrels (§5.4) for pilot descriptives only.
6. **Calibration** is as amended by A2.
7. **Excluded from every confirmatory and secondary analysis (H1–H4), and from calibration and anchors:**
   - collaborative-sheet answers (D-056);
   - the external synthetic set `X-team150` (D-058);
   - the workbooks quarantined on 2026-10-10.

   They may appear only in analyses labelled exploratory (§2, §7.7).
8. **Everything else is unchanged.**

Recorded by: Chinmai Shetti (project lead)

### A2 · 2026-10-10 · Calibration without authentic HiAQ items (§4.1–§4.2); analysis released as v2
1. **Reason.** With no authentic HiAQ queries (A1), §4.1's 30 authentic HiAQ items cannot exist, and its 70 synthetic HiAQ items cannot receive neighbours from a D4 R. §4 cannot be executed as written.
2. **Items: 100.** The HiAQ half is removed, and no item is added or substituted. The remaining items are:
   - the 30 authentic FiQA `CAL-auth` items already drawn under §4.1 (seed 17; in R for every split seed);
   - the 70 synthetic FiQA items drawn under §4.1 (35 P-std + 35 P-persona, seed 17).
3. **Split.** 60 `cal-dev` / 40 `cal-test`, stratified by dataset × authentic/synthetic, seed 17: the same 60% `cal-dev` share as §4.2 (authentic 18/12, synthetic 42/28). Each FiQA item keeps the assignment §4.2 gives it. `cal-test` stays sealed.
4. **Everything else in §4 is unchanged:**
   - neighbours and the practice items;
   - annotation and blinding;
   - κ and the one-time revision with 40 re-annotated `cal-dev` items;
   - the human reference;
   - prompt versions v0–v4 from `cal-dev` only (the D4 Hinglish-anchor clause has nothing to apply to);
   - the selection criterion;
   - NNLS weights for M1 and M1-iso with 5-fold CV;
   - the gate F ≥ 4;
   - permutations;
   - §4.9.

   D2 contributes no calibration items. The data excluded in A1 item 7 are not used.
5. **Versioning (line 3; D-030).** This is a methodological change, so the analysis is released as **prereg-v2 = prereg-v1 + A1 + A2**.
   - prereg-v1's primary analysis cannot be executed, because its calibration set does not exist. It is reported as not executable, with this reason.
   - A2 was recorded on 2026-10-10, **before any calibration annotation or calibration result existed**.

Recorded by: Chinmai Shetti (project lead)

### A3 · 2026-10-14 · No human calibration: prompt v0 with equal weights (§4.3–§4.7); analysis released as v3
1. **Reason.** On 2026-10-14 the project lead decided that the study proceeds without the human calibration annotation of §4.3 (D-062). No calibration annotation exists or will be collected, so §4.4–§4.7 (κ, the human reference, version selection, NNLS weights) and the `cal-test` report cannot be executed.
2. **Judge.** Prompt **v0** (spec Appendix B.4 verbatim; `configs/judge.yaml`, unchanged) is the final and only prompt version. No v1–v4 and no anchor examples are created. The model, the decoding settings, the output schema, the permutation rule (§4.8) and the neighbours (§5.2) are unchanged.
3. **Weights.** M1 and M1-iso both score with **equal weights** on (H, S, P, C*), where C* = C, or the mean of H, S and P when C is null (the spec §9.3 equal-weights ablation). They are fixed a priori, not fitted, and `results/FROZEN.json` (§6) records them in place of the NNLS weights.
4. **Unchanged:** the gate F ≥ 4 and all of §5.2 (the N rule, B0, B1, the strata and quotas, greedy selection, the 0.90 dedupe, the near-duplicate exclusion). The hypotheses, the primary endpoint, the datasets as amended by A1, the configurations, B, α, the resampling scheme and the decision rule are unchanged.
5. **Not executed, and reported as such:** κ, the human reference, the ρ-based version selection, the NNLS weights and their 5-fold CV ρ, and every `cal-test` statistic. The A2 calibration pack exported on 2026-10-14 stays unannotated, and `cal-test` is never unsealed.
   - The judge's agreement with human ratings is therefore unknown.
   - The paper states that the judge was **not human-calibrated** and makes no claim about judge–human agreement.
6. **Freeze sign-off (§6).** The freeze is signed in `decisions.md` by the project lead before `heldout_A` is unsealed.
7. **Timing.** Recorded before any bulk judging output, query-set selection or held-out result exists. The only judge outputs so far are the P4.2 engineering pilot (12 FiQA queries, D-049), which drives no choice.
8. **Versioning (line 3; D-030).** This is a methodological change, so the analysis is released as **prereg-v3 = prereg-v1 + A1 + A2 + A3**. prereg-v2's primary analysis cannot be executed, because its calibration annotation does not exist. It is reported as not executable, with this reason.

Recorded by: Chinmai Shetti (project lead)
