# Replication pre-registration: overlap-matched selection on fresh datasets (R1)

**Status: written 2026-10-17, before any data of D5 or D6 was downloaded or seen (D-069).** The git commit that adds this file is its timestamp. Any later change is a dated deviation in `docs/decisions.md`, made before the evaluation step (§7 below).

## 1. Why
The exploratory overlap-matched selector OVL (D-068; PAPER_LOG §6.5) beat random selection on D1, D2 and D4. Its idea came from a post-freeze diagnosis that looked at those datasets' test sets, so that result is hypothesis-generating. R1 tests it on datasets the project has never touched, with every rule fixed here first.

## 2. Hypothesis
**R1 (primary, confirmatory):** pooled over D5 and D6, the system ranking produced by OVL agrees more closely with the ranking produced by the authentic test queries A than the rankings produced by B0 do, **and** than those produced by B1 do. Agreement is τ_b(X, A) over C1–C7.

## 3. Data
| ID | Dataset | Source (Hugging Face, revision listed 2026-10-17) | Expected counts (BEIR) |
|---|---|---|---|
| D5 | CQADupStack-Webmasters | `mteb/cqadupstack-webmasters` @ `c691d4af16…`: `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` | 17,405 passages; 506 test queries |
| D6 | NFCorpus | `BeIR/nfcorpus` @ `b5026a0e96…`; qrels `BeIR/nfcorpus-qrels` @ `a451b3b26d…` `test.tsv` | 3,633 passages; 323 test queries |

- They are built by the existing builder (`ha data build`). If a count differs from the expected value, the build stops and the study pauses for a dated decision.
- **Relevance:** official test qrels, grade > 0, are used as given; synthetic queries use their source passage, grade 1 (prereg §5.4).

## 4. Pipeline: the frozen settings, unchanged
1. **Splits.** `build_splits(ds, n_cal_auth=0)`: R 40% / A 60%, split seeds 0–4, stratified by language class × topic cluster.
2. **Synthetic pool.** U = P-std ∪ P-persona with `configs/generation.yaml` unchanged: 800 source passages (seed 7), temperature 0.7, top-p 0.9, the P3 filters and 0.95 within-pool dedupe.
3. **Neighbours.** The P4.1 procedure, used here only for the near-duplicate flag (`nn_max_cos` > 0.97). Near-duplicates are excluded from every set.
4. **Query sets,** per split seed s:
   - eligible = U minus the seed's near-duplicates; N = min(|A_s|, |eligible|);
   - **B0:** 20 uniform draws of N, SeedSequence `[0, s, d]`;
   - **B1:** 20 draws, random within the k-means strata of R_s with largest-remainder quotas, `[1, s, d]` (D-063);
   - **OVL:** exactly as D-068, `[3, s, d]`: 5 equal-frequency gold-overlap bands from R_s (never A), with R_s's band shares filled by random draws.
   - No LLM judging; M1, M1-iso and NN are not part of R1.
5. **Retrieval and metric.** C1–C7 exactly as `configs/systems.yaml`. Per-query nDCG@10 (ranx). Each set's score for a system is its mean over the set's queries (draws averaged at the τ level).

## 5. Analysis (identical in form to prereg §7)
- B = 1,000 bootstrap iterations, numpy seed 20260927.
- Each iteration, for each dataset d ∈ (D5, D6) in that order and each seed s = 0…4:
  - resample A_{d,s} once with replacement (shared by every set);
  - resample every set independently at its own size, in the fixed order OVL, B0, B1 (draws ascending);
  - compute τ_b(X, A); average τ_b over each method's 20 draws.
- **Statistic:** Δ̄_X = mean_d mean_s [τ_b(OVL, A) − τ_b(X, A)], for X ∈ {B0, B1}.
- **Decision:** p_X = the share of iterations with Δ̄_X ≤ 0. **R1 is supported if and only if p_B0 < 0.05 and p_B1 < 0.05** (intersection-union test).
- **Also reported, with no significance claim:**
  - per-dataset Δ with 95% percentile CIs;
  - seed-averaged τ_b per method with CIs;
  - per-seed values;
  - the overlap manipulation check (mean overlap of OVL vs R_s vs B0).

## 6. Sequence and sealing
1. Commit this file (and D-069).
2. Build the data; then splits, pools, neighbours, C1–C7 on U, and the sets. **A's queries and qrels are not used** before step 3. N uses the A labels only.
3. **Replication freeze:** commit `results/FROZEN_replication.json` with the hashes of the pools, neighbours, sets and U runs, and the code commit.
4. Only then: unseal A (`HA_UNSEAL=heldout_A`, logged), run C1–C7 on the authentic queries, and evaluate **once**.

## 7. Reporting
The outcome is reported whatever it is, with both datasets and all 5 seeds. No dataset, seed or configuration is dropped after results are seen. Results go to the registry (exp `R1`) and to PAPER_LOG.
