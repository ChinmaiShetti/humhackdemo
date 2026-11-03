# Phase 4 report (in progress): Neighbours, judge and calibration

**Status on 2026-09-28: P4.1 (D1, D2) and P4.2 are complete, with the pilot done. P4.3–P4.5 are blocked by the unresolved Phase 1 human gate and by team decisions.** P4.3 export machinery added under D-050 (committed with the Phase 1 checkpoint of 2026-09-29). Checklist: `reports/phase_4_checklist.md`. Decision: D-049. No P3 pool was modified; the prereg and frozen configs are unchanged; no seal was opened (no `results/unseal_log.jsonl`).

## P4.1 Neighbours: complete for D1 and D2 (D4 blocked: no R before the survey)
**Settings (prereg §5.2):**
- bge-m3 dense cosine against **R_s only**;
- top-30 → bge-reranker-v2-m3 at **max_length 128** → top-5;
- pair cache shared across seeds;
- features `nn_max_cos`, `nn_mean5_cos`, `nn_lang_mix`; near-duplicate if `nn_max_cos` > 0.97.

Output: `data/neighbours/<ds>/s<seed>/U.parquet` plus `.meta.json` (includes the top-30 candidate lists for the later neighbour ablations). Log: `logs/p4_neighbours.log`.

| Dataset | Seeds | Queries per seed | \|R_s\| | Near-duplicates | Median nn_max_cos | Median nn_mean5_cos | Rerank time per seed |
|---|---|---:|---:|---:|---:|---:|---:|
| D1 fiqa | 0–4 | 1,443 | 260 | 0 (every seed) | 0.566–0.570 | 0.508–0.510 | 103–108 s |
| D2 cqa_android | 0–4 | 1,295 | 279 | 0 (every seed) | 0.638–0.645 | 0.579–0.581 | 97–101 s |

A pilot (`U_pilot`, 20 D1 queries, seed 0) was inspected by hand: topical neighbours for on-topic queries, and weak neighbours for off-topic synthetic ones.

## P4.2 Judge: engine complete; pilot and 50-call benchmark done
- **Engine:** `src/ha/judge/{prompt,run,aggregate}.py`, with frozen v0 = B.4 verbatim.
  - Decoding: temperature 0, seed 0, `num_ctx` 4096, `think=False`, JSON-schema `format`.
  - Cache in `cache/judge.sqlite`.
  - Retry once with the validation error appended; transport errors fail without retry.
  - M1: permutations 0 and 1, plus 2 on any Δ ≥ 2; median per dimension; permutation seed `hash(query_id, perm_idx)`.
  - M1-iso: no neighbour block, 1 call.
  - Aggregation: gate F ≥ 4, C* imputation, NNLS plus 5-fold CV (fold seed 17), equal-weights ablation.
  - Weights stay `null` until calibration.
- **Pilot, run `P4-judge-pilot-20260928T230859-1f27d5`** (`reports/phase_4_judge_pilot.json`; ENGINEERING, not a result): D1 seed 0, 12 U queries (all happened to be P-std), 38 calls.
  - **38/38 valid JSON on first try**; 0 retries, 0 failures.
  - Third-permutation rate 2/12.
  - **Deterministic fresh-state uncached repeat 4/4.**
  - **1.50 s per call median** (mean 1.64); median prompt 596 tokens.
  - Gate pass: M1 8/12, M1-iso 9/12.
  - Observation, not a result: M1-iso gave H = 5 to nearly every exam-style P-std query; with neighbours, H and S dropped for document-echo queries.
- **Bulk projection (P5.1; not run):** about 3.5 s per M1 query → D1 about 7.0 h and D2 about 6.3 h over 5 seeds, plus about 0.6 h each for M1-iso. Under the spec's 40 GPU-hour threshold, so local execution is feasible.

## P4.3 Calibration set: code ready, execution BLOCKED
- `src/ha/judge/calibrate.py` has deterministic selection (30 + 30 CAL-auth, 70 + 70 synthetic split 35/35 by pool, seed 17) and a stratified 120/80 split (seed 17), with tests.
- It **refuses to build a partial set** without the HiAQ half.
- **Blocker:** the 30 HiAQ CAL-auth items come from the first survey tranche (H-07). The survey waits for the Phase 1 human gate.
- **Export machinery (D-050):** `src/ha/human/calibration.py`; CLI `ha calib status | export [--practice] | import | finalize`.
  - Blind CSV sheets per annotator, plus a separate key; adjudication and final labels are separate files.
  - Authentic passages come from qrels (highest grade, sha256 tie-break).
  - Human-facing neighbours come from R_s0 \ all 60 CAL-auth; the judge side is unchanged.
- **Status:**
  - FiQA CAL-auth 30/30 passages resolved; HiAQ 0/30 (no CAL-auth, no qrels yet).
  - The practice pack is written (`data/calibration/sheets/practice.csv`, 20 items).
  - The 200-item pack is **blocked**.
- `docs/annotation_guidelines.md` is **v1**. It waits for team confirmation before annotation starts.

## P4.4 and P4.5: BLOCKED
Human annotation (H-08) → κ, prompt v0–v4 on `cal-dev` only, NNLS weights, freeze candidate, and a one-time `cal_test` unseal.

## Team decisions (resolved in D-050)
1. Authentic items → a qrels passage (highest grade; sha256 tie-break). HiAQ pending until its qrels exist.
2. Annotators see 5 neighbours from R_s0 minus all 60 CAL-auth. Display only; judge-side retrieval unchanged.
3. The D4 domain description is in `configs/domains.yaml`.
4. Spreadsheet (CSV) export; Label Studio not installed.

## Tests
`pytest -q` 137 passed (new: `test_calibration_sheet.py` 11, `test_neighbours.py` 4, `test_judge.py` 10, `test_calibrate.py` 2). `ruff check .` clean. One CLI test example was switched from `neigh` (now implemented) to `select` (still a stub); the intent is unchanged.

## Storage
`data/neighbours` 5.0 MB; `data/judge` pilot 20 KB; `cache/judge.sqlite` 24 KB; plus rerank pair-cache growth.
