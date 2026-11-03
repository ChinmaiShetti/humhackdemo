# Phase 3 report: Synthetic pools (AUTO GATE)

**P3 generation is proceeding in parallel with the remaining P1 human gate.** Phase 1 is **not** passed (scenarios, LID labels and ethics metadata are all pending).

- Started 2026-09-28; full run finished the same day. Log: `logs/p3_full_run.log`. Stats: `reports/phase_3_full_stats.json`; pilot: `reports/phase_3_pilot_stats.json`; smoke test: `reports/phase_3_smoke.json`.
- Checklist and scope: `reports/phase_3_checklist.md`. Decisions: D-046 and D-047.
- **Not committed**, by instruction; generation records carry `git_sha` `3e45616…-dirty`.

## Settings (frozen; `configs/generation.yaml` unchanged)
- **Generator:** `qwen3:4b-instruct-2507-q4_K_M`, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` (checked), `think=False`.
- **Decoding:** temperature 0.7, top_p 0.9, `num_predict` 64, seed = 1000 + i.
- **Source passages:** 800 per dataset, seed 7.
- **Templates:** `gen_std_v1`, `gen_persona_v1` (4 personas, plus Hinglish for D4), `gen_fewshot_v1` (spec B.3 verbatim; 5 exemplars from R_s).
- **Post-processing:** strip quotes and numbering; drop < 2 tokens; drop > 60% first-sentence overlap; exact dedupe, then within-pool BGE-M3 cosine > 0.95.
- **Output:** `D:\humanalign-rag\data\synthetic\<ds>\<pool>\{generations.jsonl, queries.parquet, stats.json}`.

## Results per pool (target 800 each)
| Dataset | Pool | Target | Success (ok) | Failed | Failure % | Dropped as copies | Exact + near duplicates | Final unique | % of target | ≥ 90%? | Failures < 5%? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| D1 fiqa | P-std | 800 | 727 | 0 | 0.00 | 73 | 0 + 0 | **727** | 90.9 | ✅ | ✅ |
| D1 fiqa | P-persona | 800 | 723 | 0 | 0.00 | 77 | 3 + 4 | **716** | 89.5 | ❌ | ✅ |
| D2 cqa_android | P-std | 800 | 666 | 1 | 0.13 | 133 | 0 + 1 | **665** | 83.1 | ❌ | ✅ |
| D2 cqa_android | P-persona | 800 | 630 | 1 | 0.13 | 169 | 0 + 0 | **630** | 78.8 | ❌ | ✅ |
| D4 hiaq | P-std | 800 | 601 | 0 | 0.00 | 199 | 5 + 10 | **586** | 73.3 | ❌ | ✅ |
| D4 hiaq | P-persona | 800 | 670 | 0 | 0.00 | 130 | 13 + 12 | **645** | 80.6 | ❌ | ✅ |
| D1 fiqa | P-fewshot s0 | 800 | 780 | 0 | 0.00 | 20 | 0 + 0 | **780** | 97.5 | ✅ | ✅ |
| D1 fiqa | P-fewshot s1 | 800 | 772 | 0 | 0.00 | 28 | 0 + 0 | **772** | 96.5 | ✅ | ✅ |
| D1 fiqa | P-fewshot s2 | 800 | 779 | 0 | 0.00 | 21 | 0 + 0 | **779** | 97.4 | ✅ | ✅ |
| D1 fiqa | P-fewshot s3 | 800 | 773 | 0 | 0.00 | 27 | 0 + 0 | **773** | 96.6 | ✅ | ✅ |
| D1 fiqa | P-fewshot s4 | 800 | 783 | 0 | 0.00 | 17 | 0 + 0 | **783** | 97.9 | ✅ | ✅ |
| D2 cqa_android | P-fewshot s0 | 800 | 688 | 1 | 0.13 | 111 | 0 + 0 | **688** | 86.0 | ❌ | ✅ |
| D2 cqa_android | P-fewshot s1 | 800 | 668 | 1 | 0.13 | 131 | 0 + 0 | **668** | 83.5 | ❌ | ✅ |
| D2 cqa_android | P-fewshot s2 | 800 | 658 | 1 | 0.13 | 141 | 0 + 0 | **658** | 82.3 | ❌ | ✅ |
| D2 cqa_android | P-fewshot s3 | 800 | 666 | 1 | 0.13 | 133 | 0 + 0 | **666** | 83.3 | ❌ | ✅ |
| D2 cqa_android | P-fewshot s4 | 800 | 676 | 1 | 0.13 | 123 | 0 + 0 | **676** | 84.5 | ❌ | ✅ |

"Success (ok)" = 800 − failed − dropped (before dedupe). Attempted = 800 in every pool. Possible-truncation flags: 0.

## P3 AUTO GATE: **FAIL (partial)**. **FAIL** on size ≥ 90% (6/16); **PASS** on failures < 5% (16/16; maximum 0.13%)
- **Failure rate < 5%:** passed by all 16 pools (maximum 0.13%).
- **Final unique ≥ 90% of target:** passed by **6 of 16** pools (D1 P-std and the 5 D1 P-fewshot pools). **Failed by 10:** D1 P-persona (4 short of 720), all 7 D2 pools, and both D4 pools.
  *Correction:* the totals were first mis-added as 7 / 9; the per-pool rows were always correct (D-047).
- Evaluated per pool; nothing is averaged.

## Diagnosis
1. **The copy filter works as specified.** A pilot inspection matched every dropped query to its passage's first sentence (e.g. "Does UbuntuOne have a sync app for contacts?" ← "It doesn't look like UbuntuOne has a sync app for contacts."). The corpus structure drives the drop rate:
   - **CQADupStack passages are user questions:** 14–21% dropped.
   - **HiAQ chunks open with their own FAQ question:** 16–25% dropped.
   - FiQA answers: 9–10% dropped (standard and persona), 2–3.5% for few-shot.
   - It was **not** changed.
2. **Few-shot prompting with real R queries reduces copying:** FiQA 17–28 dropped against 73–77 for standard and persona. Descriptive observation only.
3. **The single failure in every D2 pool is the same passage:** `cqa_android` pid 50091 (i = 323). Its prompt is **7,885 tokens**, over Ollama's default 4,096 context, so Ollama **rejects** the request (`exceed_context_size_error`) rather than truncating it. It is recorded as a failure; `num_ctx` was not changed.

## Downstream meaning (no action taken)
- The 90% gate is a PLAN-level engineering gate. The pre-registration fixes 800 source passages and the filter rules; it does **not** set a minimum pool size.
- **Pool sizes against the selection budget:** U (P-std ∪ P-persona) is 1,443 for D1, **1,295 for D2** and **1,231 for D4**. D2's |A| is 420 per seed; D4's |A| is unknown until the survey.

## Team decision (D-048, 2026-09-28): option (a), retain the pools as generated
- The pools are retained for downstream selection. The retained candidate pools are still sufficient for the planned selection procedure: U = 1,443 (D1), 1,295 (D2), 1,231 (D4), against N = |A| per seed (D2: 420).
- **The filter was not changed. No top-up was performed. No context-size change was performed. No pre-registration amendment was made.** Pools, sampling, model and generation parameters are untouched.
- **This is a P3 gate outcome, not a termination of the study.** Phase 4 has not started.

## Options that were considered
- **(a) Accept the pools as generated.** Record the P3 gate as FAIL on size with this diagnosis, and treat the corpus-driven copy rate as a finding. No prereg change (the prereg sets no size threshold). **Recommended.**
- **(b) Top up with extra source passages** until each pool reaches 720. This changes the prereg-fixed "800 source passages, seed 7": it needs a prereg amendment and a v2 label.
- **(c) Change the filter or the context size.** Not recommended; the filter is prereg-fixed.

## Deferred or stopped pools
| Pool | Status | Reason |
|---|---|---|
| D4 P-fewshot | deferred | needs D4 R (after the survey) |
| P-std-G2 (D1–D3) | stopped | generator G2 (gemma3:4b) is not available locally; no substitution |
| P-docT5 (D1, D2) | stopped | a dataset download that needs approval; FULL-SCALE tier |
| All D3 pools | stopped | MIRACL-Hindi not downloaded (D-026) |

## Engineering checks
- **Smoke test:** valid output; cache hit on repeat; failure path raises and is recorded.
- **Determinism:** fixed by starting every pool from a fresh model state in fixed order (6/6 reproduced).
- **Tests:** `pytest -q` 110 passed (7 new in `tests/test_generate.py`); `ruff check .` clean.
- **Leakage:** P-std and P-persona see only passages; P-fewshot loads only R_s (a test enforces it). No A, qrels, judge criteria, calibration data or human labels are used.
- **Storage added:** about 17 MB on D: (pools 12 MB, pilot 0.6 MB, generation cache 4.1 MB), plus query-embedding cache growth for dedupe (D: total change about 264 MB).
