# Phase 3 execution checklist (derived from the sources, 2026-09-28)

**Sources:** spec §7 (with its scope banner: PLAN v2 and decisions.md override §7 on pool size and generators), prereg-v1 §5.2 (frozen), PLAN v2 §1 and §4 (P3), `configs/generation.yaml`, D-026.

## Frozen settings (the build must match these)
- **Generator:** `qwen3:4b-instruct-2507-q4_K_M`, digest `0edcdef3…` (checked by `require_digest`), `think=False`.
- **Decoding:** temperature 0.7, top_p 0.9, `num_predict` 64, seed per call = 1000 + i (i = the passage's position in the sample).
- **Source passages:** 800 per dataset, uniform at random, `source_sample_seed` 7. `source_pid` is the single gold passage (grade 1).
- **Templates:** `gen_std_v1` (B.1), `gen_persona_v1` (B.2; 4 personas; plus the Hinglish persona for D3/D4 only), `gen_fewshot_v1` (B.3, verbatim; 5 exemplars from **R** only).
- **Post-processing** (spec §7, prereg §5.2):
  - strip quotes and numbering;
  - drop queries under 2 tokens;
  - drop queries with more than 60% token overlap with the passage's first sentence;
  - dedupe within each pool at BGE-M3 cosine > 0.95.
- **Cache key:** (prompt, model digest, params). The model digest is recorded.
- **AUTO GATE, per pool:** final unique ≥ 90% of target (≥ 720 of 800), **and** generation failure rate < 5%.

## Pools
| Dataset | Pool | Decision | Why |
|---|---|---|---|
| D1 fiqa, D2 cqa_android | P-std, P-persona | **run** | core, U (prereg §5.2) |
| D1 fiqa, D2 cqa_android | P-fewshot × split seeds 0–4 | **run** (team-authorised, optional-robustness tier) | R_s exists for D1 and D2; exemplars from R_s only |
| D4 hiaq | P-std, P-persona | **run** | the corpus is final (1,572 chunks, D-038) |
| D4 hiaq | P-fewshot | **defer** | R does not exist until the survey has run |
| D1–D3 | P-std-G2 | **stop** | generator G2 (gemma3:4b) is not available locally; no substitution |
| D1, D2 | P-docT5 | **stop** | a dataset download that needs approval; FULL-SCALE tier in PLAN v2 |
| D3 | all | **stop** | MIRACL-Hindi was never downloaded (optional, D-026) |

## Leakage guards
- P-std and P-persona see **only** corpus passages.
- P-fewshot sees only R_s texts, via `load_split(ds, s, "R")`, and asserts that every exemplar is in R_s. **A is never loaded.**
- No judge criteria, neighbours, calibration data or human labels appear in any prompt.

## Order
1. Engine and unit tests.
2. Smoke test: schema, cache hit, deterministic repeat, failure handling.
3. Pilot of 20 per pool; inspect.
4. Full run with checkpoints.
5. Dedupe and statistics.
6. Gate per pool.
