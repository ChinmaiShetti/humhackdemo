# Phase 4 execution checklist (derived from the sources, 2026-09-28)

**Sources:** spec §8 and §9 (plus Appendices B.4 and C), prereg-v1 §3.3, §4 and §5.2 (frozen), PLAN v2 P4.1–P4.5, `configs/judge.yaml`, `configs/calibration.yaml`, D-002 and D-003. Not committed until the team approves.

## Frozen settings (the build must match these)
| Item | Value | Source |
|---|---|---|
| Judge model | `qwen3:4b-instruct-2507-q4_K_M`, digest `0edcdef3…` (checked); `think=False` | prereg §5.1; D-028 |
| Judge decoding | temperature 0, seed 0, `num_ctx` 4096, `format=` the JudgeOutput schema | prereg §5.2; spec §9.2 |
| Prompt | v0 = spec B.4 verbatim (`judge_v0_system`, `judge_v0_user`) | prereg §4.5 |
| Retry | once, with the validation error appended; still failing → the query cannot be selected; failure rate reported | prereg §5.2 |
| Permutations | 2, plus a 3rd if any dimension differs by ≥ 2; median per dimension; order seeded by `hash(query_id, perm_idx)`; disagreement rate logged | prereg §4.8 and §5.2 |
| M1-iso | the same frozen prompt version **without** the neighbour block; 1 call per query | prereg §5.2 |
| Neighbours | bge-m3 dense top-30 from **R_seed** → bge-reranker-v2-m3 (max_length **128**) → top-5; features `nn_max_cos`, `nn_mean5_cos`, `nn_lang_mix`; `nn_max_cos` > 0.97 means near-duplicate, excluded from every set; authentic items: no self, nothing with cos > 0.97 | spec §8; prereg §5.2 and §4.1 |
| Cache | SQLite `cache/judge.sqlite`, key = (rendered prompt, model digest, decoding params) | spec §9.2 |
| Aggregation | gate F ≥ 4 (a priori); s = w·(H, S, P, C), C imputed as the mean of H, S, P when null; NNLS weights on `cal-dev` (M1 and M1-iso fitted separately); 5-fold CV ρ; equal-weights ablation; `keep` logged only | spec §9.3; prereg §4.7 |
| Throughput | benchmark 50 calls in P4 | spec §9.2 |
| Isolation | R only; A is never loaded; `cal-test` sealed | prereg §3.3 |

## Stages
| Stage | WP | Can run now? |
|---|---|---|
| Neighbours for U, D1 and D2, split seeds 0–4 | P4.1 | **yes** |
| Neighbours for D4 | P4.1 | **no:** D4 R does not exist until the survey (Phase 1 gate) |
| Judge engine + M1-iso + aggregation (code and tests) | P4.2 | **yes** |
| Judge pilot + 50-call throughput benchmark (D1, seed 0) | P4.2 | **yes** (pilot only; bulk judging is P5.1) |
| Calibration set of 200 items + split + export | P4.3 | **blocked:** needs 30 HiAQ CAL-auth from the first survey tranche (H-07). The selection and export code can be built and tested now. |
| `docs/annotation_guidelines.md` | P4.3 | **v1** (D-050); awaits the team's confirmation before annotation |
| κ, prompt v0–v4 on `cal-dev`, NNLS weights | P4.4 | **blocked:** needs human annotation (H-08) |
| Freeze candidate + one `cal_test` unseal | P4.5 | **blocked:** after P4.4 |

## Known specification gaps (all resolved by the team in D-050)
1. **SOURCE PASSAGE for authentic calibration items.** The B.4 prompt shows "the passage the TEST QUERY was generated from". Authentic queries have no source passage, and neither the spec nor the prereg says which passage to show (for example a grade ≥ 1 qrel). **Resolved (D-050):** the highest-grade qrels passage with a sha256 tie-break.
2. **DOMAIN description for D4.** `judge_v0_user` needs `domain_description`. D1 and D2 have it in their dataset configs; D4 has none. **Resolved (D-050):** `configs/domains.yaml`.
3. **Label Studio** is not installed (it is a package, not a model). **Resolved (D-050):** a CSV spreadsheet export instead; nothing installed.
