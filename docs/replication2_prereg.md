# Replication R2 pre-registration: overlap-matched selection on three more fresh datasets

**Status: written 2026-10-17, before any data of D7–D9 was downloaded or seen (D-072).** The git commit that adds this file is its timestamp. R2 reuses `docs/replication_prereg.md` (R1) unchanged, including the D-070 clarification on splitting, except for the datasets listed below.

## 1. Why
R1 was supported when pooled, but its gain came from Webmasters; NFCorpus showed none. R2 asks whether OVL's advantage holds on three further datasets, one of them from a different dataset family.

## 2. Hypothesis
**R2 (primary, confirmatory):** pooled over D7, D8 and D9, τ_b(OVL, A) > τ_b(B0, A) **and** τ_b(OVL, A) > τ_b(B1, A), over C1–C7.

**A priori expectation, stated for interpretation only (not a test):** if relevance-judgement density explains NFCorpus, OVL should help on SciFact (about 1 relevant passage per query).

## 3. Data
| ID | Dataset | Source (Hugging Face, revision listed 2026-10-17) | Expected counts |
|---|---|---|---|
| D7 | SciFact | `BeIR/scifact` @ `b3b5335604…`; qrels `BeIR/scifact-qrels` @ `2938d17dc3…` `test.tsv` (339 rows) | 5,183 passages; 300 test queries |
| D8 | CQADupStack-Mathematica | `mteb/cqadupstack-mathematica` @ `fc1af0b6c8…` | 16,705 passages; 804 test queries |
| D9 | CQADupStack-Programmers | `mteb/cqadupstack-programmers` @ `339629ee11…` | 32,176 passages; 876 test queries |

- D8 and D9 have more than 700 test queries, so the existing builder rule applies unchanged: 700 queries are sampled with seed 13 before splitting.
- A count mismatch stops the build and pauses the study for a dated decision.

## 4–7. Pipeline, analysis, sequence, reporting
Identical to `docs/replication_prereg.md` §4–§7, with:
- the dataset list (D7, D8, D9), in that order for the bootstrap;
- results in `results/eval/R2-replication.json`, registry exp `R2`, freeze record `results/FROZEN_replication2.json`.

**R2 is supported if and only if** the pooled p_B0 < 0.05 **and** p_B1 < 0.05. The outcome is reported whatever it is.
