# Phase 0 report: Setup (HUMAN GATE)

Status: **P0.1–P0.4 done; D-028 resolved. Waiting for the team to review the pre-registration.** It is not frozen and not tagged.

## Done
- **Repo** at `D:\humanalign-rag`, with temp files on D: (D-019). Python 3.11.15, torch 2.11.0 + cu128 (RTX 3050 6 GB), pinned packages, no ONNX Runtime.
- `ha doctor`: all OK. `pytest -q`: all pass. `ruff check`: clean.
- **Data (P1.1, approved, 40.8 MB):**
  - FiQA: 57,638 passages, 648 test queries, 1,706 qrels
  - CQADupStack-Android: 22,998 passages, 699 queries, 1,696 qrels

  Both match the BEIR paper exactly.

## Model stack (`env/models.json`, verified against pinned digests)
| Role | Model | Identity | Check |
|---|---|---|---|
| Generator + judge | `qwen3:4b-instruct-2507-q4_K_M` (downloaded after team approval, D-028; 2.33 GB on D:) | digest `0edcdef34593eac1…` | ok |
| Embeddings | `bge-m3` (Ollama, F16, dense) | digest `790764642607…` | ok; dim 1024, normalised |
| Reranker | `BAAI/bge-reranker-v2-m3` (existing weights, hard-link repair, D-025) | revision `953dc6f6…` | ok; relevant −0.13 vs irrelevant −11.03 |
| *Not used* | `qwen3:4b` (= `qwen3:4b-thinking-2507-q4_K_M`) | digest `359d7dd4…` | thinking-only; excluded from the core (D-028) |

## Measurements (engineering, not research results)
| Item | Value | Source |
|---|---|---|
| Generation | **0.54 s median** per call; 20.3 output tokens; VRAM 2.96 GiB | `phase_0_llm_verification.json` |
| Judge (B.4 v0, ~650 prompt tokens) | **1.60 s median** (mean 2.03, max 9.3) per call; JSON valid 20/20; VRAM 2.96 GiB | same |
| bge-m3 embeddings (Ollama, sharing the GPU with the LLM) | passages 6.5–16/s; queries 19/s | `phase_0_benchmark_qwen3-4b-thinking.json` |
| Reranker | query–passage 74 pairs/s; query–query 393 pairs/s; peak 1.42 GiB | same |

The earlier file `phase_0_benchmark_qwen3-4b-thinking.json` measured the thinking build; its LLM numbers are superseded. Its embedding and reranker numbers still apply.

**Revised compute:**
- generation ≈ 0.5 h for all pools;
- primary judging ≈ 22–28 h (about 50k calls);
- M1-iso ≈ 2 h.

## Verification of the new model (`tools/bench_llm.py`)
- **Generation:** 20/20 outputs are genuine queries; 0 reasoning markers; 0 hit the token cap; 18/20 end with "?".
  - P-std queries look typically synthetic: long, formal, "…mentioned in the passage?".
  - P-persona queries are shorter and more natural, e.g. "how do i find out how much house i can afford without using apps?". This is exactly the contrast the study measures.
- **Judge:** 20/20 valid JSON.
  - Fidelity separates cleanly: F is 4.5 on matched passages and 1.0 on mismatched controls.
  - Permutation stability: 0 of 8 pairs differ by 2 or more on any dimension.
  - C is null on all English items, as it should be.

**Weaknesses of the v0 prompt, recorded for P4 calibration.** They are not fixed now: prompt changes happen only on `cal-dev`, after human labels.
- H is too generous to exam-style P-std queries (H 4 for "What economic relationship does the passage suggest…"), and harsher on terse human-like queries (H 2 for "help with macro wage increase and unemployment link").
- The `reason` label does not match `keep` (for example `keep=true` with `too_document_like`).
- Halo effect: with a mismatched passage, H, S and P all collapse to 1.

These are the problems human calibration and the anchor examples (v1 onwards) exist to fix. If calibration cannot fix them, that is a reportable finding.

## For the team at the gate
1. **Review and approve `docs/preregistration.md`** (H2 is the pooled agreement test, D-027; the model is pinned, D-028). After approval, a commit sets FROZEN and tags `prereg-v1`.
2. Sparse-qrel risk: E11 stays high-priority optional.
3. Human tasks: ethics approval (H-04) before the survey; scenario writing (H-05) starts in P1.
