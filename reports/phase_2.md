# Phase 2 report: Retrieval configurations (AUTO GATE)

Status: **AUTO GATE passed.** E0 is within tolerance, `pytest -q` passes (96 tests) and ruff is clean. Every number below comes from `results/registry.jsonl`: run `E0-20260928T003743-14d0b0`, git `53896ec` (clean), 42 rows. Timings and resources are in `results/runs/<ds>/resources.jsonl`. **No sealed data was accessed** (there is no `unseal_log.jsonl`).

## E0 sanity check (D1 FiQA, all 648 authentic test queries; development dataset)
| Config | nDCG@10 | MRR@10 | Hit@1 | Hit@5 | R@100 | Reference | Check |
|---|---|---|---|---|---|---|---|
| C1 BM25 0.9/0.4 | 0.2380 | 0.2959 | 0.2238 | 0.3951 | 0.4946 | 0.236 ± 0.02 (BEIR) | ✅ gate |
| C2 BM25 1.2/0.75 | 0.2384 | 0.2924 | 0.2176 | 0.3889 | 0.5113 | — | |
| C3 BGE-M3 dense | 0.4128 | 0.5032 | 0.4182 | 0.6142 | 0.7125 | 0.413 ± 0.03 (DRAMA Table 7) | ✅ (warning-level) |
| C4 RRF(C1, C3) | 0.3578 | 0.4344 | 0.3426 | 0.5509 | 0.6932 | — | |
| C5 C3 → rerank | 0.4354 | 0.5181 | 0.4352 | 0.6327 | 0.7125 | — | |
| C6 C1 → rerank | 0.3582 | 0.4548 | 0.3827 | 0.5679 | 0.4946 | — | |
| C7 C4 → rerank | 0.4328 | 0.5192 | 0.4352 | 0.6404 | 0.6932 | — | |

C3 matches the published BGE-M3 dense number almost exactly (0.4128 against 0.413). This confirms that Ollama's F16 GGUF build behaves like the reference model.

### Observation for later analysis (development data; nothing is changed)
Three of the 21 configuration pairs are **near-ties** on authentic FiQA queries:
- C1 vs C2: Δ = 0.0004
- C4 vs C6: Δ = 0.0004
- C5 vs C7: Δ = 0.0027

Their order in any query set is close to a coin flip, so τ_b(X, A) cannot reach 1 even for a perfect proxy. This is the coarseness the pre-registration already flags (§7.6). The configurations are frozen by prereg §5.3, so **nothing changes**.

**Suggestion for the team, exploratory only:** report an "authentic noise ceiling", τ_b between two random halves of A. It would give readers a scale for the observed τ_b values. It would be labelled exploratory under §7.7 and would not affect H2.

## Runtime and resources (RTX 3050 Laptop 6 GB)
| Stage | Dataset | Size | Wall time | Rate | Peak VRAM |
|---|---|---|---|---|---|
| BM25 index (C1 / C2) | FiQA | 57,638 passages | 10.5 s / 10.4 s | — | CPU |
| BM25 index (C1 / C2) | CQA-Android | 22,998 | 3.3 s / 3.1 s | — | CPU |
| bge-m3 corpus encoding (`num_ctx` 8192) | FiQA | 57,638 | **28.6 min** | 33.6 passages/s | 0.62 GiB (Ollama) |
| bge-m3 corpus encoding | CQA-Android | 22,998 | **10.1 min** | 37.9 passages/s | 0.62 GiB |
| Runs C1 / C2 / C3 / C4 | FiQA | 648 queries | 0.8 / 0.9 / 13.6 / 0.2 s | — | — |
| Runs C5 / C6 / C7 (reranking; pair cache shared) | FiQA | 648 × 50 pairs | 7.2 / 9.4 / 8.3 min | — | 1.42 GiB (torch) |
| Smoke run C1–C7 | CQA-Android | 10 made-up queries | < 45 s in total | — | 1.42 GiB |

**Latency per query** (median of 3 passes over a fixed 200-query FiQA sample, caches off; the Ollama and torch stages are timed separately and summed):

| C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|
| 0.97 ms | 0.96 ms | 16.2 ms | 17.5 ms | 650 ms | 876 ms | 784 ms |

- Reranking over 50 passages at max_length 512 dominates (about 0.65–0.88 s per query).
- CQA-Android latency waits for the freeze, because its authentic queries are sealed (D-039).

**Compute projection for later phases.** Each dataset has about 1,600 pool queries. Running C5–C7 over them costs roughly 3 × (1,600 / 648) × 8 min ≈ **1 h per dataset** on the 3050, less with the pair cache.

## Held-out handling (D-039)
For CQADupStack-Android only corpus-side work ran before the freeze: encoding, BM25 indexes, and a smoke run on 10 made-up queries that were never scored. `query_set(..., "auth")` for a held-out dataset raises `SealedDataError`, and a test enforces this.

## Not started (by instruction)
Phase 3 (generation) and Phase 4 (calibration) wait for the Phase 1 data gates.
