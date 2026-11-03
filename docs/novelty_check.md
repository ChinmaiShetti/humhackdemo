# Novelty check

Spec §3.3. The check runs twice: #1 in Phase 0 and #2 before writing the paper (P8.4). Only hits from 2025 onwards are recorded. Contributions C1–C4 are defined in spec §3.2.

## Check #1 · 2026-09-27 · web search (arXiv, ACM DL, ACL Anthology via the search engine)

**Queries run:** the 8 queries from §3.3 plus 2 targeted ones: "filter synthetic queries using real query log similarity nearest neighbours LLM judge", and "Romanised Hindi Hinglish help-desk queries dataset retrieval RAG public services India".

**Verdict:** no hit overlaps C1–C4. Nobody curates synthetic query sets with a judge grounded in authentic neighbours and validates the result by system-ranking agreement with held-out authentic queries. There is also no public scenario-elicited EN / Romanised-Hindi / Hinglish public-service query set. **No re-scoping is needed.** One design risk surfaced (see the Rahmani row below).

| Work (date) | What it does | Overlap / difference with ours |
|---|---|---|
| Kucia & Gawlik, *Beyond Benchmark Scores*, arXiv 2609.14579 (Sep 2026; CIKM '26) [spec ref 1] | Diagnoses the synthetic–authentic divergence (6.8 vs 15.7 words; 53 vs 165 sources); the top synthetic config is not the top authentic one. Proposes treating the two sets as complementary extremes. | Diagnosis only. It does not curate synthetic sets. C1 and C2 remain open. It motivates our P dimension and the topic strata. |
| Hussain & Nielbo, *The Coverage Illusion*, arXiv 2605.27220 (2026) [ref 2] | Production evidence that synthetic queries mislead routing decisions. | Diagnosis only. |
| Vardasbi et al., *As It Was*, SIGIR '26, arXiv 2607.01040 [ref 11] | A behaviour-grounded LLM **relevance** judge: "QRI cards" summarise historical interactions with similar queries. +5% Spearman; +15% on multilingual human judgments. | The closest analogue to neighbour grounding, but it judges **relevance of results**, not **realism of queries**, and it does not select test sets. Cite it as the inspiration for grounding. |
| MAPLE / MAPLE-Synth, Wei et al., arXiv 2608.15624 (2026) [ref 9] | Retrieval-based in-context generation of scientific-search queries. Realism is checked by blinded pairwise human tests, plus an LLM judge rating human-likeness on a 1–5 scale (reported ρ ≈ 0.73 with humans). | Grounding happens at **generation** time, and the LLM realism score is an **evaluation** metric, not a curation criterion with ranking validation. Scientific search only. Useful as a reference point for our judge-versus-human ρ. |
| Rahmani et al., *Synthetic Test Collections*, SIGIR '24, arXiv 2405.07767 [ref 5] | Synthetic vs real TREC DL queries: τ = 0.815; with synthetic judgments τ = 0.857. **With sparse judgments (only the source passage relevant), τ drops to 0.157.** | No overlap. **Design risk for us:** every synthetic set uses sparse `source_pid` qrels (spec §7), which may cap the achievable τ_b for all methods and compress the differences between them. Recommendation for the P0 gate: raise E11 (pooled qrels for M1 queries) from optional to planned, and report the sparse-qrel ceiling explicitly. |
| Rahmani et al., *Bias in Synthetic Data*, CIKM '25, arXiv 2506.10301 [ref 3] | LLM queries are longer; LLM judges inflate relevance. | Diagnosis. |
| PQR, Lu et al., arXiv 2605.16551 (May 2026) | Iteratively refines prompts under "realism policies" to generate realistic queries that make e-commerce QA agents fail. | Adversarial **generation** for stress-testing. No authentic-neighbour judge, no curation for ranking fidelity. Different goal. |
| SeedRG, Liu et al., *Generating Leakage-Free Benchmarks for Robust RAG Evaluation*, arXiv 2605.08838 (May 2026) | "Leakage" means answerable from the LLM's parametric memory; entity-replacement generation. | A different meaning of "leakage" from our S dimension (passage-specific wording). No overlap. Avoid confusion in the paper: call S "specificity leakage (document-to-query)". |
| MirrorBench, Hathidara et al., arXiv 2601.08118 (KDD '26 D&B) | Benchmarks conversational user-proxy agents for human-likeness, with LLM judges calibrated by human–human and proxy–proxy controls. | Conversational utterances, not search queries, and no selection or retrieval validation. The calibration-control idea is worth borrowing for the judge analysis. |
| SynRewrite, arXiv 2509.22325 (2025) | Synthetic query rewrites aligned to user intent, trained with RL, to improve RAG performance. | Rewriting for **retrieval effectiveness**, not realism or evaluation validity. Relevant background for M2. |
| BenchBrowser, arXiv 2603.18019 (2026) | Retrieves benchmark items via embedding kNN, then filters them with an LLM judge, to assess benchmark validity. | kNN plus a judge used for **auditing benchmark content**, not for grounding a query-realism judge in authentic queries. Weak overlap in mechanism only. |
| Code-mixed search benchmark (GitHub, tanmay0401) | 348 hand-authored Hinglish product-search queries on Amazon ESCI; BM25 with reranking beats multilingual dense retrieval. | Hand-authored, not scenario-elicited; e-commerce; not about synthetic evaluation. Useful as related evidence for C3. |
| Springer 2026 chapter on hallucination-aware RAG for Indian code-mixed queries; SchemeGPT (GitHub) | Code-mixed RAG systems for Indian domains. | Systems, not evaluation data. No overlap with C3. |

**Carry forward to check #2:**
- Re-search Kucia & Gawlik follow-ups (any "fix the gap" paper).
- Search MAPLE follow-ups.
- Run "query realism" searches on the ACL Anthology directly (the search engine underweights it).
- Look for any HiAQ-like Indian public-service query set.

**Sources:**
- https://arxiv.org/abs/2609.14579
- https://arxiv.org/pdf/2605.27220
- https://arxiv.org/abs/2607.01040
- https://arxiv.org/pdf/2608.15624
- https://arxiv.org/abs/2405.07767
- https://arxiv.org/pdf/2506.10301
- https://arxiv.org/abs/2605.16551
- https://arxiv.org/abs/2605.08838
- https://arxiv.org/abs/2601.08118
- https://arxiv.org/html/2509.22325v1
- https://arxiv.org/pdf/2603.18019
- https://github.com/tanmay0401/code-mixed-search-benchmark
- https://link.springer.com/chapter/10.1007/978-3-032-27754-1_37
- https://github.com/aditya0si/schemeGPT

## Check #2 · 2026-10-17 · web search, before writing the paper (P8.4)
**Queries:**
- synthetic query selection by lexical overlap and ranking agreement;
- follow-ups to Kucia & Gawlik;
- LLM-judge realism filtering grounded in real-query neighbours, validated by ranking fidelity.

**Verdict:** no work curates synthetic query sets for system-ranking fidelity, and none selects them by matching authentic passage overlap. The contributions stand.

| Work | Relevance |
|---|---|
| van Elburg, van der Putten & Marx, *Can we Evaluate RAGs with Synthetic Data?*, SynDAiTE @ ECML-PKDD 2025, arXiv 2508.11758 | Synthetic QA benchmarks rank RAG systems with different retrievers reliably, but not systems with different generators. Background for ranking agreement; cited. |
| Abdalla, Abdallah, Sedek & Jatowt, *OBLIQ-IR*, EMNLP 2026, arXiv 2609.29649 | Appendix J, Table J.1: query-word coverage of synthetic vs official queries in their source documents is .036/.004, .097/.081, .105/.106 and .359/.102. Independent evidence of synthetic vocabulary leakage; cited. |
| *Mine and Refine*, arXiv 2602.17654 (2026) | E-commerce training data: synthetic queries with high lexical similarity bias a model toward lexical matching. Training, not evaluation; not cited. |

**Sources:**
- https://arxiv.org/abs/2508.11758
- https://arxiv.org/html/2609.29649
- https://arxiv.org/html/2602.17654
