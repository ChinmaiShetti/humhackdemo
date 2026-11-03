# Decisions log

Every design choice the spec does not fix (spec §0). Newest at the bottom. Format: date · choice · reason · alternatives · who decided.
Plan issue numbers (I-n) refer to [PLAN.md](PLAN.md) §2.

---

### D-001 · 2026-09-27 · H2 wording (I-1) · team
**Choice.** "M1 achieves higher τ_b than B0 and B1 on ≥2 of the 3 held-out datasets (D2–D4), and is at least as good as the best non-LLM baseline on average." D1 is development only and is not counted as held-out. Holm correction runs across the 3 held-out datasets.
**Reason.** The spec's "≥3 of 4 held-out datasets" counted D1, which is the development dataset.
**Alternatives.** Requiring all 3 of 3 (rejected as too brittle with 10 held-out systems); counting D1 (rejected because it would leak design decisions into the test).

### D-002 · 2026-09-27 · Seed tiers (I-2) · team
**Choice.**
- **Primary tier, all 5 split seeds:** RQ1 and RQ2, i.e. E1, E2 (all methods in `tab_main`, including M3 and O), E10, and E5, which re-slices E2.
- **Secondary tier, seeds 0–2:** E3, E4, E6 and E7.
- **Robustness and sensitivity, seed 0:** E9, the LID ±0.05 sensitivity check, the FiQA-only weights row (D-013) and E11.

Reduced-compute analyses are labelled as such in every table and figure.
**Reason.** Everything that depends on R is per-seed. The full design needs about 365k judge calls (plan §5).
**Alternatives.** 5 seeds everywhere (infeasible); a single seed everywhere (too weak for the primary claim).

### D-003 · 2026-09-27 · Calibration items stay in R; no self-neighbours (I-3) · team
**Choice.**
- `CAL-auth` (30 FiQA + 30 HiAQ authentic queries) is selected once, deterministically, with a fixed seed and stratified by language class.
- It is forced into R for every split seed and never enters A. The remaining queries are split as usual.
- When the query being judged is an authentic R query, its own ID, and any R query with cosine > 0.97 to it, are excluded from its neighbour list.

**Reason.** Otherwise the judge weights could be tuned on queries that are test queries under another seed, and an authentic query would trivially find itself as its own neighbour.
**Alternatives.** Keeping calibration items out of both R and A (this loses them as neighbours and shrinks R).

### D-004 · 2026-09-27 · Judge backend (I-7) · team
**Choice.** All main results use Ollama with the same frozen model digest and prompt, locally and on Kaggle. Kaggle runs check the digest against `env/models.json`, and a 50-prompt parity test compares local and Kaggle outputs. vLLM is allowed only as a documented compute fallback: a new D-entry, a parity report, and a visible label on every affected result.
**Reason.** vLLM (HF fp16 weights) and Ollama (quantised GGUF) are different judges.
**Alternatives.** vLLM everywhere (faster, but a different model from the one run locally).

### D-005 · 2026-09-27 · HiAQ relevance labelling (I-8) · team
**Choice.** The full protocol (spec §6.3 step 5), budgeting about 10–12 h per annotator. The 30% fallback is used only if the full protocol becomes genuinely infeasible, and is then recorded as a deviation.
**Reason.** Pooled qrels make D4 much more credible, and D4 is the new-dataset contribution (C3).
**Alternatives.** The 30% fallback from the start.

### D-006 · 2026-09-27 · Code on Kaggle (I-18) · team
**Choice.** `ha kaggle pack` builds a `git archive` zip of a clean commit, which is uploaded as a new version of a private Kaggle dataset. Notebooks record the commit SHA. No GitHub token is ever stored in Kaggle.
**Alternatives.** A read-only GitHub token as a Kaggle Secret (rejected).

### D-007 · 2026-09-27 · Timeline (I-16) · team
**Choice.** The timeline is phase-based, counted in weeks from the start. No exam dates are assumed, and HUMAN GATEs are scheduled around the team's exam calendar.

### D-008 · 2026-09-27 · Everything on D: (I-11) · team
**Choice.** The repo, venv, data, caches, model caches and generated artefacts all live on D:. The repo and `HA_ROOT` are `D:\humanalign-rag`, with `data/` and `cache/` inside it (git-ignored). `results/runs/` is git-ignored until the artefact release. The original folder `C:\Users\chinm\Desktop\Projects-SEM5\AIML_EL` keeps only a pointer.
**Reason.** C: has about 23 GB free, and the project needs about 40 GB.

### D-009 · 2026-09-27 · Scoped seals (I-4) · agent
**Choice.** `HA_UNSEAL` is `0`, `1`, or a comma-separated list drawn from `cal_test`, `heldout_A` and `heldout_systems`. Every granted access is appended to `results/unseal_log.jsonl`.
**Reason.** The spec unseals `cal-test` at the end of P4, before the global freeze in P5. A single flag cannot express that.
**Alternatives.** One global flag, which would unseal A and the held-out systems at the P4 gate.

### D-010 · 2026-09-27 · E0 sanity checks for held-out systems (I-5) · agent
**Choice.** On FiQA, E0 covers only the dev systems: S1 against 0.236 ± 0.02, and dense dev models against their model cards ± 0.03. The held-out systems are checked on BEIR SciFact against published MTEB/BEIR scores ± 0.03.
**Reason.** Computing FiQA metrics for held-out systems before the freeze would break the seal. SciFact catches the same preprocessing bugs.

### D-011 · 2026-09-27 · Reuse the existing model caches · agent
**Choice.** Keep `HF_HOME=D:\ai-models\huggingface` and `OLLAMA_MODELS=D:\ai-models\ollama`, which were already set as user environment variables, instead of the spec's `D:\hf_cache`.
**Reason.** Both are on D:, and reusing them avoids downloading models twice. The spec allows a different layout if it is logged (§4.2).

### D-012 · 2026-09-27 · Python and awkward packages (I-10) · agent
**Choice.**
- Python 3.11.15 managed by uv, installed in `D:\ai-envs\uv-python`, with the venv at `D:\humanalign-rag\.venv`.
- PyTorch from the `cu128` index (the driver supports CUDA 13.1).
- fastText through `fasttext-wheel`.
- IndicXlit is never installed on Windows. It runs once in Colab to build the romanised-Hindi lexicon, which is then committed.

**Reason.** 3.11 matches the spec. IndicXlit's fairseq dependency is unreliable on Windows, and it is only needed for a one-off lexicon build.

### D-013 · 2026-09-27 · E10 disclosure for D4 (I-6) · agent
**Choice.** `cal-dev` contains HiAQ items, so D4 is not fully held out from the judge. The paper says so, and a robustness row applies weights and anchors fitted only on the FiQA part of `cal-dev` to D4. The "fully held-out" claim is made only for D2 and D3.

### D-014 · 2026-09-27 · Survey variant assignment (I-15) · agent (hosting needs team approval)
**Choice.** A Google Apps Script builds 25 forms, and a static random-redirect page gives respondents a single link.
**Reason.** Google Forms cannot randomly assign respondents to variants.

### D-015 · 2026-09-27 · Registry provenance · agent
**Choice.** `git_sha` gets a `-dirty` suffix when tracked files have uncommitted changes. `ha paper build` (P8.1) will refuse rows with a dirty SHA. Callers can never pass the auto fields.
**Reason.** Every reported number must map to exact code.

### D-016 · 2026-09-27 · Freezing the pre-registration · agent
**Choice.** `docs/preregistration.md` is committed as a DRAFT. When the team approves it, a commit changes its status to FROZEN and is tagged `prereg-v1`. After that the file is append-only, with dated amendments.
**Reason.** The tag gives verifiable evidence of what was registered, and when.

### D-017 · 2026-09-27 · Operationalising the H2 clauses · agent (the team confirms at the P0 gate)
**Choice.**
- "Higher τ_b" on a dataset means the one-sided paired bootstrap p-value of the seed-averaged Δτ_b is < 0.05 after Holm correction across D2–D4.
- "At least as good as the best non-LLM baseline on average" means the point estimate of mean Δτ_b (M1 minus the best of B2–B7 by mean τ_b), averaged over D2–D4, is ≥ 0, reported with its 95% CI.

**Alternatives.** A non-inferiority margin (for example −0.05). Rejected because it would need a justification we don't have.

### D-018 · 2026-09-27 · PYTHONUTF8 as a user environment variable · agent
**Choice.** `PYTHONUTF8=1` is set as a Windows user environment variable, and all code also passes `encoding="utf-8"` explicitly.
**Reason.** UTF-8 mode must be on when the interpreter starts, so setting it in `.env` is too late. It is needed for Devanagari I/O. To undo it, remove the variable from the user environment.

### D-019 · 2026-09-27 · Temp files on D: · team (options 1 and 3)
**Choice.**
- `ha.paths` sets `TEMP`/`TMP` to `HA_TMP` (default `D:\humanalign-rag\cache\tmp`) for every Python process in the project.
- Install and build shell commands set `$env:TEMP` to the same folder.
- The system-wide `TEMP` is unchanged.

**Reason.** C: slowed down badly during the PyTorch install. Everything else was already on D:; temp files were the remaining project-driven C: activity.
**Alternatives.** Changing the user-level TEMP (rejected: it affects every program); Windows Defender exclusions for the project folders (offered; the team did not choose it).

---

## Scope correction, 2026-09-27 (prototype-first)

### D-020 · 2026-09-27 · Prototype-first scope · team
**Choice.** The project is a research prototype for a paper, not a full evaluation platform. Work is split into three tiers, **CORE PROTOTYPE** (required), **OPTIONAL ROBUSTNESS** (later) and **FULL-SCALE** (future work), as defined in `PLAN.md` v2 §1. The research controls are kept in full:
- held-out authentic evaluation data;
- no leakage between judge or reference data and evaluation;
- a frozen judge;
- human calibration;
- deduplication;
- language and code-mixing analysis;
- negative results reported;
- no tuning on held-out data.

**Reason.** A small end-to-end experiment done cleanly (3+ query sets × 7 configurations × 5 seeds) beats a 20-system model zoo that can't be executed or interpreted on this laptop.
**Supersedes.** The 20-system pool and model list of spec §4.4 and §12 (they move to FULL-SCALE). D-004, D-010 and parts of D-002, as amended below.

### D-021 · 2026-09-27 · Locked local model stack; no downloads without asking · team
**Choice.**
- `qwen3:4b` (Ollama) is both the generator **and** the judge.
- `bge-m3` (Ollama) provides the embeddings and neighbours.
- `BAAI/bge-reranker-v2-m3` (the existing HF cache) does the reranking.
- BM25 needs no model.

Phase 0 downloads zero models, and the project's code cannot download models (`ha models pull` has been removed). Any further model needs the team's approval before it is downloaded.
**Consequence.** Generator and judge are the same model, so there is a risk of self-preference. Mitigations: human calibration on Qwen3-generated items; the M1-iso control; an optional `qwen3:8b` judge check (already on disk). This is disclosed as a limitation.
**Amends D-004.** Ollama everywhere still holds, but the judge is now `qwen3:4b`, not `qwen3.5:4b`.

### D-022 · 2026-09-27 · No dev/held-out system split · agent (follows from D-020)
**Choice.** All 7 retrieval configurations are used for every agreement measure. Protection against tuning comes from the held-out **datasets** (D2, D4; `heldout_A` sealed) and from making every design choice only on D1 and `cal-dev`. Selection never looks at system results.
**Reason.** Splitting 7 configurations into 3 and 4 leaves τ_b over 3 or 6 pairs, which is meaningless.
**Alternatives.** Keeping the split with more BM25 and fusion variants (rejected as artificial padding).
**Supersedes D-010.** E0 compares C1 against BEIR FiQA (0.236 ± 0.02); C3 against the MTEB score is a warning only.

### D-023 · 2026-09-27 · Seven retrieval configurations · agent
**Choice.**
- C1: BM25 (0.9 / 0.4)
- C2: BM25 (1.2 / 0.75)
- C3: BGE-M3 dense
- C4: RRF(C1, C3)
- C5: C3 → reranker
- C6: C1 → reranker
- C7: C4 → reranker (top-50 in each case)

**Reason.** They use only the locked stack. With 7 configurations τ_b has 21 pairs, compared with 10 for 5. Their marginal cost is small: the corpus is encoded once and reranker pair scores are cached.

### D-024 · 2026-09-27 · Ollama BGE-M3, dense only · agent
**Choice.** Use the Ollama `bge-m3` (F16) for all embeddings. BGE-M3 sparse and dense+sparse configurations need the HF weights (~2.3 GB download), so they become optional and are subject to approval.

### D-025 · 2026-09-27 · Reranker repair by hard links · agent (runs after the team's go-ahead)
**Choice.** The existing download in `D:\ai-models\hf-cache` has complete blobs but 0-byte snapshot files. `ha models repair-reranker` creates `D:\ai-models\local\bge-reranker-v2-m3\` from NTFS hard links to those blobs, identified by their content. This needs no download and no extra space, and leaves the original cache untouched.

### D-026 · 2026-09-27 · Core datasets · agent (download awaiting approval)
**Choice.**
- D1 FiQA (dev): HF `BeIR/fiqa` and `BeIR/fiqa-qrels`, ~28 MB.
- D2 CQADupStack-Android (held-out): HF `mteb/cqadupstack-android`, ~15 MB.
- D4 HiAQ (held-out, code-mixed): our survey.
- D3 MIRACL-Hindi becomes optional.

The pool is 800 source passages per dataset, about 1,600 queries (P-std + P-persona). P0.3's throughput measurement re-checks this.
**Amends D-002.** The seed tiers are unchanged.

### D-027 · 2026-09-27 · H2 as a pooled agreement test; HiAQ stays core · team
**Choice.**
- The primary test is the **pooled** comparison over the two held-out datasets (D2 and D4). It is framed as system-ranking **agreement** with authentic queries (τ_b(M1, A) against τ_b(B0, A) and τ_b(B1, A)), not as M1 "winning".
- The per-dataset results are secondary breakdowns, without significance claims.
- H2 holds if both pooled one-sided bootstrap tests have p < 0.05 (an intersection-union test, so no correction is needed).
- HiAQ is **core**: it is the only source of authentic Indian multilingual and code-mixed queries. Its survey scope is kept manageable.

**Supersedes** D-001's "≥2 of 3 held-out datasets" and D-017's per-dataset Holm rule.

### D-028 · 2026-09-27 · Generator/judge model: the local qwen3:4b is thinking-only · team
**Finding (P0.3).**
- The local `qwen3:4b` (digest `359d7dd4…`, 262k context) is the Qwen3-4B-2507 **thinking** build.
- Neither `think=False`, `/no_think`, an empty-think prefill nor JSON constraints stop it writing reasoning into generated queries.
- `think=True` used 2,048 tokens (49 s) without producing an answer.
- The local `qwen3:8b` (digest `500a1f06…`, 40,960 context) is the original hybrid Qwen3, and works with `think=False`.

**Options.**
1. Use `qwen3:8b` (on disk, no download) as generator and judge. It is partly CPU-offloaded, so it is slower (to be timed).
2. Download `qwen3:4b-instruct-2507` (~2.5 GB, the non-thinking sibling of the locked model). Needs approval.
3. Keep the thinking `qwen3:4b` with thinking on. Infeasible (about 50 s or more per call).

**Status: RESOLVED on 2026-09-27. The team chose option 2.**
- **Downloaded:** `qwen3:4b-instruct-2507-q4_K_M` (2.33 GB on D:, 0 on C:). The shorthand `qwen3:4b-instruct-2507` does not exist in the Ollama library. The chosen tag pins the Q4_K_M quantisation, the same as the thinking build.
- **Digest:** `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`. It is pinned in `configs/models.yaml`, `configs/generation.yaml` and `configs/judge.yaml`, and `ha.llm.require_digest` refuses to run on any mismatch.
- **Not used:** the thinking-only `qwen3:4b` (= `qwen3:4b-thinking-2507-q4_K_M`) stays on disk but is excluded from the core; a test enforces this. The Ollama library confirms that `qwen3:4b` and `qwen3:4b-thinking-2507-q4_K_M` share ID `359d7dd4bcda`.

**Verification** (`tools/bench_llm.py` → `reports/phase_0_llm_verification.json`):
- **Generation:** 20/20 outputs are genuine queries; no reasoning markers; none hit the 64-token cap; median 0.54 s.
- **Judge:** 20/20 valid JSON; median 1.60 s; VRAM 2.96 GiB.
  - F is 4.5 on matched passages and 1.0 on mismatched controls.
  - No permutation pair differs by 2 or more (0/8).
- **Caveats for P4 calibration, not to be acted on now** (prompt changes happen only on `cal-dev`):
  - v0 gives H 4 and S 3–4 to exam-style P-std queries ("…according to the passage?"), which the anchors say should score low.
  - `reason` disagrees with `keep` (for example `keep=true` with `too_document_like`).
  - On mismatched passages, H, S and P all collapse to 1 (a halo effect).

### D-029 · 2026-09-27 · Pre-registration review against the team's 10 locking criteria · agent (the team approves before the freeze)
**Finding.** Revision 2 fully locked criteria 1 and 2 only. Criteria 3–10 were partly specified, and several free choices that could affect the primary result were not fixed.

**Changes in revision 3** (see `git diff` of `docs/preregistration.md`):
- A §0 checklist.
- §3.3 is an explicit isolation table. `cal-dev` is the only data allowed to drive prompts, anchors and weights; A of D2 and D4 is evaluation-only after the freeze.
- §4 fixes the calibration procedure: item composition and seeds; practice items outside the 200; the κ rule; adjudication; the version-selection criterion, **excluding anchor items from ρ**; NNLS weights fitted separately for M1 and M1-iso; the gate fixed a priori; no calibration outcome stops the study.
- §5 fixes every setting:
  - full model digests and the reranker revision;
  - an exact M1-iso definition;
  - a size-matching rule for gate shortfalls;
  - BM25 tokenisation, RRF k, rerank depth and max_length;
  - D4 qrels pooling over C1, C3, C4, C5, finished before the freeze.
- §6 lists the contents of `FROZEN.json`, including the analysis-code SHA and the bootstrap seed. v1 stays primary.
- §7 fixes the statistics: nDCG via `ranx` `ndcg@10` (linear gain); `scipy` τ_b; the bootstrap scheme and seed; the pooled intersection-union rule; a lock clause.
- §8 requires reporting every hypothesis and every optional analysis that is run, whatever the result.
- H1 is relabelled as secondary and descriptive (it was "primary", which conflicted with a single primary test). H1 and H4 get explicit procedures, and H4 a minimum class size of 50.

**Choices the team approved on 2026-09-27 (all six):**
- M1-iso gets its own NNLS weights.
- N = min(|A|, the number passing each judge's gate).
- The D4 pooling configurations.
- The H4 minimum class size of 50.
- The linear-gain nDCG.
- The calibration seeds 17 and the bootstrap seed 20260927.

### D-030 · 2026-09-27 · Pre-registration frozen as `prereg-v1` · team
**Choice.** The team approved revision 3 of `docs/preregistration.md` and all six D-029 choices:
1. M1-iso gets its own NNLS weights.
2. N = min(|A|, the number passing the M1 gate, the number passing the M1-iso gate), reported whenever it binds.
3. D4 pooling uses C1, C3, C4 and C5, never changed in response to results, with the bias disclosed.
4. H4 analyses a class only if it has at least 50 held-out queries.
5. `ranx` linear-gain nDCG.
6. Seed 17 for calibration sampling and 20260927 for the bootstrap.

The status is set to FROZEN, and the freeze commit is tagged `prereg-v1`. `results/PREREG.json` records the commit, the tag and SHA-256 hashes of the frozen text and the configuration artefacts. A test checks that the frozen body (everything above `## Amendments`) is unchanged.

**Rule from here.** No methodological change without an explicit v2; v1 stays the primary pre-registered analysis.

---

## Phase 1

### D-031 · 2026-09-27 · HiAQ source vetting · agent (spec §0.1 and §6.2; the team may override)
Checked each source's robots.txt and its **live** website policy. Two search-engine summaries turned out to be stale: the live Parivahan and EPFO pages were checked directly. Details are in `configs/hiaq_sources.yaml`.

| Source | robots.txt | Live policy | Decision |
|---|---|---|---|
| Income Tax e-filing help (incometax.gov.in) | none (redirects to HTML) | reproduction free with acknowledgement; no automated-access clause; linking needs prior permission | **include**; the release cites sources and publishes no links without permission |
| EPFO (epfo.gov.in) | allows all | "reproduced free of charge … without requiring specific permission" | **include** (FAQ plus 5 member-scheme pages) |
| NPCI / BHIM UPI | `Disallow: /` for `*`; ClaudeBot and anthropic-ai named | — | exclude |
| DigiLocker | 403 AccessDenied | — | exclude (permission can't be verified) |
| IRCTC | empty | terms forbid copying, reproducing or publishing content | exclude |
| ABDM (ABHA) | allows all | "may not be reproduced … without due permission" | exclude |
| Parivahan | none | live policy: reproduction only **after permission by email** | exclude unless the team obtains permission |
| india.gov.in | 403 (CDN) | not reachable | exclude |
| UIDAI (Aadhaar) | allows all | reproduction allowed (per search summary; to be verified on the live page) | **parked**: FAQ content is JavaScript-rendered, with no static pages |
| GST, Passport Seva | — | not verified or restrictive | exclude |

Income Tax: the sitemap lists about 288 English help pages that static links don't reach. They are seeded from the sitemap with URL fixes (http→https, doubled `/iec/foportal/`, fragments). The crawler uses at most 1 request every 2 s, an identifying User-Agent, robots checks per host, and logs every fetch with its URL and time.

### D-032 · 2026-09-27 · Language ID without a model · agent
Tokens are tagged hi / en / ne / univ using: Devanagari script; a hand-curated romanised-Hindi lexicon (`src/ha/data/lexicons/rom_hi.txt`, chat spellings included); an English vocabulary with counts from the FiQA corpus (D1 only, count ≥ 5); and a service-name / acronym list. A Hindi word that also occurs at least 50 times in FiQA (to, main, me, is, us, do, par …) is ambiguous and takes the majority language of its text. Unknown Latin words are `univ` and excluded from the CMI. The class thresholds are in `configs/langid.yaml`; the ±0.05 sensitivity check is a parameter.

**Validation:**
- the 30 hand-labelled test examples (provisional until H-09) all pass;
- all 1,347 English queries of D1 and D2 are classed EN, so there are no false code-mix positives;
- D4 gets the human 20% check (H-10).

fastText LID is not used (it would be a download, D-021).

### D-033 · 2026-09-27 · HiAQ chunking · agent
- FAQ items become one chunk each: `<details>/<summary>` accordions, or a block of at most 40 words ending in "?" followed by its answer.
- Other text is chunked by h1–h4 section, in windows of at most 350 words with 50 overlap.
- "Tokens" are whitespace words.
- Boilerplate: block texts appearing on at least 30% of a source's pages (and at least 3) are dropped.
- Dedupe: exact after normalisation, then MinHash (128 permutations, word 3-shingles) at Jaccard ≥ 0.9.
- Minimum length: 20 words for sections, 3 for FAQ answers.

### D-034 · 2026-09-27 · Split stratification clusters (an implementation detail within prereg §3.2) · agent
The topic clusters used only for splitting are k-means on the bge-m3 embeddings of all authentic queries, with k = min(10, ⌊n/30⌋) (at least 1) and seed 0. Within each language × cluster stratum, round(0.4·n) queries go to R, CAL-auth first, then a draw seeded `1000 + split_seed`. CAL-auth is drawn with seed 17, proportionally stratified by language class.

**Result:**
- FiQA: 648 queries, R = 260 and A = 388 for every seed, 30 CAL-auth.
- CQADupStack-Android: 699 queries, R = 279 and A = 420.

This does not change the prereg: §3.2 leaves k unspecified, and the split involves no outcome data.

### D-035 · 2026-09-27 · Scenario chunks come from individual-user topics only · agent
The HiAQ corpus (1,572 chunks after link/menu filtering: 1,418 Income Tax and 154 EPFO; 725 FAQ pairs and 847 sections) keeps **all** vetted content, including professional topics (statutory forms, companies, firms and LLPs, CAs, deductors) as retrieval distractors. The **scenario sheet** samples only chunks outside those topics (`PROFESSIONAL_TOPIC_RE`, with word boundaries so "confirm" and "where" are not excluded), because survey respondents are ordinary users.
- Allocation across services is proportional to √(chunks), so EPFO is not drowned out: 111 Income Tax and 39 EPFO rows over 117 topics.
- Seed 17; authors alternate A/B (75 each).

### D-036 · 2026-09-27 · Scenario word-overlap WARN · agent
**Finding.** A sanity check showed the spec §6.3 leakage rules alone are weak: using each chunk's own FAQ question as a "scenario" was rejected only 6/20 times.
**Choice.** Add a third signal, **WARN only**: a scenario for which at least 60% of its content words occur in its chunk. All 58 FAQ-question pseudo-scenarios scored ≥ 0.6 (median 1.0); hand-written situations scored 0–0.5. WARN rows get a human review. The spec's two automatic REJECT rules are unchanged.

### D-037 · 2026-09-27 · Scraper safety stop · agent
After 3 consecutive 403 or 429 responses, a source stops. During the Income Tax run, 403s were page-specific (5 of 399 responses; 235 pages loaded after the first one), and the stop fired on two forbidden payment-help URLs. We did **not** resume: 301 pages had already been fetched (more than the 288 English help URLs in the sitemap), and 1,572 chunks meet the spec target of 1,500–4,000. A bug where the stop skipped logging the final request is fixed; every request is now logged.

### D-038 · 2026-09-27 · Phase 1 team decisions: scenario sheet 75/75, hosting, optional sources · team
- **Scenarios:** the team lead writes all 150 (75 Income Tax + 75 EPFO), runs `ha data hiaq-check`, and reviews every WARN. The agent never writes or substitutes scenarios. The sheet was regenerated **while still empty** with fixed quotas (`configs/survey.yaml`; seed 17; 118 eligible EPFO chunks, 81 topics). This replaces D-035's √ allocation for the sheet only; the corpus is unchanged.
- **Launch** waits for faculty-advisor ethics approval (H-04).
- **LID labels:** the team reviews the 30 examples in `tests/test_langid.py` before that part of the gate counts as approved.
- **Hosting:** the redirect page goes on GitHub Pages; the 25 Google Forms are the survey endpoints.
- **Optional sources:** no UIDAI headless-browser dependency. Parivahan permission is not on the critical path and is a possible future extension. The Income Tax + EPFO corpus (1,572 chunks) is the core D4 corpus.
- **Phase 2 is approved** (FiQA + CQADupStack-Android: bge-m3 embedding and C1–C7). Phases 3 and 4 wait for the Phase 1 / data gates.

## Phase 2

### D-039 · 2026-09-27 · Phase 2 scope under the seal; C3 settings and E0 reference · agent
- **Held-out isolation (prereg §3.3).** Every CQADupStack-Android query is in A for at least one split seed. So before the freeze, D2 gets **only corpus-side work**: bge-m3 corpus encoding, BM25 indexes, and a smoke run on 10 made-up queries (never scored). `run_all.query_set(ds, "auth")` for a held-out dataset goes through `require_unsealed("heldout_A")`, and a test enforces it. D2 authentic runs, metrics and latency happen after the freeze.
- **C3 `num_ctx` = 8192** (the bge-m3 limit), with `truncate` on. Passages are sent untruncated up to the model limit, as §5.3 requires. The value is fixed in `configs/systems.yaml`.
- **Latency:** median of 3 timed passes over a fixed 200-query FiQA sample, caches off. Ollama and torch stages are timed separately and summed for C5–C7, because they can't share the GPU.
- **E0 references:**
  - C1: BEIR 0.236 ± 0.02 (gate).
  - C3: **0.413 ± 0.03, warning only.** Source: DRAMA, arXiv 2502.18460, Table 7, "M3-BGE-Dense" on FiQA, checked on the primary source. A search summary also mentioned "40.8" without a traceable source, so it was not used.

### D-040 · 2026-09-28 · Authentic noise ceiling: EXPLORATORY only · team
**Choice.** Report, descriptively, how well authentic queries agree with themselves: τ_b over C1–C7 between (a) 1,000 random 324/324 halves of the 648 FiQA queries (exploratory seed **20260928**, documented here, not pre-registered) and (b) R_s against A_s for the frozen split seeds 0–4. Also report the per-pair flip rate.
- **Data:** D1 (development) only, reusing the E0 per-query nDCG@10 tables. Held-out D2/D4 are never touched.
- **Status:** exploratory under prereg §7.7. It does **not** modify H2, the configuration pool, prompt or model tuning, query selection or the freeze, and it is **not** a hypothesis. Registry rows are tagged `exp = EXP-noise-ceiling` and `exploratory = true`.
- **Code:** `ha explore noise-ceiling` → `results/EXP_noise_ceiling.json`.

**Addendum (same day).** The first run (`EXP-noise-ceiling-20260928T011823-654ce0`) showed that complementary halves are anti-correlated on every pair difference: their means sum to the overall mean. So near-tie pairs flip almost always (C4–C6 in 95.9%, C1–C2 in 94.1%), and the random-half τ_b **understates** agreement between independent query sets. A sensitivity variant (d) is added, clearly labelled and still exploratory: two **independent** resamples (with replacement, 324 each), seed **20260929**. The random-half result is kept and reported as approved; (d) is the matching reference when an independent synthetic set is compared with A. The rerun also re-records (a), (b) and (c) under a new run_id; the first run stays in the registry.


## Phase 1 human-side intake (2026-09-28)

### D-041 · 2026-09-28 · Ethics approval reported · team
`data/humanside/humanalign_phase1_review.md` reports faculty-advisor ethics approval as **DONE**. **Still to add here:** approver name and role, the approval date, the scope (survey and rating study), and any conditions. The survey still waits for the scenario and LID gates below.

### D-042 · 2026-09-28 · HiAQ scenario draft received: ChatGPT-assisted, pending team-lead approval · team (agent intake check)
- **Input:** `data/humanside/scenario_sheet_chatgpt_draft.csv`, 150 rows. The source fields are identical to the generated sheet (same chunks, same order, 75 Income Tax / 75 EPFO). `author` = "ChatGPT-assisted draft; team-lead approval required". Median scenario length is 29 words (max 43).
- **Deviation from spec §6.3,** which has scenarios written by a team member: **drafting was LLM-assisted.** It is acceptable only with per-row team-lead review and approval, and it **must be disclosed in the paper**. Risk to disclose: LLM-written situation text may prime respondents' phrasing. The authentic queries themselves remain human-typed.
- **Agent leakage check** (not yet the official check; `data/humanside/scenario_check_draft.csv`): **141 PASS, 2 WARN** (S060, S066), **7 REJECT** (all EPFO: S005 contractor, S007 job/jobs, S067 lump, S090 unmarried, S091 military/serviceman, S096 enroll, S137 daughter). The rejects are triggered by the spec's rare-term rule, mostly on everyday words that are rare in the tax-dominated corpus. Only 3 of 81 FAQ-chunk scenarios cover at least 60% of their FAQ question's words.
- **Professional role-play rows,** unsuitable for lay respondents: **S017** (tax filing agent), **S025** (chartered accountant), **S136** and **S147** (authorised tax intermediary). They passed because `PROFESSIONAL_TOPIC_RE` missed "ERI" and "(CA)" headings. The filter is now extended (ERI, "(CA)", chartered, "type of user"), with a test; it only affects future replacement draws. The sheet was **not** regenerated.
- The agent did **not** edit, approve or substitute any scenario. The decision is the team lead's.

### D-043 · 2026-09-28 · LID audit expanded to 100 examples; human labels pending · team
- **Input:** `data/humanside/langid_review_100.csv`: the original 30 plus 70 added boundary cases. Provisional labels: EN 20, ROM_HI 20, HINGLISH 20, DEV_HI 21, OTHER 19 (non-target scripts, as a control).
- `human_final_label` is **empty for all 100 rows**, so gate H-09 is **not yet met**.
- **Blind protocol:** the agent will not show per-row system predictions until the human labels are final, so the human review stays independent of the system. Afterwards: human-vs-system agreement, the disagreement list, and `tests/test_langid.py` updated to the final labels (a data change, not a code change).
- The protocol's own wording is adopted for the paper: "a 100-example hand-labelled LID audit/validation set, not a representative sample".
- `data/humanside/` is now versioned in git (a `.gitignore` exception), so approved human artefacts get commit hashes.


### D-044 · 2026-09-28 · Scenario and LID review artifacts processed; nothing approved · agent (on team instruction)
**Source artifacts** (team, 2026-09-28), all in `data/humanside/`:
- `scenario_sheet_chatgpt_draft.csv`: the 150 ChatGPT drafts. For the 7 REJECT rows, the current draft text **is** the ChatGPT rewrite candidate.
- `langid_review_100.csv`: in it, `provisional_label` **is** ChatGPT's recommendation.
- There are no separate "reviewed" or "recommendations" files.

**Tools added** (commit `1cda5f2` and after):
- `ha data hiaq-check --sheet X --out Y` never modifies the sheet it checks. A non-official sheet writes `<stem>_check.csv` beside itself; the official `data/hiaq/scenario_check.csv` is written **only** for the official sheet.
- `ha data hiaq-replace <ids> [--exclude <chunk_id> …]` draws replacement **chunks** (never scenarios) deterministically:
  - same source;
  - the sheet's eligibility rules (at least 30 words; not a professional topic by URL or heading);
  - not already in the sheet and not excluded;
  - topics not yet in the sheet are preferred;
  - ties broken by a seed-17 permutation of the chunk_id-sorted candidates.
- `tools/build_lid_adjudication.py` builds the table **without ever filling `human_final_label`**, and without system predictions (the review stays blind).

**Results** (the official sheet's sha256 `28176bc2…` is identical before and after; no official check file exists):
- **Checker on the draft** (`data/humanside/scenario_sheet_chatgpt_draft_check.csv`): 141 PASS / 7 REJECT (S005, S007, S067, S090, S091, S096, S137) / 2 WARN (S060, S066). Same as the intake check.
- **Replacement chunks** (`data/humanside/scenario_replacements.csv`; scenarios empty, status NEEDS HUMAN-WRITTEN SCENARIO):

  | Replaces | New chunk | Topic |
  |---|---|---|
  | S017 | `incometax/1fa8efc9b1cd/002` | ITR-2 user manual, contact and bank details |
  | S025 | `incometax/6cebf3b156cc/013` | new vs old tax regime FAQ |
  | S136 | `incometax/306ec57bb9dd/004` | comply-to-notice FAQ, authorised representative; borderline, taxpayer side |
  | S147 | `incometax/d51b96a53e6a/010` | ITR-2 FAQ, "what is a capital asset" |

  **Flag:** the S025 replacement is about ITR-5 and Form 10-IEA for AOP/BOI, which is professional. The filter checks URL and headings, not the chunk text. The draw rule was **not** changed after seeing this result. If the team rejects it, the next deterministic candidate comes from `ha data hiaq-replace S025 --exclude incometax/6cebf3b156cc/013` (the other three are unaffected).
- **LID adjudication table** (`data/humanside/langid_adjudication_table.csv`): 100 rows. Columns: example_id, text, source, original_test_label (30 originals), chatgpt_recommendation, human_final_label (HUMAN-CONTROLLED, empty), human_attention. Exactly the 10 team high-priority cases are flagged.

**Status:** neither the scenario gate nor the LID gate is passed. The ethics metadata placeholder is in `data/humanside/humanalign_phase1_review.md`; nothing is invented. Frozen files are untouched: the pre-registration, configs, `tests/test_langid.py` and the LID rules.


### D-045 · 2026-09-28 · ChatGPT review recommendations applied to the REVIEW ARTIFACT only; nothing approved · agent (on team instruction)
**Scope.** `data/humanside/scenario_sheet_chatgpt_draft.csv` (the review artifact) was updated. Every row is marked `pending_human_approval`, with new columns `review_action`, `review_status`, `previous_chunk_id`, `previous_scenario` and `agent_flag`. **The official `data/hiaq/scenario_sheet.csv` is byte-identical** (sha256 `28176bc2a58d…`), and no official check file exists. `langid_review_100.csv` is byte-identical (sha256 `1c24efd31cf3…`).

**Scenarios**
- **Rewrites** (verbatim as proposed by ChatGPT; `review_action = rewrite_proposed`): S005, S007, S067, S090, S091, S096, S137.
  - Checker: S090, S096, S137 PASS; S005 (contractor), S007 (job/jobs), S067 (lump), S091 (military) still REJECT on the rare-term rule. They were not reworded to beat the checker.
  - Support flags after inspecting each source chunk: S005 (worker checking is only indirectly supported); **S090 drops the "nominate someone outside the family" need**; **S091 omits the deceased's PF/EPS membership**; **S096 is employer role-play**.
- **WARN review:** S060, S066 (kept provisionally).
- **Skip and replace:** S017, S025, S136, S147. The chunk fields were swapped to the replacement chunk; the originals are kept in `previous_chunk_id` and `previous_scenario`.
  - Replacements, by the deterministic draw (seed 17, same service, same filter): S017 `incometax/1fa8efc9b1cd/002`, S136 `incometax/306ec57bb9dd/004`, S147 `incometax/d51b96a53e6a/010` (ChatGPT: ACCEPT).
  - S025: the first draw `incometax/6cebf3b156cc/013` got ChatGPT REJECT (professional). The re-draw `ha data hiaq-replace S025 --exclude <the rejected chunk> --exclude <3 accepted chunks>` gave `incometax/6cebf3b156cc/003` (an employee telling the employer their tax regime); agent: suitable. Raw output: `scenario_replacements_s025_redraw1.csv`. `scenario_replacements.csv` now carries `chatgpt_recommendation` and `decision_status = pending_human_approval`.
  - The four replacement scenarios are **agent drafts (Claude Code), not ChatGPT**, and are labelled as such in `author`. All pass the checker.
- **Checker on the review artifact** (`scenario_sheet_chatgpt_draft_check.csv`): 144 PASS / 4 REJECT / 2 WARN.

**LID**
- `langid_chatgpt_recommendations.csv`: the 10 explicit ChatGPT recommendations, plus provisional_label for the other 90. All are marked "NOT a human label".
- The adjudication table was rebuilt with `--recs`. Columns: provisional label, ChatGPT recommendation, human_final_label (HUMAN-CONTROLLED, empty) and human_attention (10 flagged). L54 and L55 are recommended HINGLISH against provisional ROM_HI.
- The builder now fails if any recommendation is missing, and flags recommendation ≠ provisional.
- **Unchanged:** `src/ha/data/langid.py`, `configs/langid.yaml`, `tests/test_langid.py`.

**Ethics.** Placeholder only; nothing inferred.

**Gates.** Neither the scenario gate nor the LID gate is passed. Phase 3 must not start. Not committed (team instruction).


## Phase 3

### D-046 · 2026-09-28 · P3 scope and generation-engine implementation details · agent (team-authorised P3)
**P3 generation is proceeding in parallel with the remaining P1 human gate.** Checklist: `reports/phase_3_checklist.md`.

**Scope, from spec §7 as overridden by its banner (PLAN v2 and decisions win on pool size and generators) and prereg §5.2:**
| Dataset | Pool | Decision |
|---|---|---|
| D1 fiqa, D2 cqa_android, D4 hiaq | P-std, P-persona | **run** (core U). D4: the corpus is final per D-038. |
| D1 fiqa, D2 cqa_android | P-fewshot for split seeds 0–4 | **run** (team-authorised; optional-robustness tier; exemplars from R_s only) |
| D4 hiaq | P-fewshot | deferred: no R before the survey |
| D1–D3 | P-std-G2 | **stopped**: generator G2 (gemma3:4b) is not available locally; no substitution |
| D1, D2 | P-docT5 | **stopped**: a dataset download that needs approval; FULL-SCALE tier |
| D3 | all | **stopped**: MIRACL-Hindi was never downloaded (D-026) |

**Frozen settings used unchanged** (`configs/generation.yaml`, not modified):
- `qwen3:4b-instruct-2507-q4_K_M` (digest `0edcdef3…`, checked), `think=False`;
- temperature 0.7, top_p 0.9, `num_predict` 64;
- seed per call = 1000 + i;
- 800 passages, sample seed 7.

Templates: `gen_std_v1`, `gen_persona_v1`, and **`gen_fewshot_v1` added verbatim from spec Appendix B.3**. The template name and k = 5 are constants in code, because `generation.yaml` is a frozen, hashed artefact.

**Implementation details the spec leaves open** (engineering, not method):
- **Reproducibility.** The smoke test showed that Ollama's KV-prefix reuse makes a repeated identical prompt sample differently at T 0.7 (7 of 9 uncached repeats identical), while the first call after a fresh model load is stable. So `generate_pool` unloads the model first, and every pool runs from a fresh state in fixed index order. A 6-passage sequence then reproduced 6/6.
  - The canonical pool is the logged and cached output.
  - A resumed run continues from a different model state; its new calls are still logged and cached.
- **Cache key:** (prompt, model digest, decoding options, think). Empty outputs are cached like any response. Only transport exceptions leave no cache entry, so only they are retried on a re-run.
- **Persona choice:** `default_rng([11, i])`. **Few-shot exemplars:** `default_rng([13, split_seed, i])`, k = 5 from R_s, sorted. It asserts every exemplar is in R_s; A is never loaded.
- **Passage text:** "title\ntext" for D1/D2 (as in retrieval); chunk text for D4.
- **Cleaning** (spec: strip quotes and numbering): first non-empty line; leading numbering or bullets and surrounding quotes removed; whitespace collapsed.
- **Tokens:** `[\p{L}\p{N}]+`, lowercased.
- **"First sentence":** the passage body (after a title line, if any) up to the first `[.!?]` followed by whitespace.
- **Overlap:** the share of the query's tokens found in the first sentence's token set; dropped if > 0.6.
- **Dedupe:** exact (lowercase, whitespace-normalised) first, then greedy keep-first within the pool at BGE-M3 cosine > 0.95 (the spec's threshold).
- **Truncation guard:** the Ollama default `num_ctx` is 4096 (not changed). A call is flagged `possible_truncation` if its prompt reaches ≥ 4096 − 64 tokens. Passages are short: 99th percentile 398–642 words; only 5 passages across D1/D2 exceed 2,000 words.
- **Pilot** (20 per pool, `data/synthetic_pilot/`): 0 failures, 0 duplicates.
  - The copy filter dropped 15–25% in the CQA pools and in HiAQ P-persona.
  - Inspection showed they are **genuine copies**: CQADupStack passages are user questions, and HiAQ chunks open with their FAQ question.
  - The filter was **not** changed.

### D-047 · 2026-09-28 · P3 AUTO GATE measured: FAIL (partial) on pool size · agent
**Full run:** 16 pools × 800 passages, 12,800 attempted calls. Report: `reports/phase_3.md`; stats: `reports/phase_3_full_stats.json`.
- **Failure rate < 5%:** **PASS, 16 of 16 pools** (maximum 0.13%).
- **Final unique ≥ 720 (90%):** **FAIL. 6 of 16 pass:** D1 P-std 727, and D1 P-fewshot s0–s4 = 780 / 772 / 779 / 773 / 783.
- **10 fail:**
  - D1 P-persona 716;
  - D2 P-std 665, P-persona 630, P-fewshot s0–s4 = 688 / 668 / 658 / 666 / 676;
  - D4 P-std 586, P-persona 645.

**Correction (2026-09-28):** the pool totals were first mis-added as "7 pass / 9 fail", in `reports/phase_3.md`, this entry, the docs and the report to the team. A consistency check against the per-pool `stats.json` files gives **6 pass / 10 fail**. The per-pool figures were always correct.

**Cause:** the spec's first-sentence copy filter, working as specified, on question-shaped passages (CQADupStack user questions; HiAQ FAQ chunks that open with their own question). Verified by inspection. Filter unchanged.

**Other facts:**
- The 1 failure in each D2 pool is pid 50091: a 7,885-token prompt against Ollama's default 4,096 context, rejected with an error rather than truncated. `num_ctx` unchanged.
- Few-shot (R-exemplar) prompting copies far less on FiQA (17–28 drops against 73–77).

### D-048 · 2026-09-28 · P3 decision: retain the pools as generated; P3 gate recorded as a partial failure · team
**Decision: option (a).** The 16 generated pools are **retained exactly as generated** for downstream selection. The P3 AUTO GATE is recorded as:
- **FAIL on the ≥ 90% pool-size criterion:** 6 of 16 pools meet it (see the correction in D-047);
- **PASS on the < 5% generation-failure criterion:** 16 of 16 pools meet it; maximum failure rate 0.13%.

**Rationale (team):**
- The pre-registered first-sentence copy filter behaves as specified.
- The shortfalls are corpus-dependent (question-shaped passages), not generation instability.
- Every pool meets the failure criterion.
- The retained candidate pools are still sufficient for the planned selection procedure: U = P-std ∪ P-persona is 1,443 (D1), 1,295 (D2) and 1,231 (D4), against a selection budget of N = |A| per seed (D2: 420). The prereg's size-matching rule N = min(|A|, gate-passing counts) applies later.
- The pre-registration specifies **no** minimum pool size (the 90% gate is a PLAN-level engineering gate).

**Explicitly not done:** no change to the filter; no top-up of source passages; no context-size (`num_ctx`) change; no change to sampling, model or generation parameters; no regeneration, deletion or rewriting of pools; **no pre-registration amendment**.

**This is a P3 gate outcome, not a termination of the study.** P3 is closed. Phase 4 has not started and waits for its own go-ahead. The Phase 1 human gate (scenarios, LID labels, ethics metadata) remains unresolved and independent.


## Phase 4

### D-049 · 2026-09-28 · P4 started: neighbours (D1, D2) and judge engine plus pilot; calibration blocked · agent (team-authorised P4; uncommitted)
**Implemented per spec §8–9 and prereg §4 and §5.2**, with no change to frozen settings. Report: `reports/phase_4.md`; checklist: `reports/phase_4_checklist.md`.

**Implementation details the spec leaves open** (engineering, not method):
- **Neighbours:**
  - Dense ties broken by R qid.
  - The top-5 ordered by rerank score, then cosine, then qid.
  - `nn_max_cos` is the maximum over all of R_s after masking. `nn_mean5_cos` is over the final reranked top-5.
  - The top-30 candidates with cosine and rerank scores are stored, so the neighbour ablations (top-1, no reranker) need no recomputation.
- **Reranker max_length:** `Reranker(max_length=…)` now accepts an override so neighbour reranking uses the prereg's 128. The default (512 from `systems.yaml`) is unchanged for C5–C7, and the pair-cache key includes max_length.
- **Permutation order:** a numpy RNG seeded with the first 16 hex digits of sha256("<qid>|<perm_idx>"). Permutation 0 is also shuffled ("order is random").
- **Third-permutation rule:** any of F, H, S, P, or C when both values are non-null, differing by ≥ 2. With 2 permutations the "median" is their mean, so dims can be x.5.
- **Failures:** if any permutation fails, the query fails (not selectable). A retry appends the invalid output as an assistant turn plus a user message with the validation error.
- **CV:** fold assignment seed 17 (the calibration seed).
- **Reproducibility:** each judge run starts from a fresh model state (unload), in fixed order; verified 4/4 uncached.

**Results (engineering):**
- Neighbours for D1 and D2, seeds 0–4: 0 near-duplicates; timings as in the report.
- Judge pilot `P4-judge-pilot-20260928T230859-1f27d5`: 38/38 valid JSON, 1.50 s per call median.

**Blocked:**
- **D4 neighbours and judging:** no D4 R until the survey.
- **P4.3 execution:** HiAQ CAL-auth (H-07).
- **P4.4–P4.5:** human annotation (H-08).

**Open team decisions:**
1. The passage shown for authentic calibration items.
2. Whether annotators see the neighbours.
3. The D4 domain description.
4. The annotation tool.

Nothing improvised; the annotation guidelines are a DRAFT.


### D-050 · 2026-09-28 · Calibration display decisions: qrels passage for authentic items, human neighbours from R_s0 \ CAL-auth, D4 domain text, spreadsheet instead of Label Studio · team (decisions 1 = A, 2 = D, 3, 4); implemented by agent (uncommitted)
**Why:**
- The first proposal (authentic items with no passage; annotators shown the judge's neighbours) conflicted with the frozen prereg:
  - F is required on all 200 items (§4.3), in the ρ criterion (§4.6) and in Appendix C ("read the passage").
  - Blinding (§4.3): a missing passage marks an item as authentic. And since CAL-auth is forced into R (§4.1), 42 of the 70 FiQA synthetic calibration items would show at least one authentic calibration query under "real user queries", exposing 20 of the 30.
- The agent stopped and reported this. The team chose the resolutions below.
- **No prereg amendment:** §4 does not specify the passage for authentic items, the tool, or what annotators see beyond blinding. These decisions fill those gaps within it.

**Decisions:**
1. **Passage for authentic calibration items (A):** a real qrels-linked passage, never written or chosen by hand.
   - **Rule:** among the item's qrels with the **highest grade** whose passage exists and is non-empty, sort the pids ascending, then take index `int(sha256("17|<dataset>|<qid>")[:16], 16) mod n`.
   - No length or content criterion: synthetic source passages were drawn uniformly with none either.
   - Text is formatted exactly as for generation (`load_source_corpus`).
   - Current state:
     - FiQA: 30/30 resolved (all qrels have grade 1; 19/30 have more than one qrel passage).
     - HiAQ: **pending** until its qrels exist (spec §6.3: the scenario's source chunk is grade 2 by construction; pooling follows). No substitute is used.
   - The same selected passage is the judge's SOURCE PASSAGE for that item in calibration. v0's wording "(the TEST QUERY was generated from it)" is frozen and unchanged. Prereg §4.5 allows v1–v4 to revise instruction wording on `cal-dev` only.
2. **Human-facing neighbours (D):**
   - **Rule:** the frozen P4.1 procedure (bge-m3 top-30 → reranker at 128 → top-5; authentic items: no self, nothing with cos > 0.97) run over **R_s0 minus all 60 CAL-auth items**.
   - Shown to annotators in a seeded random order (`permute(…, "human|<qid>", 0)`).
   - **A display rule only:** R, `data/neighbours/…` and the judge-side procedure are unchanged. The judge still retrieves from full R_s0.
   - The two are separate artefacts: human-facing neighbour qids live only in `data/calibration/key.csv` / `practice_key.csv`.
   - An assertion refuses any human-facing block that contains a CAL-auth item.
   - Dry run on the real FiQA half (100 items, no file written): 0 blocks with a CAL-auth item, 0 self-neighbours, all blocks full (the judge side for the same 70 synthetic items: 42).
3. **D4 domain description:** the team's exact sentence, in the new `configs/domains.yaml`. The dataset configs are hash-locked by PREREG.json and are not edited. `ha.config.domain_description(ds)` reads the dataset config first, then `domains.yaml`.
4. **Spreadsheet export, not Label Studio:** spec §9.4 step 1 and PLAN P4.3 said Label Studio; the prereg names no tool. Label Studio was not installed. The files are CSV, UTF-8 with BOM, in `data/calibration/`:
   - blind raw sheets per annotator: `sheets/raw_r<round>_<A1|A2>.csv`, with no qid, source kind, pool, split or judge output;
   - a separate `key.csv`;
   - `sheets/practice.csv`, rated jointly;
   - `adjudication_r<round>.csv`, built from two complete raw sheets; the team fills `<dim>_settled` only where flagged;
   - `final_labels_r<round>.csv`: the mean, or the settled value; built only when every flag is settled.
   - Flag = |difference| ≥ 2, or C given by only one annotator.
   - Raw sheets are never overwritten once they contain an entry.
   - Dimensions: exactly F, H, S, P, C and overall realism (prereg §4.3), plus the process fields `flag_unclear` and `notes` (Appendix C).

**Engineering details the spec leaves open:**
- Item IDs follow a seed-17 random presentation order (`C001…`, `P001…`); both annotators see the same order.
- Practice items: 20 FiQA synthetic items from U, excluding the 70 calibration draws, 10 P-std / 10 P-persona, seed 17.
- `cal-test` stays sealed for analysis: split membership is only in `key.csv`. The P4.4 κ/ρ code must use `require_unsealed("cal_test")` for cal-test numbers. All 200 are still annotated and adjudicated, as prereg §4.3–4.4 require.

**Run:**
- `ha calib status`.
- `ha calib export --practice` wrote the practice pack (20 items, 24 s, reproducible byte for byte).
- `ha calib export` is **blocked** (CAL-auth 30/60).

**Not done:** no labels, no weights, no bulk judging, no change to the prereg, PREREG.json, judge configs, P3 pools or the official sheet. Nothing committed.

### D-051 · 2026-09-28 · Phase 1: 9 scenario decisions approved and applied to the review artefact; gate still open · team (approvals); applied by agent (uncommitted)
**Approved by the team (instruction of 2026-09-28) and applied verbatim** to `data/humanside/scenario_sheet_chatgpt_draft.csv`. Each row has `review_status = approved_by_team (2026-09-28; D-051)` and `author = team-approved text …`, and its `agent_flag` notes that the earlier flag refers to the superseded draft.
- **Replacements (chunk and text):**
  - S017 → `incometax/1fa8efc9b1cd/002`
  - S025 → `incometax/6cebf3b156cc/003` (re-draw 1; `…/013` is marked "not used")
  - S136 → `incometax/306ec57bb9dd/004`
  - S147 → `incometax/d51b96a53e6a/010`
- **Rewrites (same chunk):** S005, S007, S067, S096, S137.
- **None of the 9 approved texts matched the current draft.** The drafts they supersede (kept in git at `4d5f2f0`):
  - S017: "…the bank account filled in on the form is an old one you have since closed…"
  - S025: "You recently joined a company and heard there are two different ways…"
  - S136: "…a relative or someone you trust to reply to it for you on the website…"
  - S147: "…shares … about a year and a half ago … short-term or long-term." The approved text asks instead whether the sale counts as a capital asset, which matches the chunk's question.
  - S005: "…through another contractor…"
  - S007: "…started a job … change jobs later…"
  - S067: "…as a lump sum…"
  - S096: "You run a workplace with employees…", employer role-play. The approved text is the employee's side.
  - S137: "A family member receiving a pension has died…"
- **Checker on the review artefact** (`ha data hiaq-check --sheet …draft.csv`, rules unchanged): **147 PASS / 2 WARN / 1 REJECT** (was 144/4/2). All 9 approved texts PASS; none was reworded to satisfy the checker.

**Not approved: still pending a human decision.**
- **S090:** the current shortened draft is NOT approved, because it drops the need to nominate someone **outside the family**. A text must preserve that need.
- **S091:** the current shortened draft is NOT approved, because it drops that the deceased was covered by **both military service/pension and the employee pension scheme**. A text must preserve that need.
  - The current draft is REJECTed ("military").
  - Information only: the earlier versions in `previous_scenario` keep both needs, but both are REJECTed by spec §6.3's rare-term rule (S090 "unmarried"; S091 "military", "serviceman").
  - `ha data hiaq-variants` refuses to build forms while any scenario is not PASS/WARN (`survey.py`). If the team keeps a faithful S091 that fails the rule, it must decide how to record that as a deviation from **spec §6.3** (a build-spec rule, not the prereg). The agent does not decide this.
- **S060, S066:** remain human-review WARNs (not rejected). WARN is accepted by the variant builder.
- **137 `keep_draft` rows:** not explicitly approved; still `pending_human_approval`.

**Also unchanged:**
- The official `data/hiaq/scenario_sheet.csv` (`28176bc2…`); nothing is copied until all approvals are complete.
- LID: `human_final_label` is empty for 100/100; no approved labels exist in the record.
- The ethics metadata placeholders.
- The guidelines' "Confirmed by the team on" line (no date supplied).

### D-052 · 2026-09-29 · Phase 1: LID audit finalised (100/100); scenario sheet finalised (150/150) and copied to the official sheet; spec §6.3 checker exception for S091 (S090 needs confirmation); variant builder blocked · team (all labels and approvals); applied by agent (uncommitted)
**LID (team decisions, applied verbatim):**
- `human_final_label` is filled for **100/100** rows in `data/humanside/langid_review_100.csv`:
  - the 10 flagged rows get the team's labels: L14 ROM_HI, L29 DEV_HI, L53 ROM_HI, **L54 HINGLISH**, **L55 HINGLISH**, L65 HINGLISH, L88 DEV_HI, L89 OTHER, L99 DEV_HI, L100 DEV_HI;
  - the other 90 take their provisional label, as the team instructed.
- **Only L54 and L55 differ from the provisional (audit-draft) label** (ROM_HI → HINGLISH). `current_label` and `provisional_label` are kept unchanged for audit; only the `human_final_label` column changed.
- Final counts: DEV_HI 21, EN 20, HINGLISH 22, OTHER 19, ROM_HI 18.
- `langid_adjudication_table.csv` was rebuilt with the existing `tools/build_lid_adjudication.py --recs data/humanside/langid_chatgpt_recommendations.csv`, which only carries over human labels. Only its `human_final_label (HUMAN-CONTROLLED)` column changed.
- The 30 original test cases' final labels equal their original test labels, so `tests/test_langid.py` needs no change and was not touched.
- The review stays blind: no classifier predictions were computed.

**Scenarios (team decisions): 150/150 approved.**
- The 137 `keep_draft` rows, S060 and S066 are approved **as drafted** (text unchanged).
- The 9 D-051 texts are unchanged.
- **S090 and S091 are replaced with the team's exact texts:**
  - S090 keeps "someone outside your family".
  - S091 keeps "served in the military and was also covered by an employee pension scheme".
- In the draft sheet, every `review_status` = `approved_by_team (…; D-051|D-052)`. Only S090 and S091 changed text in this step.

**Official sheet `data/hiaq/scenario_sheet.csv`** (not tracked by git: `data/*` is ignored):
- It keeps the same schema and row order, and the official conventions (`author = team`, as in `configs/survey.yaml`/D-038; `notes` empty). The ChatGPT-assisted and agent provenance is recorded in the draft artefact and in D-042/D-051/D-052, not in the official file.
- Filled: `scenario` for all 150 rows, plus the chunk fields of the 4 approved replacements (S017, S025, S136, S147) exactly as approved.
- **Verified before writing:**
  - 150 unique IDs and unique chunks;
  - every required field populated;
  - every row's source, url, heading_path and chunk_text equal `chunks.parquet`, and its topic equals `topic_of(url)`. `heading_path` is legitimately empty for 66 rows, as in the chunk store;
  - the replacement chunks are eligible (≥ 30 words, not professional);
  - non-replaced rows' chunk fields are unchanged;
  - sources: incometax 75, epfo 75.
- **sha256 `4968e202f87efd5734d456c300603e742beca93859a528f602735fb947d35596`** (was `28176bc2…`, empty scenarios).

**Official checker** (`ha data hiaq-check`, rules unchanged): **146 PASS / 2 WARN / 2 REJECT**.
- WARN: S060, S066 (65% word overlap), approved by the team.
- REJECT S091: "military".
- REJECT S090: "unmarried". The team's instruction anticipated only S091 (147/2/1). The approved S090 text uses "unmarried", which is also the chunk's own word ("Whether an unmarried person can nominate…").
- The review-artefact check is identical.

**Spec §6.3 exception (documented; the checker is not modified):**
- **S091:** human-approved. The source-supported need (the deceased was an ex-serviceman who was also an EPS member; family pension payable in addition to the military pension) requires the word "military". The REJECT is a conflict with the lexical rare-term rule, and the scenario is **retained unchanged by human decision**. This is a deviation from build spec §6.3 ("Rewrite rejected scenarios by hand"), not from the prereg: prereg §3.1 has no scenario-leakage rule.
- **S090:** the same kind of lexical conflict ("unmarried"). The text is team-approved, but the team has **not yet explicitly** extended the exception to S090. **Pending team confirmation**; the agent does not decide.

**Variant builder blocked:**
- `ha data hiaq-variants` fails with `RuntimeError: 2 scenarios are not PASS/WARN yet: rewrite and re-check` (`src/ha/data/survey.py:95–97`). Nothing was written; there is no `survey_variants.json`.
- **No existing mechanism records an approved exception.** The checker has no override, the variant builder accepts only PASS/WARN from `scenario_check.csv`, and no exceptions config exists.
- The agent did **not** edit `scenario_check.csv`, weaken either scenario, or change code. A mechanism must be chosen and approved by the team.

**Unchanged:**
- ethics metadata placeholders;
- the guidelines' "Confirmed by the team on" date (both blank, awaiting the team);
- prereg, PREREG.json, P3 pools, judge and neighbour code.

### D-053 · 2026-09-29 · Spec §6.3 exception mechanism: explicit fail-closed allowlist (S090, S091); survey variants built · team (S090 exception and mechanism approved); implemented by agent (uncommitted)
**Team decisions:**
- **S090:** the same documented §6.3 exception as S091 is **approved**. The text stays exactly as in D-052. The checker REJECTs it on "unmarried", which the source chunk itself uses ("Whether an unmarried person can nominate somebody from outside?").
- **S091:** the exception stands (D-052 is the justification).
- **Mechanism:** approved, provided it is the smallest explicit allowlist, the checker, rules and scenarios are unchanged, there is no global REJECT bypass, and it fails closed. This is a build-spec §6.3 implementation exception; **no prereg amendment**.

**Implementation:**
- **`configs/scenario_check_exceptions.yaml`** (new; not hash-locked by PREREG.json). One entry per allowlisted ID, currently **only S090 and S091**, each with:
  - `decision: D-052`;
  - the exact `checker_reason` (`rare document terms: unmarried` / `… military`);
  - the `scenario_sha256` of the approved text (S090 `7590d482…`, S091 `2fc8c7dd…`);
  - a `why`.
- **`src/ha/data/survey.py`:** `load_check_exceptions()` and `gate_check(sheet, check, exceptions)` replace the PASS/WARN-only test in `write_variants()`.
  - PASS/WARN pass as before.
  - A REJECT passes **only if** its ID is listed **and** the checker reason matches exactly **and** the sheet's scenario text hash matches **and** the entry names a decision.
  - Everything else blocks and nothing is written: an unlisted REJECT, EMPTY, an edited text, a new or extra rare term, a missing decision reference, or a check file that doesn't cover the sheet's IDs.
  - `survey_variants.json` gains a `check_exceptions` list (ID, decision, reason) as an audit record. The Apps Script `tools/make_forms.gs` reads only the keys it already knew, so it ignores this one.
- **Tests (`tests/test_survey.py`, +12):**
  - each of S090 and S091 alone lets the builder proceed, and both together do;
  - an unlisted REJECT (S005) blocks and writes no file;
  - WARN rows pass with an empty allowlist;
  - fail-closed on a changed reason, an edited text, an empty decision, or EMPTY status;
  - a stale check file blocks;
  - the shipped allowlist is exactly S090/S091 with D-052, the exact reasons and the hashes of the approved texts.

**Run:**
- `ha data hiaq-check` (official, unchanged rules): 146 PASS / 2 WARN (S060, S066) / 2 REJECT (S090, S091).
- `ha data hiaq-variants`: **25 variants written** to `data/hiaq/survey_variants.json`, sha256 `e125c5c29db41e19ee69a73b017427e47634e640f0ce6a5be6759645b3b71b1f`. A rebuild gives the same file.
  - 25 forms × 6 items; each of the 150 scenarios exactly once; every form mixes incometax and epfo; every item's text equals the official sheet.
  - `check_exceptions` = [S090, S091].
- **Real-data negative check (in memory):** making S005, S060 or S137 a REJECT blocks the gate, and an empty allowlist blocks the real sheet.
- The official sheet is unchanged (`4968e202…`).

**Not done:** no forms created, no pilot, no survey launch, no calibration, no Phase 5. The ethics metadata and the guidelines confirmation date remain blank (team).

### D-054 · 2026-09-29 · Google Forms builder made runnable and fail-closed; forms created closed; optional questions on page 2 · agent (team asked for form creation only; nothing created externally yet; uncommitted)
**Found while checking `tools/make_forms.gs` against `survey_variants.json`:**
- **Blocking bug:** the documented step "paste the JSON into `VARIANTS_JSON = '…'`" could not run. The file has 757 newlines and 4 apostrophes (the Appendix D intro, and S137 "child's"), so the JavaScript string breaks.
- **Priming risk:** on one page, the optional language question ("Hindi in English letters / Hindi in Devanagari …") was visible while people typed their answers. This conflicts with spec §6.3 ("Do not mention Hinglish") and the team's instruction to keep the forms neutral.
- `setRequireLogin` works only for Google Workspace accounts and may throw on a personal account.
- The script would have created forms that accept responses immediately.
- It gave no machine-readable record of what was created.

**Minimal changes (the design and all Appendix D text are unchanged):**
- **`tools/make_forms.gs`:**
  - `VARIANTS` is now a JavaScript object literal: the file content is pasted as is.
  - Before creating anything it checks 25 × 6 items, 150 distinct IDs, and `EXPECTED_VARIANTS_SHA256` = `0b3bb2e7…`, a sha256 of `JSON.stringify(variants)` that Python and Node reproduce identically. A truncated or edited paste stops with 0 forms.
  - A page break goes before the 2 optional questions.
  - `setRequireLogin` is wrapped in try/catch.
  - Forms are created with `setAcceptingResponses(false)`; opening them is the pilot/launch step.
  - It logs one `MANIFEST_JSON` line per form with the number, ID, edit URL, response URL, scenario IDs, item titles read back from the form, accepting state and creation time.
  - `FIRST_VARIANT` allows resuming after a timeout.
- **`src/ha/data/survey.py`:** `variants_fingerprint`, `parse_forms_log`, `check_forms_manifest` and `record_forms_manifest`.
- **CLI:** `ha data hiaq-forms-manifest`. It verifies all 25 forms against `survey_variants.json` (IDs in order, item text verbatim, Google URLs, not accepting responses, no duplicates or missing forms) and only then writes `data/hiaq/forms_manifest.json`.
- **Tests:** +8 in `tests/test_survey.py`:
  - the manifest is recorded when every form matches;
  - 5 kinds of mismatch are refused;
  - a missing form is refused;
  - the script keeps the object paste, closed forms and page order, has no study-revealing words, and its pinned fingerprint equals the current variants.

**Dry run** (Node, mock FormApp, the real JSON pasted exactly as a user would; no Google calls):
- 25 forms; each has 6 optional text items, then a page break, then the 2 optional questions; email not collected; not accepting responses.
- A truncated paste stops with "expected 25 variants, got 24" and an edited S091 stops with a sha256 mismatch, both with 0 forms created.
- The mock log passes `record_forms_manifest` (scratch folder).
- All 150 texts match the official sheet verbatim: S090 is in form 05, S091 in form 11.

**Participant-facing content:** unchanged Appendix D text. The title, the intro (consent sentence included), "What would you type? (short text)" per scenario, and the 2 optional spec questions (languages usually typed; phone or computer). No email, name, phone, sign-in or other demographics.

**Not done:** no forms created (that runs in the team's Google account), no redirect-page edit, no pilot, no launch. Ethics metadata is still blank.

### D-055 · 2026-09-29 · Ethics metadata not recorded (team-lead decision); pilot and launch no longer wait on it; pilot data kept separate · team lead (decision); recorded by agent (uncommitted)
**Team-lead decision (verbatim intent):** the faculty-advisor approval was informal ("not that formal of thing"). Skip recording the approver, role, date, scope and conditions, and proceed with the survey.
- **What is on record:** approval was reported as obtained by the team (D-041; `humanalign_phase1_review.md` at `d163086`: "Faculty/advisor ethics approval: DONE").
- **Placeholders:** the five fields are marked **"not recorded: waived (D-055)"**. **No values were filled or inferred.**
- The survey protocol §0 gate ("record it in `docs/decisions.md`; do not launch without it") is satisfied **by this entry**, in the form the team chose: informal approval reported, with details not recorded.
- **Consequence for the paper (fact, not a gate):** the ethics statement can say only that approval was obtained informally from the faculty advisor, as reported by the team. It must not state an approver, a date, a scope or conditions. Whether the approval covers the rating study (E8) is **not documented**.
- **Pilot data kept separate (agent safeguard, reversible):** `ha data hiaq-ingest --pilot` reads `data/hiaq/pilot_responses/` and writes `pilot_queries.parquet`, **never** `survey_queries.parquet`. Before, any ingest wrote the dataset file. Whether pilot answers join the dataset is left to the team; the default is that they don't. Pilot responses stay inside the forms and must be deleted from them before launch (protocol §3).

### D-056 · 2026-09-29 (decided) / 2026-10-10 (implemented) · HiAQ answers collected in a shared collaborative sheet, kept as a SEPARATE labelled set · team lead (decision); implemented by agent (uncommitted)
**Team decision ("option 3"):**
- Collecting from 100 separate form respondents is impractical. Real people (not the team) instead type their answers into one centralised, collaborative sheet.
- The result is kept as a **separate, clearly labelled set** (`source = collab`): outside HiAQ's authentic survey data, the splits and the main test, and disclosed as such.
- The team clarified that the answers are written by people, not by the team.

**Implementation:**
- **Template:** `ha data hiaq-collab-sheet` writes `data/hiaq/collab/collab_sheet_template.csv` (git-ignored).
  - 100 blocks × 6 = **600 rows**. Each block is one form's 6 situations (B001–B025 = forms 01–25, round 1; B026–B050 round 2; …), so each of the 150 situations appears 4 times.
  - Columns: `block, round, form, item, scenario_id, situation, answer, contributor_code, languages, device`.
  - The situation text equals the official sheet. It refuses to overwrite an existing template.
- **Ingest:** `ha data hiaq-collab-ingest` checks the filled CSV and writes only `data/hiaq/collab/collab_queries.parquet` (`source = collab`; qid `collab-<block>-<item>`).
  - Cleaning is the same as the survey: PII stripped, whitespace only, typos kept, LID and CMI attached.
  - Contributor codes are stored only as a hash.
  - **It refuses the sheet if:** the fixed columns or situation text were edited; an answer has no contributor code; a block has two contributors; or one contributor answered the same situation twice (two blocks of one form).
  - It **never** writes `survey_queries.parquet` or anything under `data/processed/`.
- **Instructions to send:** `data/hiaq/collab/HOW_TO_FILL.txt`. The contributor text mirrors Appendix D (consent sentence included, no language cue); the team rules follow it.
- **Tests:** +6 in `tests/test_survey.py`.

**Known limitation (disclose):** in a shared sheet, contributors can see others' answers, which weakens independence. Mitigation: per-person filter views or copies.

**Open consequences (team decisions, not taken here; project report §9.2):**
1. With no form-survey answers, HiAQ's authentic set is empty. Prereg §8 fallback: H2/H3 on D2 alone and D4 as a pilot, recorded as an amendment **before** `heldout_A` is unsealed.
2. The 30 HiAQ CAL-auth items for calibration (prereg §4) have no source unless some form responses are collected, or the team formally decides otherwise.

### D-057 · 2026-10-10 · HiAQ data roles: collaborative sheet exploratory only; Google Forms for the initial authentic tranche; plan for the §8 fallback; amendment A1 drafted (not recorded) · team (decisions A–D); recorded by agent (uncommitted)
**A. Collaborative sheet (D-056): exploratory only.**
- Its answers never enter HiAQ's R, held-out A, `CAL-auth`, strata, neighbours, qrels decisions or any pre-registered test (H1–H4).
- They do not count toward prereg §8's 300-query threshold.
- **The sheet is not to be distributed yet.** Open quality issues are in the 2026-10-10 audit: visibility of others' answers, Google Sheets editor identities, the language cue in the instructions, no instruction against AI or translation tools, and no timestamps.

**B. Initial authentic tranche via the existing Google Forms.**
- **Target:** 25 completed forms (up to 150 scenario answers). The early tranche supplies the 30 HiAQ `CAL-auth` items (prereg §4.1; H-07).
- **Respondent independence is preserved: one form per respondent** (spec §6.3: "each respondent gets 6 random scenarios").
- **Protocol inspection for a multi-form deviation:**
  - The prereg is silent.
  - The spec requires one form per respondent.
  - The forms cannot enforce it (`setLimitOneResponsePerUser(false)`, because limiting would need sign-in, which collects identities).
  - Ingest assigns one respondent ID per submission (`survey.py`: hash of variant, row and timestamp), so a person who filled several forms would look like several independent people, and their answers could fall in both R and A.
- **Therefore no deviation is adopted.** Allowing several forms per person would need a new dated decision.
- **Open operational choice (team):** for 25 completions to cover all 25 forms once, hand each person one direct link through a seeded random assignment, rather than the random redirect (D-014), which can leave forms unfilled.

**C. Fallback planning.**
- If HiAQ has **fewer than 300 eligible authentic queries after the §3.1 rules**, prereg §8 applies: H2 and H3 on D2 alone, D4 reported as a pilot, recorded as an amendment before `heldout_A` is unsealed.
- **The fallback is not claimed to have triggered:** no eligible count exists yet (no form responses collected).

**D. Amendment A1: DRAFT only.** `docs/prereg_amendment_A1_DRAFT.md` covers:
- the condition and timing;
- H2 and H3 on D2 alone;
- H1 (wording kept; D4 counts as not met if unevaluable);
- H4 (descriptive on the pilot);
- D4 qrels for pilot analyses (source chunk grade 2 plus the pre-registered 30% pooled fallback; source-only as sensitivity);
- calibration (§4 unchanged: 30 authentic from the Forms tranche plus 70 synthetic);
- exclusion of collaborative-sheet answers.

**Not recorded; `docs/preregistration.md` and `results/PREREG.json` are unchanged.**

### D-058 · 2026-10-10 · The 150 "team" queries are an AI-generated external synthetic set (`X-team150`); no authentic HiAQ data; amendment A2 drafted · team lead (decision); implemented by agent (uncommitted)
**Team decision:**
- The 150 queries in `human_written_version5*.csv` are a **researched, AI-generated synthetic dataset**. They will **not** be represented as human-authored or used as authentic observations.
- The team **cannot collect independent human responses**.

**Implementation:**
- `tools/import_external_synthetic.py` wrote `data/synthetic_external/hiaq/X-team150/{queries.parquet, stats.json}` (git-ignored).
  - The input is the `source=synthetic` file `human_written_version5.csv`, sha256 `8baea782…`. Its text is identical to the later "human"-labelled copy.
  - 150 rows; text kept exactly; P3-like columns plus `scenario_id`, `service`, `language_mix_given` and `provenance`; `source_pid` = the scenario's official chunk.
  - **It sits outside `data/synthetic/`:** `load_U('hiaq')` still returns only P-std and P-persona (1,231).
- **Allowed use:** exploratory, labelled analyses only. It is **never** used for authentic HiAQ data, `CAL-auth`, calibration synthetic items, anchors, R/A, neighbours, strata or any pre-registered pool.

**Consequences:**
- **No authentic HiAQ data.**
- Once the team declares HiAQ collection closed (eligible count 0), the §8 fallback condition is met. The amendment A1 draft needs the revisions listed in A2.
- **Calibration cannot run as pre-registered:** there are no HiAQ `CAL-auth` items, and HiAQ synthetic items would have no R for neighbours. Proposed fix: `docs/prereg_amendment_A2_DRAFT.md` (FiQA-only, 100 items, 60/40). **Draft only; not recorded.**
- **Blocked without new human data:** H4; D4 in H1, H2 and H3; D4 qrels, R, neighbours and strata; the D4 P-fewshot pool.

### D-059 · 2026-10-10 · Reduced-scope study: HiAQ authentic collection closed at zero; A2 (FiQA-only calibration, 100 items) implemented behind a recording guard; A1 revised (rev 2) · team lead (decisions); implemented by agent (uncommitted)
**Team decisions:**
1. **Authentic HiAQ collection is closed at zero** for this study; no further recruiting. The eligible HiAQ count is therefore 0.
2. **A2 calibration design:** 30 authentic FiQA + 70 synthetic FiQA (100 items).
3. **A1 is revised for consistency with A2** and with the actual scope, including the H1 Holm correction: the frozen m = 9 family is kept, with unevaluable D4 tests as non-rejections.
4. **`X-team150` stays separately labelled external synthetic data,** excluded from confirmatory data, calibration, anchors and held-out evaluation.

**Drafts (not recorded; the prereg and PREREG.json are unchanged):** `docs/prereg_amendment_A1_DRAFT.md` (rev 2) and `docs/prereg_amendment_A2_DRAFT.md` (rev 2, with a rule-by-rule check against prereg-v1 §4 and its consequences).

**Implementation (uncommitted):**
- `configs/calibration.yaml` implements A2 and names it (`amendment: A2`).
- `ha.judge.calibrate.build_items` **refuses until "### A2 · …" is recorded under `## Amendments`**. Verified on real data: `ha calib export` → BLOCKED.
- The FiQA calibration items and their dev/test assignment are identical to what prereg-v1 would produce (test).
- Guidelines v1: counts 100, FiQA domain only; rubric unchanged.
- D-050's human display rule now removes "all `CAL-auth`" (30 under A2), not "all 60".

**Open for the team:**
- (a) the v1/v2 labelling question under prereg line 3 (see A2 notes);
- (b) the guidelines confirmation (below);
- (c) approval and recording of A1 rev 2 and A2 by both members;
- (d) the commit.

**Guidelines confirmation required (prereg §4.3: "finalised before annotation starts"):**
- Both annotators read `docs/annotation_guidelines.md` v1 (with its A2 counts) and confirm it is final.
- The date and both members' initials go on its line "Confirmed by the team on: …", and a dated D-entry is added.
- This must happen before the practice round, after A2's wording is final.

### D-060 · 2026-10-10 · Amendments A1 and A2 recorded; analysis released as prereg-v2 = prereg-v1 + A1 + A2 · project lead (Chinmai Shetti)
**Versioning (project lead's decision, 2026-10-10):**
- Prereg line 3 and D-030 require an explicit v2 for a methodological change, and A2 is one (it changes the locked §4 calibration procedure).
- The analysis is released as **prereg-v2 = prereg-v1 + A1 + A2**.
- prereg-v1's primary analysis is reported as **not executable**, because its calibration set cannot be constructed.
- A1 alone is the pre-registered §8 contingency.

**Approval and recording rules (checked 2026-10-10):**
- **Amendments must be dated and appended below `## Amendments`** (prereg line 3; D-016; spec line 69). **No binding rule names who must approve an amendment.** The "both members" wording in D-059 and the earlier drafts was the agent's drafting, not a project rule.
- **The project lead's approval is the project decision.**
- **Joint signatures are required later,** at the freeze: both team members sign in `decisions.md` before `heldout_A` is unsealed (prereg §6).
- **Faculty approval** applies only to the survey launch (spec §15.3; PLAN H-04). It does not apply here, and the survey is closed (D-059).

**Recorded:**
- A1 and A2 are appended to `docs/preregistration.md` on **2026-10-10** by **Chinmai Shetti (project lead)**. The name was taken from the lead's configured git identity and **confirmed by the lead on 2026-10-14**.
- **Approved text** sha256 `6194857e34e1eb275373db828aa51f31c2075f956fe967cbbbb9259a7aade242`. The **recorded text** is the same with only the date and name placeholders filled; sha256 `8b91f44d6833572f93f1c93a77d42813f411baa792c97202a72ffa8fc155ac0b`. Reversing the placeholders reproduces the approved hash.
- **Byte-level append-only:** the previous file is an exact prefix. The frozen-body hash is unchanged (`487199ff…`, `results/PREREG.json`) and `check_frozen()` passes. As expected, the whole-file hash now differs from PREREG.json's tag-time artefact record.
- **Recorded before any calibration annotation or result:** no calibration pack exists (`data/calibration/key.csv` absent), and no annotation has started.

**Supersedes or corrects:**
- D-027, for the confirmatory test (H2/H3 on D2 alone);
- D-059's notes: "v1 as amended" (now v2) and "approval … by both members" (not required for amendments).

**Consequence:** the A2 guard in `ha.judge.calibrate` now permits building calibration items.

**Still required before `ha calib export`:** the annotation-guidelines confirmation by both annotators (prereg §4.3; date and initials in `docs/annotation_guidelines.md` plus a D-entry), and the reviewed commit.

### D-061 · 2026-10-14 · Annotation guidelines v1 finalised · project lead (Chinmai Shetti)
- **Rule:** prereg §4.3 requires `docs/annotation_guidelines.md` to be finalised before annotation starts. It names no signatories, so the lead finalised v1 (100 FiQA items, A2; rubric unchanged) on 2026-10-14. The earlier "both annotators' initials" step was the agent's process suggestion, not a project rule.
- **Still binding (§4.3–§4.4):** the annotation itself needs **two annotators**. They do the 20 practice items together, then each rates all 100 items independently; κ is computed between them.
- **Next:** `ha calib export` builds the 100-item A2 pack.

### D-062 · 2026-10-14 · No human calibration; amendment A3 recorded; analysis released as prereg-v3 · project lead (Chinmai Shetti)
**Decision (project lead, 2026-10-14):** the study proceeds **without the human calibration annotation** (H-08) and without two annotators.

**Recorded as amendment A3** in `docs/preregistration.md` on 2026-10-14:
- byte-level append-only; the previous file is an exact prefix;
- A3 text sha256 `08435f6a43b2318d34d713d7d02cd4d41d0bee951e78f4ca0ecbae79d8271909`;
- frozen-body hash unchanged (`487199ff…`, `results/PREREG.json`); `check_frozen()` passes.

**What A3 fixes:**
- **Judge:** prompt v0 (spec B.4 verbatim) is final. No v1–v4, no anchors. Model, decoding, schema, permutations and neighbours unchanged.
- **Weights:** equal weights on (H, S, P, C*) for M1 and M1-iso (`ha.judge.aggregate.EQUAL_WEIGHTS`, the spec §9.3 ablation), recorded in `FROZEN.json` in place of NNLS weights. `configs/judge.yaml` is **not** edited (it was hash-recorded at the `prereg-v1` tag; its `weights: null` stays); the selection code takes the weights from A3.
- **Unchanged:** gate F ≥ 4 and all of §5.2.
- **Not executed:** κ, the human reference, ρ-based version selection, NNLS weights and CV ρ, every `cal-test` statistic. The A2 pack exported on 2026-10-14 stays unannotated; `cal-test` is never unsealed.
- **Freeze sign-off (§6):** the project lead signs before `heldout_A` is unsealed.
- **Versioning:** **prereg-v3 = prereg-v1 + A1 + A2 + A3**; prereg-v2's primary analysis is not executable (no calibration annotation).

**Disclosure for the paper:** the judge was not human-calibrated, and the paper makes no claim about judge–human agreement.

**Timing:** recorded before any bulk judging output, selection or held-out result. Only the P4.2 engineering pilot (12 FiQA queries, D-049) exists.

**Supersedes:** D-061's "still binding: two annotators" and D-060's "both team members sign at the freeze".

**Consequence:** `ha judge <ds>` (P5.1, `src/ha/judge/bulk.py`) refuses to run until A3 is recorded; it is now permitted.

### D-063 · 2026-10-15 · Selection details left open by prereg §5.2, fixed before any ranking result · agent, under the lead's instruction to proceed
prereg §5.2 does not specify these details. They are fixed here, **before any retrieval run on a selected set and before any τ**, and go into `FROZEN.json` (§6) with the selection code's git sha. The lead may veto any of them before the freeze.
1. **Tie-break.** Ties in the greedy score order are broken by a seeded uniform random key per U query (numpy SeedSequence `[2, split seed]`), the same key for M1 and M1-iso.
   - Why: about 31% of FiQA gate passes share the maximum equal-weight score (20).
2. **Quotas.** n_k = N·π_k, rounded by largest remainder (ties go to the lower stratum index), so the quotas sum to exactly N. A stratum's shortfall is re-allocated in proportion to π, iterated until N is reached or the candidates run out.
3. **Dedupe** (cosine > 0.90) applies to the judge-based selections (M1, M1-iso) only, because §5.2 lists it under "Selection". B0 and B1 are plain random draws (U is already deduped at 0.95 within each pool, P3).
4. **Seeds.**
   - B0 draw d uses `[0, split seed, d]`; B1 uses `[1, split seed, d]`; d = 0…19.
   - Strata: scikit-learn KMeans on the R_s bge-m3 embeddings, random_state 0, n_init 10. U queries join the nearest centroid.
5. **N's |A_s|** is read from the split labels only, never A's text; D2's A stays sealed.

**FiQA (dev) selection, 2026-10-15** (`ha select fiqa`):
- N = 388 = |A| in every split (binding: A); M1 passes 954–978, M1-iso 1,100; no shortfall.
- Ties at the cutoff score: in the worst split, 478 M1-iso candidates for 52 places.

### D-064 · 2026-10-15 · H1 operational details, fixed before any H1 result · agent, under the lead's instruction to proceed
prereg §2 names H1's measures and tests but not these details. They are fixed here, before any H1 statistic was computed (`src/ha/eval/h1.py`):
1. **"Raw synthetic queries (B0)"** = the population B0 draws from: U minus the seed's near-duplicates. A single 388-query draw would add sampling noise.
2. **Length** = whitespace words per query.
3. **Gold-passage overlap** = the share of a query's content-word types (lowercase alphanumeric tokens, scikit-learn English stopwords removed) that occur in its gold passage.
   - Synthetic queries use their source passage.
   - Authentic queries use the best-matching passage among their highest-grade qrels.
4. **Topic** = each query's R_s stratum from the §5.2 k-means (the same model as selection). The test is JS divergence (base 2) with a 1,000-permutation null of the pooled labels.
5. **W1** gets a 1,000-iteration bootstrap CI. All randomness uses numpy seed 20260927.
6. **Seeds to one p per dataset × measure:** the largest of the 5 per-seed p-values. A measure counts for a dataset only if it is significant in every seed (an intersection-union rule; conservative).
7. **Holm** over 3 datasets × 3 measures (m = 9). D4's three tests are p = 1 (A1). Until D2's A is unsealed, a D1-only run treats D2 as p = 1 too; that run is a development preview, not the H1 decision.
8. **τ_b(B0, A) < 0.8** is read from the evaluation bootstrap: the seed-averaged point estimate with its 95% CI (`tau_seed_mean`).

### D-065 · 2026-10-16 · HiAQ collected responses, batch "group a", accepted on the lead's statement · project lead (Chinmai Shetti)
**Decision (project lead, 2026-10-16):** the 150 queries in `15-10-26 collected response group a.csv` were written by people for the 150 scenarios. They are accepted as **collected responses**. The lead directed that no further provenance verification be done.

**Recorded facts:**
- **Raw files:** kept byte-identical in `data/hiaq/incoming/2026-10-16_group_a/raw/`.
  - CSV sha256 `c0a6bb4d…`; xlsx `d4e9af23…`.
  - The xlsx was created 2026-10-16 04:57 UTC in Excel; sheet "collected_human".
- **Imported** by `tools/import_hiaq_collected.py` to `data/hiaq/collected/2026-10-16_group_a.parquet`:
  - 150 rows, 0 empty, 0 PII;
  - LID: EN 86, HINGLISH 57, DEV_HI 4, ROM_HI 3. The 9 romanised-Kannada rows are classed EN, because the LID has no Kannada class.
- **Situation wording:**
  - 141/150 match the official sheet.
  - **S005, S007, S017, S025, S067, S096, S136, S137, S147 use the same non-official wording as the 2026-10-10 workbook.** They are flagged.
  - S017, S025, S136 and S147 answer scenarios whose chunk was replaced (D-051), so they get **no chunk**: 146 rows are scoreable.
- **Automated similarity check**, run before the lead's instruction: 106/150 rows have similarity ≥ 0.6 to the AI-labelled set of 2026-10-10 for the same scenario; none ≥ 0.85; none identical.

**Role:**
- 150 is below prereg §8's 300, so **A1 stands**. H2/H3 stay on D2 alone; H1's D4 tests stay at p = 1; H4 stays not estimable. This data does not change any confirmatory or secondary test.
- HiAQ becomes a **pilot with collected queries**, used only in analyses labelled exploratory. The paper states their provenance as the team's statement, not as an independently verified survey.
- Not used for calibration (A3: none), prompts, weights, selection settings or the freeze.

### D-066 · 2026-10-16 · Freeze signed (prereg §6 as amended by A3) · Chinmai Shetti
- `results/FROZEN.json` record sha256 `5b1ab03978b6251fef32cbb13a57b8632e67dbed1cecce31b753937ec09047c3`; analysis code `12c81111f5cccfa0edf2438442ffedc25eb6c932`.
- `heldout_A` may now be unsealed. Any later change to a frozen item is a separately reported -v2.

### D-067 · 2026-10-17 · Exploratory HiAQ pilot: design · project lead (Chinmai Shetti), agent proposal
The lead chose to run the HiAQ pilot on the collected batch "group a" (D-065). **Everything here is exploratory** (prereg §2, §7.7; A1). It feeds no confirmatory or secondary decision. The confirmatory results (D-066, PAPER_LOG §6.1–6.2) were produced before it and are unchanged.
- **Data** (`tools/build_hiaq_pilot.py`, `configs/datasets/hiaq.yaml`, role `exploratory`):
  - corpus: the 1,572 chunks, with chunk text only, the same text the pools were generated from;
  - queries: the 146 collected queries that map to an official chunk;
  - qrels: the scenario chunk, grade 2. No pooling, so the qrels are sparse; disclosed.
- **Splits:** R 40% / A 60% (57 / 89 per seed), stratified by language × topic cluster (k = 4), seeds 0–4. No CAL-auth (no calibration under A3).
- **Pipeline: the frozen settings unchanged.**
  - judge v0 with equal weights and the gate;
  - neighbours from R_s (5 per query);
  - selection per D-063;
  - C1–C7;
  - the §7 bootstrap, with H2/H3-style contrasts reported descriptively only.
- **Code changes after the freeze**, for this pilot only:
  - the `hiaq` dataset config;
  - an evaluation label that names the dataset's role (`exploratory`), where it previously defaulted to `heldout`.
  - The frozen analysis code (`12c8111`) is otherwise unchanged.

### D-068 · 2026-10-17 · Exploratory "-v2" selectors, specified before any run · project lead (Chinmai Shetti), agent design
Motivated by the overlap diagnosis (PAPER_LOG §6.4), two judge-free selectors are **fully specified here before they were run on any dataset, FiQA included**. They are run unchanged on D1, D2 and D4, and every result is reported. **Exploratory:** D2's A has already been seen (D-066), so nothing here is confirmatory; the main results stay as reported.
- **OVL, overlap-matched random:**
  - 5 equal-frequency bands of gold-passage overlap, computed on R_s only, never A;
  - eligible U queries fill the bands in R_s's proportions (largest remainder; a band's shortfall goes to the others);
  - random within a band; 20 draws, SeedSequence `[3, split seed, draw]`; no dedupe, like B0/B1.
- **NN, nearest-to-real:**
  - score = `nn_mean5_cos`, the mean cosine to the 5 R_s neighbours from P4.1;
  - the M1 procedure with this score: k-means strata of R_s, quotas, 0.90 dedupe, tie key `[4, split seed]`.
- **Shared with the frozen sets:** N, eligibility (U minus near-duplicates) and the evaluation.
  - Evaluation: §7 procedure, B = 1,000, seed 20260927.
  - Comparators: the frozen B0, B1 and M1 sets.
- **Reported:** τ_b seed means with CIs, and Δ for OVL and NN vs B0, B1 and M1, with one-sided p. There is no decision rule.
- **Code:** `src/ha/select/variants.py`, `tools/run_v2.py`, `tests/test_variants.py`.

### D-069 · 2026-10-17 · Replication R1 pre-registered: overlap-matched selection on two fresh datasets · project lead (Chinmai Shetti), agent design
- **Protocol:** `docs/replication_prereg.md`, committed **before any data of D5 or D6 was downloaded**.
  - Datasets: D5 CQADupStack-Webmasters and D6 NFCorpus (BEIR test splits).
  - The frozen pipeline is unchanged: splits, pools, near-duplicate exclusion, D-063/D-068 selection, C1–C7.
  - R1: OVL > B0 and OVL > B1, pooled; paired bootstrap B = 1,000, seed 20260927; intersection-union test at α = 0.05.
- **The lead approved the dataset downloads** by choosing this step (2026-10-17). The two datasets total about 16 MB, on D:.
- A is not used before the replication freeze (`results/FROZEN_replication.json`, protocol §6). The outcome is reported whatever it is.

### D-070 · 2026-10-17 · R1 protocol clarification, recorded before any D5/D6 download · agent
- `docs/replication_prereg.md` §6 says A is not used before the replication freeze. **One unavoidable read is clarified here:** building the splits (§4.1) reads every authentic query's text to assign its language class and topic cluster.
  - This is the frozen prereg §3.2 procedure, which D2 also went through before its freeze.
- It is the only touch of A before the freeze. No A qrels, retrieval run or metric involves A until §6 step 4. N uses the A labels only.
- **Code** (written before the download): `src/ha/select/replication.py`, `configs/datasets/{cqa_webmasters,nfcorpus}.yaml`, `tests/test_replication.py`.

### D-071 · 2026-10-17 · Three exploratory analyses, specified before they are run · project lead (Chinmai Shetti), agent design
All three reuse existing runs, sets and judgements; no new generation or judging. **Exploratory:** they inform the discussion only, and no confirmatory result changes. Every result is reported.

**Shared settings:**
- bootstrap as §7: B = 1,000, seed 20260927, A resampled once per seed, sets independently, draws averaged at the τ level;
- comparisons reported as Δ̄ with 95% percentile CI and one-sided p, per dataset (no pooling).

**E-density (relevance density).** Datasets D5 and D6, with the R1 sets (OVL, B0, B1).
- Authentic per-query nDCG@10 is recomputed with **sparsified qrels**: one passage per authentic query, the highest grade, ties broken by the smallest sha256 of the passage id.
- Reported: τ_b per method and Δ̄(OVL − B0), Δ̄(OVL − B1).
- **Prediction:** if relevance density explains D6's null, D6's OVL − B0 turns positive.

**E-metrics (metric robustness).** τ_b is recomputed with three other metrics: MRR@10, Recall@100 and Hit@5 (ranx).
- Sets: D1, D2, D4 with M1, M1-iso, B0, B1 (frozen) plus OVL and NN (D-068); D5, D6 with OVL, B0, B1 (R1).
- Comparisons: M1 − B0, M1 − B1, OVL − B0, OVL − B1 where present.

**E-dims (judge-dimension ablation).** Datasets D1, D2, D4.
- Four selectors, SEL-F, SEL-H, SEL-S and SEL-P, each scoring with one median dimension of the M1 judgement.
- They follow the M1 procedure otherwise: gate (ok, F ≥ 4, not near-duplicate), k-means strata of R_s, largest-remainder quotas, dedupe 0.90, tie key `[2, split seed]`, frozen N.
- Comparisons vs B0 and vs M1.

### D-072 · 2026-10-17 · Replication R2 pre-registered: D7 SciFact, D8 CQADupStack-Mathematica, D9 CQADupStack-Programmers · project lead (Chinmai Shetti), agent design
- **Protocol:** `docs/replication2_prereg.md`, committed **before any download**. It is the R1 protocol unchanged except for the datasets.
- **Counts:** expected counts were checked from Hugging Face metadata (datasets-server sizes) on 2026-10-17. D8 and D9 are sampled to 700 queries by the existing builder rule.
- **Downloads approved by the lead** (2026-10-17, choice of item 9), about 60 MB in total, on D:.
