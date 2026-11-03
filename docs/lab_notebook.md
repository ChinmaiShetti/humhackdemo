# Lab notebook

One entry per work session: what was run (with `run_id`s), what was learned, and what comes next.

---

## 2026-09-27 · Session 1 · Phase 0 setup

**Done**
- Read the spec in full; wrote the execution plan (`docs/PLAN.md`) and got the team's decisions on I-1, I-2, I-3, I-7, I-8, I-11, I-16 and I-18. These are recorded as D-001 to D-008, and the agent defaults as D-009 to D-018.
- Moved the project to `D:\humanalign-rag` (D-008). `git init` on branch `main`. The C: folder now holds only the original spec and a pointer file.
- Python 3.11.15 via uv (in `D:\ai-envs\uv-python`), venv at `.venv`. PyTorch cu128 install started.
- Core infrastructure (P0.3):
  - `paths.py`, `config.py`, `seal.py` (scoped seals, D-009), `registry.py` (auto-provenance, D-015), `cache.py` (SQLite key-value store with Kaggle merge), `utils.py`;
  - `llm.py` (Ollama wrapper with `unload_all` for the VRAM rule), `models.py` (pull/verify to `env/models.json`), `doctor.py`, `kaggle.py` (`git archive` zip, D-006);
  - a CLI with every §20.1 command (stubs point to their WPs).
- Tests: seal, registry, cache/config and CLI.
- Pre-registration drafted (`docs/preregistration.md`, DRAFT). H2 amended per D-001; the H2 decision rules are operationalised in D-017, which awaits team confirmation.
- Novelty check #1 (`docs/novelty_check.md`): no overlap with C1–C4.
- `PYTHONUTF8=1` set as a user environment variable (D-018).

**Learned**
- Rahmani et al. (SIGIR '24): with sparse synthetic qrels (only the source passage relevant), system-ranking τ dropped to 0.157. All our synthetic sets use sparse `source_pid` qrels, so absolute τ_b may be capped and the differences between methods compressed. **Proposal for the P0 gate:** make E11 (pooled qrels for M1 queries) planned rather than optional, and report the sparse-qrel ceiling.

**Later the same session**
- C: slowed down during the torch install. Temp files now go to D: (D-019). Dependencies installed cleanly: torch 2.11 + cu128, transformers 5.17, FlagEmbedding 1.4.2, sentence-transformers 6.1, ranx, faiss-cpu, fasttext-wheel; **no ONNX Runtime**.
- **Scope correction (D-020 to D-026):** prototype-first; the locked stack is qwen3:4b, Ollama bge-m3 and bge-reranker-v2-m3; no model downloads. PLAN rewritten as v2 (CORE / OPTIONAL / FULL-SCALE). Pre-registration redrafted. `ha models pull` removed.
- Model inventory (read-only checks):
  - `qwen3:4b` and `bge-m3` are present in Ollama.
  - The reranker's weights are complete in `D:\ai-models\hf-cache` (2,271,071,852 bytes), but its snapshot files are 0-byte placeholders. It will be repaired by hard links (D-025) after the go-ahead.
  - There is no HF copy of BGE-M3.
- Dataset sizes from HF metadata: `BeIR/fiqa` 28 MB, `BeIR/fiqa-qrels` 0.3 MB, `mteb/cqadupstack-android` 14.8 MB.

**Team answers:** H2 is a single pooled agreement test (D-027); HiAQ is core; P0.2–P0.4 and the D1/D2 download are approved.

**P0.2–P0.4 (same day)**
- The reranker was repaired by hard links (0 bytes, no download).
- `ha models verify`: all three OK.
- `ha data build fiqa` and `ha data build cqa_android`: exact BEIR counts; 40.8 MB.
- Benchmark (`reports/phase_0_benchmark.json`): judge 1.73 s per call (JSON 20/20); reranker 74 query–passage pairs/s and 393 query–query pairs/s; bge-m3 via Ollama slow (6.5–16 passages/s).
- **Blocking finding:** the local `qwen3:4b` is the thinking-only 2507 build. Generation leaks reasoning under every no-thinking method (`tools/diag_gen_embed.py`). The local `qwen3:8b` (hybrid) works with `think=False`. Decision D-028 is open with the team.
- `ha doctor` OK; 26 tests pass; ruff clean; pip freeze written; first commit.

**D-028 resolved (same day)**
- The team chose option 2. The shorthand `qwen3:4b-instruct-2507` is not an Ollama tag. The Ollama library lists `qwen3:4b-instruct-2507-q4_K_M` (ID `0edcdef34593`), and confirms that `qwen3:4b` = `qwen3:4b-thinking-2507-q4_K_M`. Pulled it: 2.33 GB on D:, 0 on C:.
- The digest is pinned in `configs/{models,generation,judge}.yaml`; `ha.llm.require_digest` enforces it, and a test checks that the configs agree.
- `tools/bench_llm.py`:
  - generation: 20/20 clean queries, 0.54 s median;
  - judge: 20/20 JSON, 1.60 s median, F 4.5 matched vs 1.0 mismatched, stable across permutations;
  - v0 prompt weaknesses (H leniency toward exam-style queries, `reason`/`keep` mismatch, halo effect) noted for P4 and **not** tuned.
- Old benchmark renamed to `phase_0_benchmark_qwen3-4b-thinking.json`.

**Next:** the team reviews the pre-registration. After approval: freeze and tag `prereg-v1`, then P1 (splits, LID, the HiAQ track).

**Superseded "Next" (kept for the record)**
- P0.2: repair and verify the models (no downloads).
- P0.3: mini throughput benchmark.
- P0.4: tests, freeze and the first commit.
- P0.5: pre-registration approval.

---

## 2026-09-27 · Session 1 (continued) · prereg-v1 frozen; Phase 1 engineering

**Done**
- **Pre-registration:** revision 3 locks the team's 10 criteria (D-029). All six choices approved; **frozen and tagged `prereg-v1`** (commit `b1c18e0`). `results/PREREG.json` records the commit and hashes, and `tests/test_prereg_frozen.py` guards the frozen body.
- **HiAQ source vetting** on the live policies (D-031). Two search summaries were stale: Parivahan's live policy needs permission, and EPFO's was confirmed live. Included: Income Tax and EPFO. UIDAI parked (JavaScript-only).
- **Polite scraper**, seeded from the Income Tax sitemap (288 English help URLs): 301 + 6 pages. The safety stop fired on page-specific 403s (D-037) and was not resumed. A logging bug in the stop path is fixed.
- **Chunker:** 1,572 chunks, after fixing a false-FAQ bug in which navigation links ending in "?" became questions.
- **Model-free LID** (D-032); 30 provisional test labels pass; 0 false code-mix labels on 1,347 English queries.
- **Splits for D1 and D2** (D-034).
- **Scenario sheet:** 150 rows, individual-user topics only (D-035). A regex bug that would have excluded "where" and "confirm" was caught by a test. Leakage WARN signal (D-036).
- **Survey tooling:** variants, Apps Script, redirect page, ingest with PII stripping. Protocol in `docs/survey_protocol.md`.

**Learned**
- The spec's leakage check misses document-echo scenarios (only 6/20 own-FAQ questions rejected). The word-overlap WARN catches all 58.
- Search-engine summaries of site policies can be stale; always read the live page.

**Next:** the P1 human gate (H-04 ethics, H-05 scenarios, H-09 LID labels, redirect hosting). Meanwhile P2 (retrieval configurations on D1/D2) needs the team's go-ahead for corpus encoding (⚙: about 1–2 h of bge-m3 on the GPU).

---

## 2026-09-28 · Session 2 · Phase 1 team decisions; Phase 2 retrieval

**Team decisions (D-038):**
- The team lead writes all 150 scenarios (75 Income Tax / 75 EPFO); the sheet was regenerated while still empty.
- Launch waits for ethics approval.
- GitHub Pages hosting.
- No UIDAI browser; Parivahan permission is not on the critical path.
- Phase 2 approved.

**Phase 2 runs:**
- Corpus encoding with Ollama bge-m3 (`num_ctx` 8192): FiQA 57,638 passages in 28.6 min (33.6/s); CQA-Android 22,998 in 10.1 min (37.9/s). Peak VRAM 0.62 GiB.
- **E0 run `E0-20260928T003743-14d0b0`** (git `53896ec`): C1 0.2380 (ref 0.236 ± 0.02 ✅); C3 0.4128 (ref 0.413 ✅). The FiQA ranking is C5 > C7 > C3 > C6 ≈ C4 > C2 ≈ C1.
- Latency: BM25 ~1 ms, dense 16 ms, rerank 650–880 ms per query.
- CQA-Android: corpus-side work plus a smoke run only (D-039); no sealed access.

**Learned**
- 3 of 21 configuration pairs are near-ties on authentic FiQA (Δ ≤ 0.003). τ_b resolution is lower than 21 pairs suggests. An exploratory "authentic noise ceiling" is suggested to the team; nothing changes.
- Ollama's bge-m3 matches the published dense number, so a HF download is not needed.

**Next:** waiting for H-04 (ethics), H-05 (scenarios) and H-09 (LID labels). Phases 3 and 4 wait for the data gates.

---

## 2026-09-28 · Session 2 (continued) · Documentation system; exploratory noise ceiling

- The team approved Phase 2 as-is and approved the noise ceiling as **exploratory only** (D-040).
- **Noise ceiling** (D1 FiQA; E0 per-query tables; no retrieval re-run; no sealed data):
  - (a) complementary halves: mean τ_b 0.779, median 0.810;
  - (d) independent resamples: mean 0.857, median 0.905, 14.6% of samples reach τ_b = 1;
  - (b) frozen R_s vs A_s: 0.905 / 0.810 / 0.714 / 0.810 / 0.524.

  Runs: `EXP-noise-ceiling-20260928T011823-654ce0` (a–c) and `…011917-5bc429` (a–d).
  **Finding:** complementary halves are anti-correlated on pair differences, so near-ties flip about 95% of the time instead of about 50%. Variant (d) was added as a labelled sensitivity check (D-040 addendum).
- New `docs/PAPER_LOG.md`: a cumulative, paper-section-organised record with sourced numbers.
- New `docs/AGENT_CONTEXT.md`: the live snapshot for agents.
- CLAUDE.md and PLAN §0 now require reading `AGENT_CONTEXT.md` first, and updating both files at the end of every session.
- `ha.eval.agreement.tau_b` (scipy variant b) is implemented and tested; the H2 analysis will reuse it.

**Next:** the gates (ethics, scenarios, LID labels). No Phase 3/4 work until they clear.


---

## 2026-09-28 · Session 2 (continued) · Human-side intake

The team uploaded `data/humanside/`: a phase 1 review note, a ChatGPT-assisted scenario draft (150), and a 100-example LID audit plus its protocol.
- **Ethics** is reported done (D-041; details pending).
- **Scenario draft** (D-042): structure identical to the generated sheet. Agent leakage check: 141 PASS / 2 WARN / 7 REJECT (everyday words that are rare in the corpus). 4 professional role-play rows found; the filter gap (ERI, "(CA)") is fixed with a test.
- **LID audit** (D-043): `human_final_label` is empty for all 100 rows; per-row system predictions are withheld to keep the review blind.
- `data/humanside/` is now versioned (a `.gitignore` exception).

**Next:** team-lead approval of the scenarios, the LID human labels, and the ethics details.


---

## 2026-09-28 · Session 2 (continued) · Review artifacts processed (D-044)
- The team confirmed that the source artifacts are the existing `data/humanside/` files; the separately named "reviewed" and "recommendations" files never existed. (Earlier I reported them as missing after a filesystem search.)
- Added a review-safe checker (`--sheet/--out`), a deterministic replacement draw (with `--exclude`), and the LID adjudication-table builder. 103 tests pass.
- The checker on the draft gives 141/7/2. Replacement chunks were drawn; the S025 replacement is professional and flagged. The adjudication table flags the 10 cases.
- The official sheet's hash is unchanged. Ethics placeholder added. Neither gate is passed.


---

## 2026-09-28 · Session 2 (continued) · ChatGPT review recommendations applied to the review artifact (D-045)
- Inspected the source chunk of every row touched: the 7 rewrites, 2 WARNs and 4 replacement chunks, plus the S025 re-draw.
- The rewrites were applied verbatim, marked pending. Support flags: S005 (weak), S090 (need dropped), S091 (premise omitted), S096 (employer role-play). 4 still fail the rare-term rule (everyday words); not reworded to beat the checker.
- S025 re-drawn deterministically with exclusions → `incometax/6cebf3b156cc/003`. The 4 replacement scenarios are agent (Claude Code) drafts, pending approval.
- Review-artifact check: 144/4/2. The official sheet and `langid_review_100.csv` are byte-identical before and after.
- LID recommendations file created; table rebuilt (L54 and L55 differ from provisional). `human_final_label` is untouched.
- The ethics placeholder is unchanged (no facts added). **Not committed**, by instruction.


---

## 2026-09-28 · Session 2 (continued) · Phase 3 synthetic pools (D-046, D-047)
**P3 generation proceeded in parallel with the remaining P1 human gate.**

**Scope from spec §7, the banner and the prereg:**
- Run: P-std and P-persona for D1, D2 and D4; P-fewshot s0–4 for D1 and D2.
- Deferred: D4 P-fewshot.
- Stopped: G2 (gemma3:4b missing), docT5 (download), D3.

**Engine**
- Frozen settings; cache; resumable; every attempt logged.
- Few-shot reads R only (tested). 7 new tests; 110 in total pass.

**Smoke test:** found Ollama KV-prefix nondeterminism at T 0.7. Fix: every pool starts from a fresh model state in fixed order (6/6 reproduced).

**Pilot:** the copy filter drops 15–25% on question-shaped corpora. Verified genuine; not changed.

**Full run:** the gate is FAIL (partial).
- Size ≥ 90%: 6/16 (D1 P-std and all D1 few-shot); the total was first mis-added as 7/16.
- Failures < 5%: 16/16.
- The D2 single failure is pid 50091: 7,885 tokens against 4,096, rejected.

**Next:** the team decides on (a), (b) or (c). Phase 4 must not start. Nothing is committed.


---

## 2026-09-28 · Session 2 (continued) · P3 closed: pools retained (D-048)
- The team chose option (a): retain all 16 pools exactly as generated.
- P3 AUTO GATE recorded as a partial failure: **FAIL** on size ≥ 90% (**6/16**), **PASS** on failures < 5% (16/16; max 0.13%).
- **Correction:** the pool totals were first mis-added as 7 pass / 9 fail. The consistency check against the per-pool `stats.json` files gives 6 / 10 (D-047).
- No filter change, top-up, context change or pre-registration amendment. Not a termination of the study.
- Verified: pool files are hash-identical before and after (48 files); stats, logs and parquet files are consistent; frozen artefacts are unchanged.
- P3 committed on its own (the Phase 1 D-045 work stays uncommitted). Phase 4 has not started.

---

## 2026-09-28 · Session 2 (continued) · Phase 1 D-045 commit; Phase 4 started (D-049)
- **D-045 committed separately:** `4d5f2f0` ("Phase 1: finalize human gate review artifacts"). The artifacts were verified (draft 150 rows, checker reproduces 144/4/2, S025 re-draw reproducible, LID human labels empty). One stale snapshot phrase in AGENT_CONTEXT was corrected. P3 at `809ceec` is untouched.
- **P4.1 neighbours:** D1 and D2, seeds 0–4, 0 near-duplicates, about 100 s of reranking per seed. `Reranker` gained a max_length override (128 for neighbours; the retrieval default is unchanged).
- **P4.2 judge engine:** prompt, run and aggregate code, plus 10 tests. Pilot `P4-judge-pilot-20260928T230859-1f27d5`: 38/38 JSON, 1.50 s per call, deterministic 4/4.
- **P4.3:** calibration selection and split code (tests; blocked without HiAQ). The annotation guidelines are a DRAFT.
- **Tests:** 126 pass (one CLI example switched from `neigh` to `select`, because `neigh` is now implemented). Ruff clean.
- **Blocked:** D4 (no R), P4.3 (H-07), P4.4–P4.5 (H-08). 4 team decisions are open. **Nothing committed.**

---

## 2026-09-28 · Session 2 (continued) · P4 calibration decisions (D-050)
- **Stop and report:** the first proposal (no passage for authentic items; annotators see the judge's neighbours) conflicted with prereg §4.3 (F on all 200; blinding) and §4.6. It was measured on the real draw: 42/70 FiQA synthetic calibration items would show a CAL-auth query as a neighbour, exposing 20/30. Reported with options; nothing changed.
- **The team chose** 1 = A (qrels passage), 2 = D (human neighbours from R_s0 \ CAL-auth), 3 (HiAQ domain sentence), 4 (spreadsheet).
- **Implemented:**
  - `src/ha/human/calibration.py`, `configs/domains.yaml`, `ha.config.domain_description`;
  - the CLI `ha calib status/export/import/finalize`;
  - guidelines v1;
  - `tests/test_calibration_sheet.py` (11).
- **Run:**
  - `ha calib status`: FiQA 30/30 passages resolved (19 with several qrels); HiAQ 0/30 pending.
  - `ha calib export`: BLOCKED (30/60).
  - `ha calib export --practice`: 20 items, 24 s, reproducible byte for byte.
  - Dry run on the FiQA half: 0 CAL-auth in human blocks (judge side 42/70), 0 self-neighbours.
- **Git:** HEAD is `4b8d224` ("phase 123, +4ing", made by the user; it contains the earlier P4 work). Nothing committed by the agent.

---

## 2026-09-28 · Session 2 (continued) · Pre-calibration checkpoint (no design change)
- **Phase 1 human gate re-inspected (read only):**
  - LID `human_final_label` is empty in 100/100 rows.
  - The official scenario sheet is unchanged (`28176bc2…`; 150 rows, `scenario` empty in all).
  - The draft has 137 keep_draft, 7 rewrite_proposed, 4 skip_replace and 2 warn_human_review rows, all pending approval. S025 uses the re-drawn chunk `…6cebf3b156cc/003`.
  - The ethics metadata fields are still placeholders.
- **P4 calibration workflow re-verified:**
  - `ha calib status`: FiQA 30/30 passages resolved; HiAQ 0/30.
  - The main export is BLOCKED (exit 3).
  - The practice pack is unchanged (sha256 re-checked).
- **Checks:** 137 tests pass; ruff is clean. The prereg body, PREREG artefacts, the 55 snapshotted frozen files, P3 pools 48/48 and `data/neighbours` are all unchanged. No unseal log, no FROZEN.json.
- **Gap noted (not built):** the post-survey step survey_queries → processed/hiaq (queries, source-chunk qrels, first-tranche CAL-auth, splits), plus a hiaq dataset config.
- **Status:** human calibration and Phase 5 have not started. Nothing committed.

---

## 2026-09-28 · Session 2 (continued) · Phase 1 gate: approved decisions applied (D-051)
- **Ethics:** no metadata exists in any project file; placeholder unchanged.
- **LID:**
  - The current files flag 10 rows: L14, L29, L53, L54, L55, L65, L88, L89, L99, L100. The adjudication table's `human_attention` and review_status agree.
  - No approved labels exist, so none were written. `langid_review_100.csv` sha256 is unchanged (`1c24efd3…`).
- **Scenarios:**
  - Applied the team's 9 approved texts (4 replacements, 5 rewrites) verbatim. All differed from the current drafts; every change was asserted to touch only these rows and the columns `scenario`, `author`, `review_status` and `agent_flag`. CSVs round-trip byte-identically otherwise (CRLF, BOM).
  - S090, S091, S060 and S066 were annotated in `agent_flag` only (text and status unchanged).
  - Draft check: 147 PASS / 2 WARN / 1 REJECT.
  - Scratch-only check: the earlier S090 and S091 versions also REJECT (unmarried; military, serviceman).
- **Guidelines v1:** no confirmation date supplied; the line is left blank.
- **Not touched:** the official sheet, the prereg, PREREG.json, P3 pools, and the judge and neighbour code. Nothing committed.

---

## 2026-09-29 · Phase 1 artefacts finalised (D-052)
- **LID:** the team's 10 labels plus the 90 provisional labels written to `human_final_label` (100/100). Asserted: only that column changed, and only L54/L55 differ from provisional. The adjudication table was rebuilt with the existing tool (`--recs`). The CSV format is preserved (BOM, CRLF).
- **Scenarios:** S090/S091 set to the team's exact texts; all 150 marked approved. Official sheet built with its own conventions and verified against `chunks.parquet`. The first run stopped before writing on a too-strict "heading_path required" rule; 66 official rows legitimately have an empty heading, so the rule was changed to "equals the chunk store". Official sha256 `4968e202…`.
- **Checker:** 146 PASS / 2 WARN / 2 REJECT. S090 also REJECTs ("unmarried"; not anticipated by the instruction). `hiaq-variants` refused, and nothing was written.
- **Not touched:** ethics placeholders, the guidelines date, prereg, PREREG.json, P3 pools, judge and neighbour code. Nothing committed.

---

## 2026-09-29 · §6.3 exception allowlist; survey variants built (D-053)
- **Team:** approved the S090 exception (like S091) and an explicit, fail-closed allowlist mechanism.
- **Implemented:**
  - `configs/scenario_check_exceptions.yaml` (S090/S091: decision, exact checker reason, text sha256, why);
  - `survey.gate_check` plus `load_check_exceptions`, used by `write_variants`;
  - 12 tests.
- **Run:**
  - `hiaq-check`: 146/2/2 (REJECT S090, S091; WARN S060, S066).
  - `hiaq-variants`: 25 variants, `survey_variants.json` sha256 `e125c5c2…`, deterministic on rebuild.
  - Real-data negative check: an injected REJECT on S005, S060 or S137 blocks; an empty allowlist blocks.
- **Unchanged:** the checker, the scenarios and the official sheet (`4968e202…`), LID labels, prereg, PREREG.json, P3 pools, judge and neighbour code. The ethics metadata and guidelines date are still blank. Nothing committed.

---

## 2026-09-29 · Google Forms builder checked and fixed (D-054)
- **Found:**
  - pasting the JSON into a single-quoted JavaScript string breaks (757 newlines, 4 apostrophes);
  - the optional language options were visible while people answered (a priming risk);
  - `setRequireLogin` works only on Workspace accounts;
  - forms would have opened immediately;
  - there was no creation record.
- **Fixed minimally:** object-literal paste; a fail-closed pre-check (25 × 6, 150 IDs, sha256 `0b3bb2e7…`); a page break before the optional questions; try/catch around `setRequireLogin`; forms created closed; one `MANIFEST_JSON` log line per form. Added `ha data hiaq-forms-manifest` and 8 tests.
- **Dry run (Node, mock FormApp):** 25 forms OK. Truncated and edited pastes create 0 forms. The manifest check passes, with all 150 texts verbatim (S090 in form 05, S091 in form 11).
- **Not done:** no forms created in Google (the team runs them), no pilot, no launch. Nothing committed.

---

## 2026-09-29 · 25 Google Forms created (team) and verified
- **Team run:** `make_forms.gs` ran in the team's account from 18:51:15 to 18:54:20 local time. The log was saved as `data/hiaq/forms_creation_log.txt.txt` (Windows double extension; left as is). It has 25 MANIFEST_JSON lines, plus 25 "setRequireLogin not applicable" notes (personal account: no sign-in required) and no errors.
- **`ha data hiaq-forms-manifest --log …`:** 25 forms verified → `data/hiaq/forms_manifest.json` (sha256 `b5775eeb…`; records variants sha256 `e125c5c2…`).
- **Independent check against the official sheet — all PASS:**
  - forms 1–25 each once; unique form IDs and response URLs; well-formed edit and response URLs;
  - 6 items per form; the 150 IDs each exactly once; order equals `survey_variants.json`;
  - every item text equals the official sheet verbatim. S090 (form 05) and S091 (form 11) match the allowlist text hashes;
  - all forms were not accepting responses at creation.
- **Not done:** no forms opened, shared or published; `redirect.html` untouched; no pilot or launch. Ethics metadata is still blank.

---

## 2026-09-29 · Ethics metadata waived by the team lead (D-055); pilot prepared
- **Team lead:** approval was informal, so the approver, role, date, scope and conditions are skipped. The five placeholders are marked "not recorded: waived (D-055)"; no values were invented. The protocol §0 gate is updated; the paper's ethics line is limited to "informal approval".
- **Safeguard:** `ha data hiaq-ingest --pilot` (reads `pilot_responses/`, writes `pilot_queries.parquet`). Before this, ingest always wrote the dataset file. +1 test.
- The protocol §3.4 now lists concrete pilot steps: open 5 forms, direct links, close, export, ingest `--pilot`, delete pilot responses before launch.
- **Checks:** 157 tests pass; ruff is clean. No form was opened by the agent. Nothing committed.

---

## 2026-10-10 · Collaborative-sheet collection built (D-056)
- **State on return:** HEAD `7368887`, with 10 files uncommitted. The repo `CLAUDE.md` is deleted in the working tree, and an untracked `kl.md` with identical content (CRLF) is present. It was left as is and flagged to the team.
- **Built (option 3, decided 2026-09-29):** `collab_template`, `check_collab`, `write_collab_template` and `ingest_collab` in `survey.py`; `ha data hiaq-collab-sheet` and `ha data hiaq-collab-ingest`; +6 tests.
- **Generated** `data/hiaq/collab/collab_sheet_template.csv`: 600 rows, 100 blocks, each situation × 4, text equal to the official sheet. Also `HOW_TO_FILL.txt`.
- **Checks:** 163 tests pass; ruff clean. The official sheet (`4968e202…`) and variants (`e125c5c2…`) are unchanged. Nothing committed.

---

## 2026-10-10 · Correction to the previous entry
- **Correction:** the previous entry reported `CLAUDE.md` deleted and an untracked `kl.md`. The team has since **restored `CLAUDE.md`** (tracked; identical to HEAD) and **removed `kl.md`**. Verified 2026-10-10.

## 2026-10-10 · Repository audit; HiAQ data-role decisions (D-057); amendment A1 drafted
- **Audit (read-only):**
  - Prereg §3.1, §3.2, §4.1 and §8, D-056 and spec §6.3 checked: collaborative-sheet answers are not eligible for R, A or `CAL-auth` under current decisions.
  - The §8 fallback names only H2/H3. H1, H4, D4 qrels and calibration are not covered by its text.
  - The spec's own fallback (D1–D3) conflicts with the frozen prereg.
- **Team decisions (D-057):**
  - collaborative sheet exploratory only, not distributed;
  - Google Forms initial tranche: 25 completed forms, one per respondent (no multi-form deviation; reasons recorded);
  - fallback planned but not triggered;
  - amendment A1 drafted only (`docs/prereg_amendment_A1_DRAFT.md`).
- **Documentation fixes:** survey protocol section renumbered (§7 → §6); this correction; a supersession note in `reports/project_report_2026-09-30.md`.
- Nothing staged, committed, pushed, launched or ingested.

---

## 2026-10-10 · Incoming workbooks validated: 0 usable human queries
- **Received:** `HumanAlign_150_SYNTHETIC_Queries_LABELED.xlsx` (presented as "newly collected human responses") and `HumanAlign_150_Human_Query_Collection.xlsx`. Byte-identical copies are in `data/hiaq/incoming/2026-10-10_workbooks/raw/` (git-ignored), with SHA256SUMS and a quarantine README. The originals are untouched.
- **Finding:**
  - The first workbook's 150 queries are all labelled "AI-generated (synthetic) — not a human response". Its Read Me says they were written by an AI model (Claude).
  - The second workbook has 0/150 responses.
  - 9 situation texts differ from the official sheet. S017, S025, S136 and S147 are the replaced professional-role scenarios; S005, S007, S067, S096 and S137 match no recorded version.
- **Nothing integrated; no derived dataset.** Report: `reports/hiaq_incoming_validation_2026-10-10.md`; row-level CSV alongside it (no query text).
- **Status:** the pipeline is blocked at authentic-data validation. HiAQ `CAL-auth` shortfall is 30; about 90 for the D-050 display; 300 to avoid the §8 fallback.
- **Addendum (same day):** the team states the queries are human-written and the Read Me was attached by mistake. The in-sheet labels (sheet name, column header, the Source value on all 150 rows) and the file metadata still indicate AI generation. Classification is pending primary provenance (raw survey export). See the report addendum.
- **Second file (same day):** `human_written_version5 (1).csv` (source "human") has text identical on all 150 rows to `human_written_version5.csv` (source "synthetic") from 6 minutes earlier. Only the label was changed, and 55/150 rows closely match the earlier AI set. Quarantined in `data/hiaq/incoming/2026-10-10_human_written_v5/`; not ingested. Authentic HiAQ count remains 0.
- **Team-lead attestation (same day):** the team lead states the 150 queries in `human_written_version5 (1).csv` were written by humans and asks that they be used as the HiAQ human dataset.
  - **Structural checks pass:** 150 rows; S001–S150; 0 empty; 0 duplicates; 0 PII; mapping by ID to the official sheet; hinglish 78 / english 72; 20 rows closely echo their situation text.
  - **Not ingested by the agent as authentic data:** the provenance conflict (identical text labelled "synthetic" six minutes earlier; overlap with the AI set) is unresolved. Pending primary evidence of who wrote which rows.

---

## 2026-10-10 · D-058: the 150 queries become an external synthetic set; amendment A2 drafted
- **Team decision:** the 150 queries are AI-generated (team-researched) synthetic queries, not human. The team cannot collect authentic HiAQ responses.
- **Imported** with `tools/import_external_synthetic.py` → `data/synthetic_external/hiaq/X-team150/`: 150 rows, text exact, mapped to official chunks; hinglish 78 / english 72; 0 PII. `load_U('hiaq')` is unaffected.
- **A2 draft** (`docs/prereg_amendment_A2_DRAFT.md`): calibration is FiQA-only (30 authentic + 70 synthetic; 60/40), because the HiAQ half is infeasible (no `CAL-auth`; no D4 R for neighbours). A1 revisions are listed. Nothing recorded; the prereg is unchanged.
- **Proceedable now:** practice annotation (after the guidelines date); D1 development runs and `X-team150` descriptives once the run loader supports synthetic sets, after a commit (registry rule). After A1 and A2: calibration → freeze → D2-only H2/H3.

---

## 2026-10-10 · D-059: reduced scope; A2 implemented behind a recording guard; A1 rev 2
- **Team:** HiAQ collection closed at zero; A2 = FiQA-only 100-item calibration; A1 revised; `X-team150` separate.
- **Checked A2 against prereg-v1 §4, rule by rule** (table in the A2 draft):
  - FiQA items and their dev/test assignment are identical to v1 (test);
  - the authentic share stays 30%;
  - consequences: precision; English-only calibration; the literal 40-item re-annotation rule; the C weight is not identifiable (selection unaffected); no code-mixing calibration.
- **Implemented (uncommitted):**
  - `configs/calibration.yaml` (`amendment: A2`, hiaq 0, 60/40);
  - `ha.judge.calibrate` (`amendment_recorded` guard; generalised `build_items`);
  - `ha.human.calibration` (status shows the amendment; wording);
  - guidelines counts;
  - `tests/test_calibrate.py`.
- **Verified:** `ha calib export` → BLOCKED until A2 is recorded; 172 tests pass; ruff clean; frozen files unchanged.
- **Open:** v1/v2 labelling (prereg line 3), guidelines confirmation, recording A1 and A2, commit.

---

## 2026-10-10 · A1 and A2 recorded (D-060)
- **Rules checked:**
  - amendments need only a date and an append below `## Amendments` (prereg line 3, D-016, spec line 69);
  - no rule names approvers, so the lead's approval suffices;
  - joint signatures are required at the §6 freeze;
  - faculty approval applies only to the survey launch.
- **Appended:** A1 and A2 as approved text sha256 `6194857e…`, with only the date (2026-10-10) and name (Chinmai Shetti, project lead) placeholders filled → recorded text sha256 `8b91f44d…`.
- **Integrity:** byte-level append-only (old file is an exact prefix; CRLF preserved); frozen-body hash equals PREREG.json; `check_frozen` OK.
- **Test update:** the state test "A2 not yet recorded" now asserts that A1 and A2 are recorded.
- **Still pending:** the guidelines confirmation by both annotators, and the reviewed commit. `ha calib export` has not been run.

## 2026-10-14 · Pre-commit review; name confirmed; committed
- **Review of all 22 files:**
  - no unrelated files; no sensitive data (only decorator and fake-test-number false positives);
  - the recorded amendment text equals the approved text with only the placeholders filled;
  - 8 stale live-doc lines fixed (AGENT_CONTEXT, PAPER_LOG, survey protocol, guidelines status).
- **Recorded name "Chinmai Shetti (project lead)":** confirmed by the lead.
- **Checks:** prereg append-only; frozen-body hash equals PREREG.json; 11/11 other artefacts and P3 48/48 unchanged; 172 tests pass; ruff clean; X-team150 excluded.
- **Committed** at the lead's instruction; not pushed. Still pending before `ha calib export`: the guidelines confirmation by both annotators.
- **2026-10-14:** guidelines v1 finalised by the lead (D-061; the §4.3 rule names no signatories). Committed before running `ha calib export`.
- **2026-10-14:** `ha calib export` at clean commit `1f94dcb` → `data/calibration/`:
  - 100 items (30 authentic + 70 synthetic FiQA); `cal-dev` 60 / `cal-test` 40 (authentic 18/12, synthetic 42/28);
  - blind sheets `raw_r1_A1.csv` and `raw_r1_A2.csv` (empty) and `blank.csv`; `key.csv` sha256 `883ba588…`;
  - 0 CAL-auth shown as neighbours; 5 neighbours on every row.
  - **Next:** practice round, then annotation by two annotators (H-08).

## 2026-10-14 · Amendment A3 (no human calibration); bulk-judging runner
- **Lead's decision:** skip the human rating (H-08) and the two-annotator steps. Recorded as **amendment A3** (D-062):
  - prompt v0 is final, with equal weights and gate F ≥ 4;
  - κ, ρ, NNLS and `cal-test` are not executed;
  - the freeze is signed by the lead;
  - the analysis is prereg-v3.
- **Recording check:** byte-level append; A3 text sha256 `08435f6a…`; frozen-body hash unchanged (`487199ff…`); `check_frozen()` passes. `configs/judge.yaml` is untouched (hash-recorded at the tag).
- **P5.1 runner:** `src/ha/judge/bulk.py`, run as `ha judge <ds> [--seeds] [--no-iso] [--limit]`.
  - It refuses to run until A3 is recorded.
  - Resumable: one JSONL log per stage. Transport failures are retried, and 5 in a row stop the run.
  - `--limit` smoke runs write to `data/judge_smoke/`.
  - Registry rows are written only for a complete full run.
- **Tests:** `tests/test_bulk.py` (6). 178 tests pass; ruff clean.
- **Smoke test** at clean commit `872c069`: `ha judge fiqa --limit 16 --seeds 0` writes to `data/judge_smoke/`.
  - M1-iso: 16/16 valid, 12 pass the gate, 0 retries.
  - M1 s0: 16/16 valid, 11 pass the gate, third-permutation rate 2/16, 0 retries; max prompt 986 tokens (under num_ctx 4096).
  - The first 12 queries were cache hits from the P4.2 pilot (identical prompts); the rest were live calls, about 2 s each. Resume worked (the first, 3-query smoke run was skipped).
  - Sanity check: `fiqa-P-std-0015` (an odd, document-like query) gets F = 3 without neighbours and 1/1/1/1 with them, in both permutations. This is a genuine judgement; the `reason` label is off, a known v0 weakness that selection does not use.
- **Short live check (about 5.5 min):** 50 FiQA + 30 Android queries, seed 0 (`data/judge_smoke/`).
  - 100% valid outputs, 0 retries, about 1.5 s per call.
  - M1 gate pass: FiQA 32/50, Android 27/30. M1-iso: 41/50 and 25/30.
  - Top scores go to natural user questions; gate failures are document-echo queries ("according to the passage", "the speaker").
  - Many queries tie at the maximum equal-weight score (20). Selection needs a fixed, seeded tie-break, to be set before the freeze.
- **Pause and resume:** `tools/run_judging.bat` (double-click) runs fiqa, then cqa_android, with up to 5 automatic retries. Close the window to pause; run it again to resume.
  - Tested with a hard kill (`taskkill /T /F`) after 62 of 80 queries: the rerun resumed at 62 and finished 80/80, with unique qids and 0 failures.
- **Next:** the full run on the lead's go-ahead (about 13 h).

## 2026-10-15 · P5.1 bulk judging: D1 done, D2 paused
- **Started** 2026-10-14 23:34 with `tools/run_judging.bat` at clean commit `88ba739`.
- **Accidental PC shutdown about 05:09** (FiQA M1 s4 at 448/1443). The log was intact (no torn line); resumed at 05:56 and nothing was lost. The tool resumed as designed.
- **D1 FiQA finished at 06:49**, registry run `P5.1-judge-fiqa-20261015T064926-d3c37d`:
  - 0 failures;
  - gate pass M1-iso 76.2%, M1 s0–s4 67.2 / 66.7 / 67.8 / 67.6 / 66.1%.
- **D2 Android paused by the lead at 06:53** (M1-iso 174/1295; M1 not started). The model was unloaded.
- **Next (on the lead's instruction):** resume D2; build selection (with a seeded tie-break), evaluation, the bootstrap and `FROZEN.json`.

## 2026-10-15 (evening) · P5 code built; D2 judging resumed
- **D2 judging resumed** 23:25. A rerun no longer rewrites a finished stage's record (`c53c812`). FiQA's iso and s0–s3 meta had already been re-finalised once, at the 05:56 resume after the power-off. Their timings are recoverable from the JSONL timestamps; their rates are unchanged.
- **Built and tested:**
  - selection (`2b5af17`, D-063); FiQA sets for all 5 splits, deterministic (byte-identical rerun), at clean `263570b`;
  - evaluation and bootstrap (`a661c5d`); H1 (`3d2b192`, D-064);
  - freeze and the seal guard (`f87d272`).
- **C1–C4 on U, both datasets** (CPU, cached embeddings). Synthetic queries score far higher than real ones (PAPER_LOG). C5–C7 wait for the GPU.
- **Process note:** one commit (`2b5af17`) went in while a test was failing (a stub-test update) and a lint error was open. Fixed in `263570b`; later commit commands stop on any failure.
- **Next:** `tools/run_p5.bat` finishes P5. Then the freeze, the lead's signature, and the held-out run.

## 2026-10-16 · HiAQ collected batch "group a" imported (D-065); pipeline fix
- 150 collected HiAQ queries were imported on the lead's statement (D-065). 146 are scoreable; 9 have non-official situation wording, flagged. They serve an exploratory HiAQ pilot only (150 < 300, A1).
- `run_p5` overnight:
  - D2 judging finished at 04:49 with 0 failures;
  - C5–C7 on U for both datasets finished at 06:12;
  - D2 selection is done.
- **The FiQA dev evaluation did not run.** The D2 judging registry rows left `results/registry.jsonl` modified, and the clean-commit guard refused. Fix: `git_sha()` now ignores the append-only registry. The registry rows were committed.

## 2026-10-16 · Freeze signed (D-066); held-out run
- The lead approved; the freeze was signed at 05:14 UTC (`4bebf60`). FROZEN record sha256 `5b1ab039…`.
- `HA_UNSEAL=heldout_A`:
  - C1–C7 on D2's 699 authentic queries;
  - `ha eval cqa_android`: **H2 not supported** (p 0.211 / 0.106), **H3 not supported** (p 0.079). M1's point τ is higher in all 5 seeds.
  - `ha h1 fiqa cqa_android`: length and overlap differ on both datasets; topic on D1 only; τ_b(B0, A) < 0.8 on both.
- Details in PAPER_LOG §6.1–6.2.

## 2026-10-17 · Exploratory HiAQ pilot finished (D-067)
- Ran 00:10–06:44 via `tools/run_hiaq_pilot.bat`: neighbours, judging (0 failures), C1–C7, selection, evaluation.
- Result: M1 τ_b 0.46 vs B0 0.81 and B1 0.76. M1 is worse than random on HiAQ (PAPER_LOG §6.3).

## 2026-10-17 · Exploratory -v2 selectors (D-068)
- Specified and committed before any run (`eaae160`), then run unchanged on D1, D2 and D4.
- **OVL (overlap-matched random)** beats random on all three: +0.10, +0.18, +0.11 τ_b, p < 0.001. **NN** does not.
- Details in PAPER_LOG §6.5.

## 2026-10-17 · Paper in the repo; wrap-up
- Novelty check #2 done (`753017b`).
- Paper moved into the repo as `paper/paper.md`, with generated Figure 1 and Tables 1–3 (`cd327e8`). The repo copy is the main one; the online doc drafts are no longer updated.
- The lead consolidated all work on `main` and pushed it. AGENT_CONTEXT and PLAN now show the study complete.

## 2026-10-17 · Replication R1 (D-069, D-070)
- The protocol was committed before any download.
  - D5 CQADupStack-Webmasters and D6 NFCorpus were built with counts matching BEIR.
  - Pools: 1,361 and 1,484 queries, 0 failures. N = 305 and 193.
- Freeze `d489fe5`, then unseal and authentic runs (`06d46c3`), then one evaluation.
- **R1 supported pooled** (p = 0.005, 0.004). The effect is driven by D5 (+0.15/+0.20); D6 shows none.
- The paper got §5.2 and Table 4 (generated).

## 2026-10-17 (night) · D-071 analyses; R2 started
- R2 data steps launched 23:04. The first launch exited because Windows PowerShell 5.1 split the dataset list; fixed in `eac3aee` and relaunched.
- E-metrics first failed on an undefined τ (all systems tied, D4 Recall@100). It is now reported as undefined and kept out of the registry (`f23ebd0`); nothing partial was written.
- Results in PAPER_LOG §6.7. The density explanation for NFCorpus is **not** supported. OVL is robust across metrics on D1, D2 and D4. H drives HiAQ's M1 failure.

## 2026-10-18/19 · Replication R2 (D-072)
- The R2 run stopped overnight after SciFact P-std generation, a transient failure with no power events. The launcher now retries each step 3 times and logs to `logs/` (`a22f7f0`). Relaunched 17:49; data steps done 22:30.
- Freeze `5306510`, then unseal and authentic runs (`4f190ba`), then one evaluation.
- **R2 supported pooled** (p = 0.001, < 0.001): Mathematica +0.14, Programmers +0.17, SciFact −0.14.
- The paper got §5.2 R2 and Table 5 (generated); the abstract was updated.
