# Phase 1 report: Data (HUMAN GATE: scenarios final, survey launched)

Status: **engineering done; waiting on human tasks.** Pre-registration `prereg-v1` is frozen (commit `b1c18e0`; hashes in `results/PREREG.json`).

## Done
| WP | Result |
|---|---|
| P1.1 | D1 FiQA (57,638 passages / 648 queries) and D2 CQADupStack-Android (22,998 / 699), exact BEIR counts |
| P1.2 | Splits for D1 and D2 (5 seeds, R 40% / A 60%, stratified by language × 10 topic clusters). FiQA R = 260 / A = 388 with 30 CAL-auth always in R; CQA R = 279 / A = 420. A of D2 is sealed. D4 is split after the survey. |
| P1.3 | Model-free LID and CMI (D-032). 30 hand-labelled tests pass, **with provisional labels awaiting H-09**. All 1,347 English D1/D2 queries are classed EN, so there are no false code-mix positives. |
| P1-H1 | Source vetting on the live policies (D-031). **Included:** Income Tax e-filing help and EPFO. **Excluded:** NPCI/BHIM (robots disallow), DigiLocker (robots 403), IRCTC (terms), ABDM (no reproduction), Parivahan (permission by email), india.gov.in (403). **Parked:** UIDAI (JavaScript-rendered). |
| P1-H2 | Polite crawl: 1 request every 2 s, robots-checked, every request logged; 301 + 6 pages. **1,572 chunks** (725 FAQ pairs, 847 sections; 1,418 Income Tax, 154 EPFO), within the spec's 1,500–4,000 target and above the 800 source passages prereg §5.2 needs. |
| P1-H3 | Scenario sheet: 150 rows (111 Income Tax / 39 EPFO, 117 topics, individual-user topics only, D-035), split A/B 75 each. Leakage checker: spec rules plus a WARN signal (D-036). |
| P1-H4 | Form variants (25 × 6), Apps Script builder, random-redirect page, ingest with PII stripping. `docs/survey_protocol.md` gives the team's steps. |

Tests: 90 pass; ruff is clean.

## Waiting on the team (these block the P1 gate)
1. **H-04 · Ethics approval** from the faculty advisor, recorded in `decisions.md`. It must come before launch.
2. **H-05 · Write 150 scenarios** in `data/hiaq/scenario_sheet.csv` (rules and examples in `docs/survey_protocol.md` §1). Then run `ha data hiaq-check` until every row is PASS or reviewed WARN.
3. **H-09 · Verify the 30 LID test labels** in `tests/test_langid.py` (correct any that are wrong).
4. **Hosting of the redirect page:** where it lives is your decision (for example GitHub Pages).
5. **Optional, to widen the corpus beyond tax and PF:**
   - (a) email Parivahan's Web Information Manager for permission (driving licence and vehicle services); or
   - (b) approve a headless browser for UIDAI's JavaScript-only FAQ (Aadhaar). This is a download of roughly 150–300 MB (Playwright + Chromium, on D:); it is **not a model**, but it needs approval under the no-silent-downloads rule.

   Neither is needed to proceed; the corpus already meets its target.

## Risks noted
- **Topic concentration:** 90% of the corpus is Income Tax. Scenarios are balanced by the √ allocation (26% EPFO), but D4 retrieval is tax-dominated. This is disclosed in the data card.
- **Language balance:** Hindi speakers must be deliberately recruited (I-9) to reach ≥ 150 ROM_HI or HINGLISH queries.
