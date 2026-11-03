# HiAQ incoming workbooks: validation report (2026-10-10)

**Verdict: 0 usable authentic human queries.** Nothing was integrated.
- **"SYNTHETIC_Queries_LABELED" workbook:** all 150 queries are **AI-generated**, by the workbook's own row labels and Read Me.
- **Companion human-collection workbook:** **empty** (0 responses).

Row-level data: `reports/hiaq_incoming_validation_2026-10-10_rows.csv`. It holds scenario IDs, flags and hashes only, no query text.

## 1. Inputs (raw files preserved unchanged)
Originals are on the Desktop and in Downloads (not modified). Byte-identical copies are at `data/hiaq/incoming/2026-10-10_workbooks/raw/` (git-ignored), with `SHA256SUMS`.

| File | sha256 (first 16) | Created / modified (file metadata) | Sheets |
|---|---|---|---|
| `HumanAlign_150_SYNTHETIC_Queries_LABELED.xlsx` | `953d88038a9a7918` | created by `openpyxl` 2026-10-09 21:05 UTC; last saved in Microsoft Excel by "Chinmai Shetti", 21:07 UTC | "Synthetic Queries (AI)" (150 rows), "Read Me" |
| `HumanAlign_150_Human_Query_Collection.xlsx` | `c267679e3801436a` | created by `openpyxl` 2026-10-09 20:47 UTC; never modified (created = modified) | "150 Scenarios" (150 rows), "Read Me" |
| 4 CSV exports of the sheets above (Downloads) | see `SHA256SUMS` | 2026-10-10 02:57 local | — |

## 2. What the "SYNTHETIC_Queries_LABELED" workbook contains
- **Columns:** Scenario ID · Service · Situation to imagine · **Synthetic query (AI-generated)** · Language mix · **Source** · Notes.
- **Rows:** 150 (S001–S150, unique). Non-empty queries: 150. Service: Income Tax 75, EPFO/PF 75.
- **Source column: identical on all 150 rows:** "AI-generated (synthetic) — not a human response". Notes: "Use only as a labeled comparison baseline".
- **Read Me sheet, verbatim:**
  - "All 150 queries were written by an AI model (Claude) …"
  - "They are NOT human-collected data. Keep the Source column on every row …"
  - "Use them only as a clearly labeled synthetic comparison set."
- **Conclusion:** synthetic, from a model outside the project's locked stack.
  - This rests on the file's explicit content (column header, per-row Source label, Read Me), **not** on its filename or its style.
  - There is no evidence of any human-written row. Respondent IDs, timestamps and per-row provenance are absent.

**Automated flags** (they describe the file; they don't decide anything):

| Check | Result |
|---|---|
| Empty | 0 |
| Exact duplicates | 0 |
| Near-duplicates (SequenceMatcher ≥ 0.85) | 0 |
| Close paraphrase of the scenario text | 7 (S062, S114, S120, S121, S125, S134, S137; token overlap 0.71–0.89) |
| PII (project patterns: phone, email, Aadhaar-like, PAN) | 0 rows |
| Language label given | English 81, Hinglish 57, Kanglish 8, Hindi (Devanagari) 4 |
| Project LID (`ha.data.langid`, automated) | EN 89, HINGLISH 53, ROM_HI 4, DEV_HI 4. It agrees with the given label on 138/150. The project LID has no romanised-Kannada class, so all 8 "Kanglish" rows were classed EN. |
| Uniformity | 96.7% end with "?"; 29.3% contain two or more questions; median 118 characters. The language mix looks assigned, not observed. |

## 3. What the human-collection workbook contains
- **Columns:** Scenario ID · Service · Situation to imagine · "Your natural query" · Language used (optional) · Notes (optional).
- **150 rows; responses 0/150; language 0/150; notes 0/150.** The file was never edited after it was generated.
- **Its instructions differ from the approved instrument** (the Google Forms, Appendix D, D-057):
  - They invite "whichever language **or mix of languages**". That is a code-mixing cue; spec §6.3 says "Do not mention Hinglish".
  - There is no publication or data-use statement.
  - It is one shared sheet with ad-hoc row assignment and no timestamps, so it has the same independence problems as the D-056 sheet.

## 4. Scenario text vs the official sheet (`data/hiaq/scenario_sheet.csv`, sha256 `4968e202…`)
Both workbooks use the same situation texts. **141/150 match the official sheet; 9 do not.**

| IDs | Workbook text is… | Consequence if used |
|---|---|---|
| **S017, S025, S136, S147** | the **original professional-role scenarios** from the first draft (`d163086`/`3e45616`). Those rows were **skipped and replaced with different source chunks** in D-051. | The answers would map to the **wrong chunk**, so their qrels would be wrong. |
| **S005, S007, S067, S096, S137** | text that matches **no version on record** (similarity to official 0.10–0.85). For example, S096 is employer role-play, which the team rejected in D-051. | It isn't the approved wording, and where the scenario came from is unknown. |

## 5. Role of this data (Task 3)
- **Synthetic workbook:** **AI-generated from scenario texts.** It is not authentic and not elicited from people. It is also not part of the pre-registered synthetic pools: those are passage-to-query outputs from `qwen3:4b-instruct` (prereg §5.2; P3).
- **Human workbook:** an empty instrument.
- **Provenance categories asked about** (recalled experience, scenario-elicited, copied from earlier searches): **none apply**, because there are no human responses.

**Eligibility under the frozen design:**
- **Authentic HiAQ (R, A, `CAL-auth`, eligible count for prereg §8):** **not eligible** (synthetic).
- **Calibration synthetic items (prereg §4.1):** **not eligible.** These must come from the seed-0 P-std/P-persona pools.
- **Prompt anchors:** **not eligible**; anchors come from `cal-dev` only (§4.5).
- **Replacing or adding to the P3 pools:** **not allowed** (D-048; prereg §5.2).
- **The only defensible use** is an exploratory, labelled "external-LLM, scenario-elicited synthetic" comparison. That needs a dated decision and must stay out of every pre-registered analysis (prereg §2, §7.7).

## 6. Exclusions (raw data untouched)
| Rows | Proposed status | Reason |
|---|---|---|
| All 150 of the synthetic workbook | **Excluded from all authentic and pre-registered roles** | AI-generated (own Source label + Read Me) |
| All 150 of the human workbook | Nothing to include | No responses |

These are proposed statuses. The final §3.1 exclusion decisions are made by two team members, and only for genuine human answers.

## 7. Integration plan for genuine responses (unchanged pipeline; nothing new needed)
1. **Source:** the approved Google Forms (D-057; one form per respondent; links in `data/hiaq/forms_manifest.json`, response URLs only). Export each form to `data/hiaq/responses/form_NN.csv` and record each file's sha256.
2. **Ingest:** `ha data hiaq-ingest` → `data/hiaq/survey_queries.parquet`. This is the existing canonical schema: `qid, respondent, variant, scenario_id, chunk_id, text, pii_removed, empty, languages, device, submitted_date, lang_class, cmi`. The text is kept exactly (typos kept; whitespace only).
3. **Mapping:** form number + item position → `survey_variants.json` (`e125c5c2…`) → scenario ID → chunk from the official sheet. **Never from these workbooks.**
4. **Duplicates and flags:** report exact and near duplicates, scenario echoes and PII. Nothing is deleted automatically.
5. **Language:** automated LID and the respondent's optional self-report are stored separately; neither overwrites the other.
6. **Exclusions:** the prereg §3.1 two-person review (empty, not a query, off-topic), blind to any method output, with reasons in an exclusion log. Raw is kept.
7. **Eligible count:** recorded for amendment A1's condition (fallback if below 300).
8. **Splits:** the HiAQ processing step and `configs/datasets/hiaq.yaml` (**still to be built**): queries and source-chunk qrels (grade 2) → `CAL-auth` 30 from the first tranche (seed 17, by language) → `build_splits` (5 seeds, R 40% by language × cluster, `CAL-auth` forced into R).

## 8. Can the next pre-registered stage proceed?
**No.** The pipeline is blocked at step 1: there are 0 eligible authentic HiAQ queries. Shortfalls:

| Requirement | Needed | Have | Shortfall | Blocks |
|---|---|---|---|---|
| HiAQ `CAL-auth` (prereg §4.1) | ≥ 30 eligible | 0 | 30 | P4.3 calibration pack, and with it the freeze, bulk judging and evaluation |
| 5 annotator-facing neighbours (D-050, team rule) | about 90 eligible | 0 | about 90 | the HiAQ calibration display |
| Avoiding the §8 fallback | ≥ 300 eligible | 0 | 300 | D4 as a confirmatory dataset (amendment A1 if not met) |

Not blocked by data, and can be done now if approved: building and testing the HiAQ processing step and dataset config on fixtures.

---

## Addendum 2026-10-10: team statement on provenance (classification pending)
- **Team statement:** the queries are human-written survey answers; the "Read Me" sheet was attached by mistake and belonged to another sheet.
- **Evidence inside the workbook, independent of the Read Me** (checked in the raw file):
  - the data sheet is named **"Synthetic Queries (AI)"**;
  - its query column header is **"Synthetic query (AI-generated)"**;
  - **all 150 rows** carry the Source value "AI-generated (synthetic) — not a human response" and the note "Use only as a labeled comparison baseline";
  - the file was generated by `openpyxl` 17.5 minutes after the empty collection workbook was generated (20:47:49 → 21:05:21 UTC on 2026-10-09);
  - there is exactly one answer per scenario, with no respondent, timestamp or form information;
  - it uses the same 9 non-official situation texts as the empty collection workbook.
- **Status:** the rows stay **excluded from authentic roles until primary provenance is provided.** That means the raw survey export (for example Google Forms response CSVs with timestamps, or the sheet participants typed into, with its version history) and a description of who answered and how. Nothing was ingested.

---

## Addendum 2 (2026-10-10): `human_written_version5 (1).csv`, presented as human-written
- **Files** (Desktop; byte-identical copies in `data/hiaq/incoming/2026-10-10_human_written_v5/raw/`):
  - `human_written_version5.csv` (11:29): `source` = **"synthetic" on 150/150 rows**;
  - `.xls` (11:31);
  - `human_written_version5 (1).csv` (11:35): `source` = **"human" on 150/150 rows**.
  - The blank situations CSV had been created at 10:35.
- **All 150 `question_text` values and all 150 `language_mix` values are identical** between the "synthetic" and "human" files. Only the label changed.
- **Overlap with the AI set quarantined earlier the same day:**
  - 55/150 rows have similarity ≥ 0.6 for the same scenario;
  - 9 rows share a specific detail that is absent from the official situation, for example S002 "Germany" and "2 years".
- **Formal checks** (for completeness): 150 rows, S001–S150 unique, 0 empty, situations mapped by ID to the official sheet.
- **Conclusion:** these are the same texts that were labelled synthetic six minutes earlier. They are **not eligible** as authentic HiAQ data in any role. **Nothing ingested.**
