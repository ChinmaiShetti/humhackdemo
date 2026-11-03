# HumanAlign Phase 1 — Scenario & LID Review

## Scenario sheet

150 candidate scenarios were drafted row-by-row from the uploaded `scenario_sheet.csv`, preserving the 75 Income Tax / 75 EPFO split and the existing row order. Every candidate is written as a pre-document user situation in second person and avoids copying the chunk wording verbatim.

The source fields (`scenario_id`, `chunk_id`, `source`, `topic`, `heading_path`, `url`, and `chunk_text`) were left unchanged. The generated scenario field, author marker, and review note were added/filled only in the draft copy.

Final inclusion is still a human/team-lead decision. After approval, run `ha data hiaq-check` and manually review every WARN/REJECT before using the sheet to build survey forms.

## External web cross-check

Official Income Tax Department material was used only as contextual sanity-checking for the service domains represented in the sheet, including e-Verify, grievances, return filing, refund status, and tax-payment services.

Official EPFO material was used only as contextual sanity-checking for the member-service domains represented in the sheet, including UAN/member services, passbook and claims, pension, life-certificate submission, and international-worker services.

## LID review — expanded audit

The original 30 hand-labelled LID examples were retained and expanded to a **100-example audit set**. The audit uses the same frozen class definitions and is intended to validate the implementation, not to tune HumanAlign or the held-out evaluation.

Provisional class counts in the 100-example audit:

- `EN`: 20
- `ROM_HI`: 20
- `HINGLISH`: 20
- `DEV_HI`: 21
- `OTHER`: 19

The `OTHER` class deliberately includes non-target Indian languages/scripts rather than creating separate Kannada/Tamil/Telugu/etc. classes. Current examples cover:

- Kannada
- Bengali
- Tamil
- Malayalam
- Telugu
- Punjabi/Gurmukhi
- Gujarati
- Urdu/Arabic script
- Kannada + Latin mixed script

This is a **catch-all negative/control class**, not a multilingual language-identification benchmark. The current audit is useful for checking that non-target scripts are not accidentally assigned to Hindi classes. A remaining limitation is that the `OTHER` set is mostly script-identifiable; additional Latin-script non-Hindi examples could be added later if the study needs to test that harder boundary.

### Human adjudication cases

The current 100-example audit marks 10 rows for explicit human attention:

1. `L14` — `refund abhi tak kyu nahi aaya` → provisional `ROM_HI`. Hindi syntax is clear, but `refund` is an English lexical item. The decision should follow the project's Hindi-dominant rule.
2. `L29` — `UPI से पैसे कट गए लेकिन payment fail` → provisional `DEV_HI`. This is mixed-script and should be resolved according to the frozen `DEV_HI` definition rather than intuition alone.
3. `L53` — `otp aa gaya lekin login nahi hua` → provisional `ROM_HI`. Useful boundary case because English-domain terms occur inside Hindi syntax.
4. `L54` — `refund ka status kya hai` → provisional `ROM_HI`. Useful EN/HINGLISH boundary case.
5. `L55` — `document nahi mil raha` → provisional `ROM_HI`. Very short query; useful for checking token-share/threshold behavior.
6. `L65` — `mera request reject ho gaya reason kaise pata chalega` → provisional `HINGLISH`. Mixed English terms plus Hindi grammar make this a useful ROM_HI/HINGLISH boundary.
7. `L88` — `मेरा खाता open नहीं हो रहा` → provisional `DEV_HI`. Mixed Devanagari/Latin case for implementation behavior.
8. `L89` — `ನನ್ನ PF status pending ಇದೆ` → provisional `OTHER`. Kannada + Latin mixed-script case and useful non-target-script control.
9. `L99` — `मेरे खाते का स्टेटस pending है` → provisional `DEV_HI`. Devanagari-dominant with Latin `pending`; useful script-dominance boundary.
10. `L100` — `मेरा account बंद हो गया है` → provisional `DEV_HI`. Devanagari with a Latin English noun; useful mixed-script boundary.

The remaining 90 examples have provisional labels assigned for audit use; the CSV records the provisional label, review status, review priority, comments, and an empty `human_final_label` field for human confirmation.

## Recommended review procedure

1. Human reviewer confirms or changes the 10 explicitly flagged cases first.
2. Human reviewer may spot-check the remaining 90, especially any example that feels inconsistent with the frozen definitions.
3. Once human decisions are frozen, rerun the LID tests against the finalized audit labels.
4. Treat any changed label as a data/audit-set change, not as an implementation change, unless the frozen class definition itself is amended.

## Phase 1 gate status (agent update, 2026-09-28; D-041 to D-045)

**Neither human gate is passed. Phase 3 must not start.**

1. **Engineering tooling is complete:** review-safe checker (`--sheet/--out`), deterministic replacement draw (`--exclude`), LID adjudication builder.
2. **Scenario sheet: ChatGPT-assisted, awaiting human approval.** The review artifact `scenario_sheet_chatgpt_draft.csv` now has, for every row: `review_action`, `review_status`, `previous_chunk_id`, `previous_scenario` and `agent_flag`. **Every row is `pending_human_approval`.** The official `data/hiaq/scenario_sheet.csv` is untouched (sha256 `28176bc2…`).
3. **4 professional-role rows need skip and replacement** (S017, S025, S136, S147). Replacement chunks were selected by the deterministic draw (seed 17, same service, same filter):

   | Row | Replacement chunk | Status |
   |---|---|---|
   | S017 | `incometax/1fa8efc9b1cd/002` | ChatGPT: ACCEPT |
   | S136 | `incometax/306ec57bb9dd/004` | ChatGPT: ACCEPT |
   | S147 | `incometax/d51b96a53e6a/010` | ChatGPT: ACCEPT |
   | S025 | `incometax/6cebf3b156cc/013` → ChatGPT: REJECT, then re-drawn with `--exclude` → `incometax/6cebf3b156cc/003` | agent: suitable (an employee telling the employer their tax regime) |

   The 4 replacement scenarios are **agent drafts (Claude Code)**, pending human approval.
4. **7 rejected scenarios have proposed ChatGPT rewrites, pending approval** (S005, S007, S067, S090, S091, S096, S137).
   - Checker: S090, S096 and S137 now pass. S005 (contractor), S007 (job/jobs), S067 (lump) and S091 (military) **still fail the rare-term rule**, on everyday words. They were not reworded to beat the checker.
   - Agent support flags: S005 ("how you can check" only indirectly supported); **S090 drops the "nominate someone outside the family" need**; **S091 omits that the deceased was also a PF/EPS member**; **S096 is employer role-play**.
5. **2 WARNs remain under human review:** S060, S066 (kept provisionally).
6. **The 100-case LID audit exists; `human_final_label` is NOT finalised** (empty for all 100). ChatGPT recommendations are in `langid_chatgpt_recommendations.csv`; they are **not** human labels. Adjudication table: `langid_adjudication_table.csv` (10 flagged). **L54 and L55:** ChatGPT recommends HINGLISH, whereas the provisional label is ROM_HI.
7. **Ethics metadata is incomplete** (see the placeholder below). Only the fact that approval was obtained is known.
8. **Phase 3 must not start.**

Checker on the review artifact: `scenario_sheet_chatgpt_draft_check.csv` gives **144 PASS / 4 REJECT / 2 WARN**. No official check file was created.

### Ethics approval: metadata placeholder (to be filled by the team; do not invent)
| Field | Value |
|---|---|
| Approver name | **Not recorded: waived by the team lead, 2026-09-29 (D-055).** |
| Approver role | **Not recorded: waived by the team lead, 2026-09-29 (D-055).** |
| Approval date | **Not recorded: waived by the team lead, 2026-09-29 (D-055).** |
| Scope | **Not recorded: waived by the team lead, 2026-09-29 (D-055).** The rating study (E8) coverage is not documented. |
| Conditions | **Not recorded: waived by the team lead, 2026-09-29 (D-055).** |

After the scenario sheet and LID audit are cleared:

1. Run `ha data hiaq-check`.
2. Resolve all WARN/REJECT cases.
3. Run `ha data hiaq-variants`.
4. Build the 25 survey forms.
5. Pilot with 5 people.
6. Run `ha data hiaq-ingest` on the pilot export.
7. Proceed to survey launch and later P3/P4 stages.

Do not start Phase 3 generation or Phase 4 judging/calibration while the scenario sheet and LID audit remain unresolved.

---

## Update 2026-09-28 (D-051; appended, the text above is unchanged)
- **Scenarios approved by the team and applied to the draft sheet:**
  - replacements S017, S025 (re-draw `…/003`), S136, S147, using the team's texts, not the agent drafts;
  - rewrites S005, S007, S067, S096, S137.
- **Checker on the draft:** 147 PASS / 2 WARN / 1 REJECT (S091).
- **Still pending:**
  - S090 and S091: needs-preserving texts required;
  - S060 and S066: WARN review;
  - the 137 `keep_draft` rows;
  - the copy to the official sheet (team).
- **LID:** 10 flagged (L14, L29, L53, L54, L55, L65, L88, L89, L99, L100); `human_final_label` still empty for 100/100.
- **Ethics metadata:** the placeholder above is still unfilled.

## Update 2026-09-29 (D-052; appended)
- **LID:** `human_final_label` is final for 100/100 (team). Only L54 and L55 differ from provisional (→ HINGLISH).
- **Scenarios:** 150/150 approved by the team, including S090 and S091 with the team's exact texts. Copied to `data/hiaq/scenario_sheet.csv` (sha256 `4968e202…`).
- **Official check:** 146 PASS / 2 WARN / 2 REJECT.
  - S091 is a documented §6.3 exception.
  - S090 ("unmarried") is awaiting team confirmation of the same exception.
  - The variant builder is blocked until an exception mechanism is approved.
- **Ethics metadata:** still unfilled (placeholder above).

## Update 2026-09-29 (D-053; appended)
- The S090 exception was approved by the team, alongside S091.
- Allowlist: `configs/scenario_check_exceptions.yaml` (S090, S091 only; fails closed).
- `ha data hiaq-variants` built 25 variants (`data/hiaq/survey_variants.json`, sha256 `e125c5c2…`).
- **Still to do:**
  - ethics metadata (placeholder above);
  - the forms, the pilot and the launch (team; not started).

## Update 2026-09-29 (D-055; appended)
- **Ethics metadata:** the team lead decided not to record it (informal approval). The five fields above are marked waived; nothing was invented.
- **Next step:** the 5-person pilot (`docs/survey_protocol.md` §3 step 4).
