# HiAQ-Services survey protocol

This protocol turns spec §6.3, Appendix D and prereg §3.1 into steps. The team runs it. The agents provide the tools.

## 0. Before launch (human)
- **Ethics (H-04):** faculty-advisor approval was reported as obtained (D-041). The team lead decided it was informal and **not to record** the approver, date, scope or conditions (D-055, 2026-09-29). This is the `docs/decisions.md` record; the pilot and launch no longer wait on it. The paper may say only "informal approval by the faculty advisor".
- **Consent.** The consent paragraph is the Appendix D intro, which is already in every form.
- **No identifiers.** The forms collect no email, need no sign-in and have no name fields.

## 1. Writing scenarios (H-05; the team lead writes all 150: 75 Income Tax + 75 EPFO, D-038)

Open `D:\humanalign-rag\data\hiaq\scenario_sheet.csv` in Excel or Google Sheets (it is UTF-8). Each row has a help-page chunk. Write one situation in the `scenario` column. The agent never writes or substitutes scenarios.

**Rules for a scenario**
- Write 1–2 sentences in everyday English, in the second person ("You …").
- Describe **the situation or problem the person is in**, not the answer, and not what the page says.
- Use only words a person would know **before** reading the help page. No form numbers, section names, menu paths or official terms that appear only in the document (for example "Form 26AS", "Refund Reissue", "EDLI", "e-Verify").
- Don't mention any language or Hinglish. People will write in whatever language they naturally use.
- If a chunk is unusable (too technical for an ordinary person, only a menu, not a real need), leave `scenario` empty and write `skip` in `notes`. The agent can then draw a replacement **chunk** (never a scenario) of the same service, with the same seed rule, and log it.

**Good and bad**

| Chunk about… | ❌ Leaky | ✅ Natural |
|---|---|---|
| Refund reissue after a failed refund | "You need to submit a Refund Reissue request because your refund failed." | "Your tax refund bounced because your old bank account is closed, and you want the money sent again." |
| Transferring PF to a new employer | "You want to transfer your EPF balance using the UAN portal." | "You changed jobs and want the savings from your old company's salary deductions moved to the new one." |
| Jeevan Pramaan life certificate | "You need to submit a Jeevan Pramaan digital life certificate." | "Your grandfather's pension will stop unless he proves every year that he is alive, and he can't travel to the office." |

## 2. Automatic check (agent tool, run as often as you like)

```
ha data hiaq-check      -> data/hiaq/scenario_check.csv
```

| Status | Meaning | What to do |
|---|---|---|
| **REJECT** | It shares a rare document term with the chunk, or BM25 finds its chunk first by a margin above 2× (spec §6.3) | Rewrite by hand and re-check |
| **WARN** | At least 60% of its content words appear in the chunk (D-036) | Reread it. If it echoes the page's wording, rewrite it; otherwise keep it |
| **PASS** | — | Nothing |

Every scenario must be PASS or WARN (reviewed) before the forms are built.

## 3. Build the forms (agent tool, then human)

1. `ha data hiaq-variants` deals the 150 scenarios into 25 forms of 6 (each scenario used once, both services in every form, seed 17) and writes `survey_variants.json`.
2. A team member opens script.google.com and pastes `tools/make_forms.gs`. They replace the placeholder object after `var VARIANTS =` with the full content of `survey_variants.json` (no quotes), then run `createAllForms()`.
   - The script refuses a wrong or partial paste before creating anything.
   - Forms are created **not accepting responses**; the optional questions are on page 2.
   - Copy the whole execution log to `data/hiaq/forms_creation_log.txt` and run `ha data hiaq-forms-manifest`. It verifies all 25 forms and writes `data/hiaq/forms_manifest.json` (D-054).
3. Paste the URLs into `tools/redirect.html` and host it on **GitHub Pages** (team decision, 2026-09-27). The 25 Google Forms are the actual survey endpoints. This single link is what you share.
4. **Pilot** with 5 people (D-055). The redirect page and GitHub Pages are **not** needed for the pilot.
   1. **Open 5 forms:** e.g. forms 01–05. Take each form's edit URL from `data/hiaq/forms_manifest.json`, open the **Responses** tab, and switch **Accepting responses** on. Leave the other 20 closed.
   2. **Send each pilot person one different response URL** (`response_url` in the manifest). Ask them to answer naturally, with no extra instructions.
   3. **Close the 5 forms** again (Accepting responses off).
   4. **Export:** for each form, Responses → ⋮ → **Download responses (.csv)**. Unzip it and save it as `data/hiaq/pilot_responses/form_NN.csv`, where NN is the form number (all forms share one title, so rename each file).
   5. **Ingest:** run `ha data hiaq-ingest --pilot`. It writes `data/hiaq/pilot_queries.parquet`, never the dataset file. Check that each answer is mapped to the right scenario.
   6. **Before launch:** in each pilot form, Responses → ⋮ → **Delete all responses**, unless the team decides pilot answers join the dataset. Otherwise they reappear in the launch export.

## 4. Distribution (H-06)
- The target is at least 100 respondents (at least 600 queries), both RVCE departments and non-students (family, staff), for a wider range of ages and typing habits.
- **Language balance (plan I-9):** include people who normally type Hindi, to get at least 150 ROM_HI or HINGLISH queries. **Never** ask people to type in Hinglish; the mixing must be natural.
- Push hardest in weeks 3–5. The fallback checkpoint (fewer than 300 queries) is at the end of week 6.
- Tell the agent when about 150 responses are in (H-07). The early tranche supplies the 30 HiAQ CAL-auth items.
- **Superseded (D-059, D-060): authentic HiAQ collection is closed at zero; the forms are not used, and the §8 fallback is recorded as A1.** The two bullets below are the earlier plan.
- *(Earlier plan)* **Initial authentic tranche (D-057):** the Google Forms collect the first tranche for calibration. The target is **25 completed forms (up to 150 scenario answers)**.
  - **One form per respondent,** as in spec §6.3 ("each respondent gets 6 random scenarios"). The forms cannot enforce this, because enforcing it would require sign-in. Each person gets one link and is asked to fill it once.
  - No deviation allowing several forms per person has been adopted. One would need a new dated decision, because ingest treats every submission as a separate respondent.
- **Fallback (prereg §8):** if, after the §3.1 exclusions, HiAQ has fewer than 300 eligible authentic queries from the forms, H2 and H3 are evaluated on D2 alone and D4 is reported as a pilot. This is recorded as an amendment before `heldout_A` is unsealed (recorded as A1, 2026-10-10). Collaborative-sheet answers (§6) never count.

## 5. Ingest and cleaning
- Export each form's responses to CSV as `form_XX.csv` in `D:\humanalign-rag\data\hiaq\responses\`.
- `ha data hiaq-ingest`:
  - removes PII (phone, email, 12-digit Aadhaar-like numbers and PAN become placeholders);
  - normalises whitespace only; **typos are kept**;
  - attaches the language class and CMI;
  - prints weekly counts by language class.
- **Exclusions (prereg §3.1), human, blind to any method output:** two team members mark answers that are empty, not a query, or off-topic for their scenario, with a reason. These are dropped before any split.

## 6. Collaborative sheet (D-056; a SEPARATE labelled set, not the authentic survey)
**Status (D-057): exploratory only. Do not distribute the sheet yet.** Its answers never enter HiAQ's R, A, calibration items or any confirmatory or secondary test, and do not count toward the 300-query threshold of prereg §8.

1. `ha data hiaq-collab-sheet` writes `data/hiaq/collab/collab_sheet_template.csv` (600 rows = 100 blocks × 6).
2. Import it into Google Sheets (File → Import → Upload). Share it with `data/hiaq/collab/HOW_TO_FILL.txt`.
3. **Contributors:** each claims an empty block by writing their code (C01, C02 …) and types their own answers.
   - Several blocks per person are fine, but **never two blocks of the same form**.
   - No names; no examples; never mention a language.
   - The team never types for anyone and never edits answers.
4. **Independence:** contributors can see each other's rows. Use per-person filter views or copies where you can.
5. Download as CSV to `data/hiaq/collab/collab_sheet_filled.csv` and run `ha data hiaq-collab-ingest`. It refuses an edited or inconsistent sheet and writes only `collab/collab_queries.parquet` (`source = collab`).
6. This set stays out of HiAQ's authentic data, the splits and the main test (prereg H2) unless the team decides otherwise in a dated decision.
