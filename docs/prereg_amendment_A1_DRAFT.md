# Pre-registration amendment A1: HiAQ-Services fallback (rev 3, FINAL TEXT FOR APPROVAL)

> **Status: RECORDED 2026-10-10 in `docs/preregistration.md` under `## Amendments` (D-060), approved by the project lead (Chinmai Shetti).** The block below is the approved text (sha256 `6194857e…`). The recorded version differs only in its filled date and name placeholders (sha256 `8b91f44d…`).
> - The frozen body and `results/PREREG.json` are unchanged.
> - Versioning (team decision 2026-10-10): **prereg-v2 = prereg-v1 + A1 + A2**.

<!-- BEGIN A1 TEXT -->
### A1 · [recording date] · HiAQ-Services has no eligible authentic queries (§8 "Too few survey responses")
1. **Condition and timing.**
   - On 2026-10-10 the team closed authentic HiAQ collection for this study with zero eligible authentic queries (D-059); no survey responses were collected.
   - Zero is fewer than 300, so the §8 contingency applies.
   - This amendment is recorded before `heldout_A` is unsealed, and together with A2, before calibration annotation starts.
2. **H2 and H3** are evaluated on D2 (CQADupStack-Android) alone. The §7 procedure is unchanged with d ∈ {D2}:
   - Δ̄_B = mean over seeds s = 0…4 of [τ_b(M1, A) − τ_b(B, A)];
   - B = 1,000 query-level bootstrap iterations, seed 20260927;
   - H2 is supported if and only if p_B0 < 0.05 and p_B1 < 0.05;
   - H3 uses M1-iso as the comparator, one-sided α = 0.05.

   D4 is reported as a pilot. With no authentic D4 queries, no D4 agreement statistic can be computed, so the pilot report is limited to the HiAQ corpus, scenarios, synthetic pools and their descriptive statistics. This invokes a pre-registered contingency and is not a methodological change.
3. **H1.** The wording is unchanged.
   - D4's three tests (length KS, gold-passage overlap KS, topic JS permutation) cannot be computed. They stay in the Holm family of 3 datasets × 3 measures (m = 9) as not rejected (p = 1), so the family size and the Holm thresholds are those of §2.
   - For the τ_b criterion, D4 counts as not meeting τ_b(B0, A) < 0.8. The criterion is met only if both D1 and D2 have seed-averaged τ_b(B0, A) < 0.8, each reported with its CI.
4. **H4** (D4 only) is not estimable and is reported as such.
5. **D4 qrels.** No D4 qrels are built, since there are no authentic D4 queries to score, and the §8 qrels fallback does not apply. D4 synthetic queries keep their source-passage qrels (§5.4) for pilot descriptives only.
6. **Calibration** is as amended by A2.
7. **Excluded from every confirmatory and secondary analysis (H1–H4), and from calibration and anchors:**
   - collaborative-sheet answers (D-056);
   - the external synthetic set `X-team150` (D-058);
   - the workbooks quarantined on 2026-10-10.

   They may appear only in analyses labelled exploratory (§2, §7.7).
8. **Everything else is unchanged.**

Recorded by: [team member 1], [team member 2]
<!-- END A1 TEXT -->

## Reviewer notes (not part of the amendment text)
- **Rev 3:** the text is frozen for approval and is identical to the proposed append. It is unchanged from rev 2 except for the wording of item 7 (the same three exclusions, now also naming calibration and anchors).
- **Rev 2 versus rev 1:**
  1. The condition is now established (team decision: zero), not conditional on a future count.
  2. The D4 pilot is reduced to descriptives, because no D4 agreement statistic exists.
  3. **H1 Holm: the frozen m = 9 family is preserved**, with the D4 tests as non-rejections. Rev 1 shrank the family, which was a procedure change.
  4. H4 is not estimable.
  5. D4 qrels are not needed. Rev 1's pre-selection of the 30% fallback is removed, resolving review finding 2.
  6. Calibration refers to A2.
  7. The exclusions are extended (`X-team150`, quarantined workbooks).
  8. Rev 1's eligibility wording ("one form per respondent") and its tranche-timing clauses are removed as moot.
- **Holm detail:** with p = 1 entries, the step-down compares the k-th smallest real p-value against α / (9 − k + 1). These are exactly prereg-v1's thresholds for the smallest p-values, so nothing is made easier to reject.
- **"Reported as a pilot":** the §8 text presumes some D4 data. With zero, reading the pilot as descriptives only is an **interpretation**, recorded here explicitly.
- **Not reconciled, noted:** the build spec's own fallback ("D1–D3 become the main datasets", spec lines 706 and 787) predates the frozen prereg, and D3 is not in the core. The frozen prereg governs.
- **D-027** ("pooled over D2 + D4; HiAQ stays core") is superseded for the confirmatory test once A1 is recorded. HiAQ remains in the paper as a corpus, scenario set and synthetic-pool pilot.
