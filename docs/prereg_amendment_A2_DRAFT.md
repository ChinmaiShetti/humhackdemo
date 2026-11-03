# Pre-registration amendment A2: calibration without authentic HiAQ items (released as v2) (rev 3, FINAL TEXT FOR APPROVAL)

> **Status: RECORDED 2026-10-10 in `docs/preregistration.md` under `## Amendments` (D-060), approved by the project lead (Chinmai Shetti).** The block below is the approved text (sha256 `6194857e…`). The recorded version differs only in its filled date and name placeholders (sha256 `8b91f44d…`).
> - The frozen body and `results/PREREG.json` are unchanged.
> - Versioning (team decision 2026-10-10): **prereg-v2 = prereg-v1 + A1 + A2**.

<!-- BEGIN A2 TEXT -->
### A2 · [recording date] · Calibration without authentic HiAQ items (§4.1–§4.2); analysis released as v2
1. **Reason.** With no authentic HiAQ queries (A1), §4.1's 30 authentic HiAQ items cannot exist, and its 70 synthetic HiAQ items cannot receive neighbours from a D4 R. §4 cannot be executed as written.
2. **Items: 100.** The HiAQ half is removed, and no item is added or substituted. The remaining items are:
   - the 30 authentic FiQA `CAL-auth` items already drawn under §4.1 (seed 17; in R for every split seed);
   - the 70 synthetic FiQA items drawn under §4.1 (35 P-std + 35 P-persona, seed 17).
3. **Split.** 60 `cal-dev` / 40 `cal-test`, stratified by dataset × authentic/synthetic, seed 17: the same 60% `cal-dev` share as §4.2 (authentic 18/12, synthetic 42/28). Each FiQA item keeps the assignment §4.2 gives it. `cal-test` stays sealed.
4. **Everything else in §4 is unchanged:**
   - neighbours and the practice items;
   - annotation and blinding;
   - κ and the one-time revision with 40 re-annotated `cal-dev` items;
   - the human reference;
   - prompt versions v0–v4 from `cal-dev` only (the D4 Hinglish-anchor clause has nothing to apply to);
   - the selection criterion;
   - NNLS weights for M1 and M1-iso with 5-fold CV;
   - the gate F ≥ 4;
   - permutations;
   - §4.9.

   D2 contributes no calibration items. The data excluded in A1 item 7 are not used.
5. **Versioning (line 3; D-030).** This is a methodological change, so the analysis is released as **prereg-v2 = prereg-v1 + A1 + A2**.
   - prereg-v1's primary analysis cannot be executed, because its calibration set does not exist. It is reported as not executable, with this reason.
   - A2 was recorded on [recording date], **before any calibration annotation or calibration result existed**.

Recorded by: [team member 1], [team member 2]
<!-- END A2 TEXT -->

## Reviewer notes (not part of the amendment text)
- **Rev 3:** item 5 (versioning) is added, and the exclusions now refer to A1 item 7. The text is identical to the proposed append.

### Verification against prereg-v1 §4 (each rule checked against the code)
| prereg-v1 rule | Under A2 | Check |
|---|---|---|
| §4.1: 30 FiQA `CAL-auth`, seed 17, in R for every seed | identical items | `data/processed/fiqa/splits.parquet` `cal_auth` = 30; `ha calib status`: 30/30 passages resolved. FiQA splits are untouched, because `cal_auth.fiqa` stays 30. |
| §4.1: 70 FiQA synthetic, 35/35, seed 17 | identical items | `draw_synthetic(U_fiqa, 70, 17)` is unchanged |
| §4.1: 30 HiAQ authentic + 70 HiAQ synthetic | removed | infeasible (see above) |
| §4.1: neighbours from R of seed 0; no self or cosine > 0.97 | unchanged (FiQA R) | the human display (D-050) removes all `CAL-auth`, now 30 |
| §4.2: stratified split, seed 17 | 60/40 | **The FiQA items get exactly the assignment prereg-v1 would have given them**: one RNG visits the strata in sorted order, and the FiQA strata come first. `tests/test_calibrate.py::test_a2_items_split_and_fiqa_identity_with_v1` checks this. |
| §4.3: 20 practice items, extra FiQA synthetic, outside the set | unchanged | the practice draw excludes the same 70 FiQA items; `data/calibration/sheets/practice.csv` stays valid |
| §4.3: both annotators rate all items, blind, random order | 100 items | guidelines v1 updated (counts and domain only; rubric unchanged) |
| §4.4: κ per dimension; revise once; re-annotate 40 `cal-dev` | unchanged, literally | 40 of the 60 `cal-dev` items (see consequence 3) |
| §4.5–§4.6: v0–v4, 3 anchors from `cal-dev`, ρ excluding anchors | unchanged | ρ over 57 non-anchor `cal-dev` items |
| §4.7: NNLS on H, S, P, C*, all `cal-dev`, 5-fold CV, M1 and M1-iso | unchanged | folds of 12 items (fold seed 17) |
| §4.8: gate F ≥ 4 a priori; permutations | unchanged | — |
| §5.2: N = min(\|A\|, M1 pass, M1-iso pass) | unchanged | — |
| Authentic share of calibration items | 30% (30/100) | the same as prereg-v1 (60/200) |

### Methodological consequences of removing the HiAQ half
1. **Precision.** `cal-dev` halves (120 → 60) and `cal-test` falls from 80 to 40, so κ, ρ and the NNLS weights get wider CIs. The prompt-version choice rests on 57 items.
2. **Domain coverage.** The judge is calibrated on English personal-finance items only and applied unchanged to D2 (English Android Q&A). D2 never contributed calibration items under prereg-v1 either, so the held-out logic is unchanged. Cross-domain transfer is now the only setting and is disclosed.
3. **Re-annotation rule.** "Re-annotate 40 `cal-dev` items" is kept literally; it now covers two-thirds of `cal-dev`.
4. **The C dimension and the weights.**
   - With all-English items, C is null almost everywhere and is imputed as C* = mean(H, S, P).
   - **The NNLS weight on C is therefore not separately identifiable**: only the effective weights w_k + w_C/3 (k ∈ {H, S, P}) are.
   - For English evaluation sets (D1, D2) the score s(q), and therefore every gate and selection, depends only on those effective weights, so selection is unaffected.
   - Report the effective weights.
5. **D4 Hinglish anchors and code-mixing calibration no longer exist.** The claim about code-mixed naturalness (C) becomes untested. This is consistent with A1's H4 being not estimable.
6. **The practice pack exported on 2026-09-28** is unchanged in content: its neighbour pool already excluded exactly the 30 FiQA `CAL-auth`, since no HiAQ `CAL-auth` ever existed. Only its key file's rule label still reads "all 60". A re-export changes only that label.

### Versioning (decided by the team, 2026-10-10)
Prereg line 3 and D-030 require an explicit v2 for a methodological change, and A2 is one: it changes the locked §4 calibration procedure. The analysis is therefore released as **prereg-v2 = prereg-v1 + A1 + A2**.
- prereg-v1's primary analysis is reported as **not executable**, because its calibration set cannot be constructed.
- A2 is recorded before any calibration annotation or calibration result exists.
- A1 alone is a pre-registered contingency, not a methodological change.

### Implementation (uncommitted; for review)
- **`configs/calibration.yaml`** (not hash-locked): `amendment: A2`, `cal_auth.hiaq: 0`, `synthetic.hiaq: 0`, split 60/40. The prereg-v1 values are kept in its header comment.
- **`src/ha/judge/calibrate.py`:**
  - `amendment_recorded(tag)` reads only the `## Amendments` section;
  - `build_items` refuses until the config's amendment is recorded, skips datasets configured with zero items, and still requires exactly 30 FiQA `CAL-auth`;
  - `split_items` is unchanged in logic.
- **`src/ha/human/calibration.py`:** wording updated ("all CAL-auth"); `status()` reports `amendment` and `amendment_recorded`; `main_pack_ready` requires the amendment to be recorded.
- **`docs/annotation_guidelines.md`:** 100 items, FiQA domain only.
- **`tests/test_calibrate.py`:** prereg-v1 regression tests; A2 items and split; FiQA identity with v1; guard refusal; parser.
