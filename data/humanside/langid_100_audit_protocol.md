# HumanAlign — 100-example LID audit

## Purpose
This is an engineering-validation/audit set for the frozen rule-based language-ID implementation. It is **not** a representative language dataset and must not be used to tune the HumanAlign judge, retrieval systems, or held-out evaluation sets.

## Composition
The audit contains the original 30 hand-labelled examples plus 70 additional boundary-focused examples.

The provisional labels are balanced approximately around the five classes, but the frozen class rules take precedence over exact balance. In particular, mixed-script Devanagari cases remain useful `DEV_HI` audit cases rather than being re-labelled to preserve counts.

## Frozen class definitions from the project spec
- `EN`: Hindi-token share < 0.1
- `ROM_HI`: Hindi-token share > 0.8 and Latin script
- `HINGLISH`: Hindi-token share 0.1–0.8
- `DEV_HI`: Devanagari-script text
- `OTHER`: languages/scripts outside the target classes

Sensitivity checks are performed at ±0.05 on the numeric thresholds; this audit should be interpreted against the frozen implementation/configuration.

## Human-review procedure
1. Review all rows marked `HUMAN_REVIEW` first.
2. Then inspect a sample of the `LIKELY` rows, especially short queries and rows with common English loanwords such as `refund`, `payment`, `password`, `login`, `status`, and `form`.
3. In `human_final_label`, enter the final human decision only after considering the frozen definitions.
4. Use `adjudication_note` for disagreements, especially where the disagreement comes from script handling or token-level Hindi-share thresholds.
5. Do not change `current_label` for the original 30 rows. That column preserves the original implementation/test label.
6. Once the team has agreed, freeze the final labels as a data artifact and report human-vs-system agreement and the disagreement cases.

## Priority boundary cases
The deliberately difficult cases include:
- Romanised Hindi containing several common English/domain words (`L53–L55`)
- mixed Romanised Hindi/Hinglish (`L65`)
- Devanagari plus Latin text (`L29`, `L82`, `L88`, `L99`, `L100`)
- non-Hindi Indian scripts (`L83–L98`)
- short queries where token thresholds are sensitive (`L55`)

## Recommended paper wording
Describe this as a **100-example hand-labelled LID audit/validation set**. Do not call it a statistically representative sample of Indian-language queries.
