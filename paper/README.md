# Paper

- `paper.md`: the full draft. Its numbers come from `results/` and are traced in `docs/PAPER_LOG.md`.
- `figures/fig1_tau_by_dataset.{png,pdf}` and `tables/table{1,2,3}_*.csv` are generated, never edited by hand:

  ```
  python tools/make_paper_assets.py
  ```

  The script reads `results/eval/*.json`. Rerun it after any result changes.
- `sections/`: empty for now. Split `paper.md` here if the paper moves to LaTeX.
