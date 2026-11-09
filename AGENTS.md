# Guide for continuing this project

## State

- All experiments are finished. Their per-image outputs are in `outputs/`, their summary tables in `results/`, their figures in `figures/` and the LaTeX tables in `paper/tables/`.
- `bash reproduce/from_outputs.sh` rebuilds `results/`, `figures/` and `paper/tables/` from `outputs/` byte for byte, on a CPU.
- A full rerun from a fresh clone reproduced every table and every verdict (`docs/reproduce.md`).
- The paper text is still to be written. Everything needed for it is in `paper/` (start with `paper/README.md`).

## Where things are

| Need | Look at |
| --- | --- |
| What each experiment does | the docstring of its module in `src/fastweight_vit/experiments/` and the comment at the top of its config in `configs/` |
| Which config produces which result, figure or table | `docs/reproduce.md` |
| Pre-registered predictions, decision rules and outcomes | `docs/preregistration.md`, `results/predictions.csv` |
| Every number with its interval and verdict | `results/*.csv` (built by `scripts/make_tables.py`) |
| Checked headline numbers and the correct wording | `paper/key_facts.md` |
| Names to use in the paper vs internal codes | `paper/naming.md` |
| Complete factual record (definitions, caveats, deviations) | `paper/evidence_dossier.md` |
| The research history (decision log in Chinese, earlier runs, earlier drafts) | branch `study/v5`; not needed to continue |

## Commands

```bash
python -m pytest -q                                       # 78 tests, CPU
bash reproduce/from_outputs.sh                            # results/, figures/, paper/tables/ from outputs/ (CPU)
python scripts/make_tables.py                             # results/ from outputs/
python scripts/make_figures.py                            # figures/ from results/ (+ outputs/ for the image panels)
python scripts/make_latex_tables.py                       # paper/tables/ from results/
python scripts/run.py --config configs/<config>.yaml      # one experiment (GPU)
bash reproduce/full.sh                                    # rerun everything into outputs/rerun/ (about 12 GPU hours on one L4)
```

## Rules

- **Credentials.** `HF_TOKEN` is read only from `.env` or the environment; never print it.
- **What stays out of Git.** Data, model weights, new reruns and smoke tests (`outputs/rerun/`, `outputs/smoke/`), `.env` and third-party files (for example the NeurIPS style file) are not committed. The per-image outputs of the original runs (`outputs/<config>/...`) are committed and must not be overwritten.
- **Clean code before runs.** `scripts/run.py` refuses to run on uncommitted code changes, because every output records its git commit. Commit first.
- **Verdicts.** Registered verdicts are not changed. A new analysis is exploratory: write its definition and decision rule down (commit) before running it, and label it as exploratory.
- **Figures and tables.**
  - No internal codes (R0, C1–C3, M1–M3, P1–P7) in figures or tables; use `paper/naming.md`.
  - No figure numbers, panel letters or captions inside figures.
  - One panel per file, minimum font 7 pt.
- **Pushing.** Push to GitHub only when the project owner asks for it.
