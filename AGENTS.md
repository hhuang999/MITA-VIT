# Project guide

## Purpose and layout

This is the course code for **Is It the Model or the Reading? A Fast-Weight View of ViT Attention Maps**,
by Hongbo Huang. The final manuscript is `paper/main.pdf`; its source and build guide are in `paper/`.
`README.md` explains setup. `docs/reproduce.md` maps every paper figure, table and empirical claim to its source.

The active configs and modules support the paper. Experiments outside it are in `archive/`, which Git excludes
from exported course bundles. The archive README gives the original runnable commit.
Some shared experiment modules contain additional stages; keep their computation intact unless separately asked.

## Commands

```bash
python -m pytest -q
bash reproduce/from_outputs.sh
python scripts/run.py --config configs/<name>.yaml --output outputs/rerun
bash reproduce/full.sh
git archive --format=zip -o course_code.zip release/v1
```

Tests and rebuilding use CPU. Experiment reruns require external datasets, checkpoints and a CUDA GPU.
Rerun dependencies are ordered in `reproduce/full.sh`. The historical register evidence in `results/register/`
has a separate provenance note; the current runner does not repeat its old full-validation accuracy check.

## Project rules

- Keep work on `release/v1` for this delivery. Do not merge or change `main` or `study/v5`.
- Preserve the supplied paper text, numbers, figures and tables. Rebuild in a temporary copy.
- Keep original per-image outputs unchanged. Use `outputs/rerun/` or `outputs/smoke/` for new runs.
- Preserve original artifacts when moving them. Archive unrelated tracked files with their relative paths.
- Keep datasets, checkpoints, `.env`, credentials, reruns and third-party style files out of Git.
- Read `HF_TOKEN` from the environment or `.env`; never print it.
- Preserve ignored local writing materials and unrelated untracked files.
- Source changes must be committed before inference. Exports record their commit via `.git_archival.txt`.
- Do not change statistical definitions or verdicts. CPU summaries must match within absolute error 1e-12,
  verdicts must match exactly, and figures must retain their content. Generated paper tables must match byte for byte.
- Names R0, C1-C3 and M1-M3 remain in code and files. The premises are specificity, location and contribution.
- Push only with the owner's authorization. Do not force-push or rewrite history.
