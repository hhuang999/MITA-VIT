# Per-image outputs of the original runs

One folder per config and model (`outputs/<config>/[<dataset>/]<model>/`), as written by `scripts/run.py`. These are the per-image tables and arrays behind every number in `results/`.

- **Rebuild without a GPU.** `bash reproduce/from_outputs.sh` rebuilds `results/`, `figures/` and `paper/tables/` from these files, byte for byte (checked on 2026-11-09).
- **Where the files come from.** They come from the study's original runs (NVIDIA L4, and an A100 for parts of three reused runs), converted to the layout and method names of this package:
  - stand-ins renamed `stand_in_*`;
  - the k sweep stored under `k_sweep`;
  - the per-layer high-norm and ClickMe values taken from the earlier per-layer runs.

  That is why each `checks.json` reads `{"complete": true, "imported": true}` and the run metadata are not repeated here; the code commits of the original runs are listed in `docs/preregistration.md`.
- **Rerun from scratch.** `bash reproduce/full.sh` writes to `outputs/rerun/`, which Git ignores. A full rerun from a fresh clone on 2026-11-08 reproduced every summary table within tolerance and every verdict exactly (`docs/verification/rerun_2026-11-08.csv`).
- **Size.** About 574 MB, 326 files; no file is larger than 50 MB.
