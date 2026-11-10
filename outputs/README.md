# Original per-image outputs

Folders follow `outputs/<config>/[<dataset>/]<model>/`. They contain the original per-image CSV files,
arrays and checks used by the paper summaries. Their contents were not changed during the cleanup.
Imported runs use `checks.json` with `complete` and `imported` flags.

`bash reproduce/from_outputs.sh` rebuilds the paper results on CPU without downloading data or checkpoints.
The input/output map is in [docs/reproduce.md](../docs/reproduce.md).
The two historical register summaries used in Appendix A live in `results/register/` with their source commit.

`bash reproduce/full.sh` writes new runs to ignored `outputs/rerun/`.
`reproduce/smoke.sh` uses ignored `outputs/smoke/`. Do not write new experiments into the original folders.
New runs record their effective config, source revision and software versions.
Shared modules may produce additional per-image columns or analyses; the builders export only paper summaries.
