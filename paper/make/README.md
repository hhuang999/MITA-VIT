# Build the paper figure and tables

Run from the repository root:

```bash
python -I paper/make/make_main_table.py results paper/tables_own/main_results.tex
python -I paper/make/diag_table.py results/diagnose.csv paper/tables_own/diagnostics_per_layer.tex
python -I paper/make/make_paired_table.py paper/tables paper/tables_own/paired_differences.tex
python -I paper/make/make_diagnosis_figure.py results paper/figures_own/diagnosis.pdf
```

These scripts round or copy existing results. They do not run inference.
The paired source is first rebuilt by `scripts/make_latex_tables.py`.
The fourth table, `tables_own/relocation_explanations.tex`, is hand-written and remains unchanged.
See [the reproduction map](../../docs/reproduce.md) for every input, including its four supporting analyses.
