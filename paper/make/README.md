# How the figure and tables made for this paper were produced

All scripts only read files of the repository and write a figure or LaTeX; they compute no new numbers
(values are rounded or copied). Run them from this folder, with the repository layout
(`results/` and `paper/tables/` as in the repository):

```bash
python -I make_diagnosis_figure.py ../../results ../figures_own/diagnosis.pdf        # Figure 2
python -I make_main_table.py       ../../results ../tables_own/main_results.tex      # Table 1
python -I diag_table.py ../../results/diagnose.csv ../tables_own/diagnostics_per_layer.tex   # Table B.1
python -I make_paired_table.py     ../tables ../tables_own/paired_differences.tex    # Table B.2
```

- Figure 2 (`figures_own/diagnosis.pdf`): the four per-layer panels of `results/diagnose.csv` in one
  figure with a shared legend (5.5 in wide, fonts 7 pt or larger, TrueType fonts). The image holds no
  panel letters or titles; the labels (a)-(d) are set in LaTeX above the image (`\plabel` in `main.tex`).
- Table 1: `results/table1_long.csv` (values), `results/correct_tests.csv` and `results/variants.csv`
  (higher/lower marks).
- Table B.2: seven blocks copied from `paper/tables/appendix/paired_differences.tex`; the script
  checks that every number occurs in the source.
- `tables_own/relocation_explanations.tex` (Table C.1) is written by hand from the evidence dossier.

Figure 1, Figure B.1 and Figure D.1 are the repository's own figures (`figures/`).
