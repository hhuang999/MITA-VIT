# Build paper v8 figures and tables

From the repository root run `bash reproduce/from_outputs.sh`. Outputs go to fresh `build/rebuilt/`.
That command checks original evidence first and compares all rebuilt numeric tables with reference results.

| Paper item | Builder | Inputs |
| --- | --- | --- |
| Figure 1b / D.1 | `scripts/make_figures.py` / `make_example_maps.py` | Fixed seeded DeiT-S ClickMe gallery; first two / next four indices |
| Figure 2 | `make_diagnosis_figure.py` | `results/diagnose.csv` |
| Table 2 | `make_main_table.py` | `table1_long.csv`, `correct_tests.csv`, `variants.csv` |
| Table B.1 | `diag_table.py` | `diagnose.csv` |
| Table B.2 | `make_paired_table.py` | Paired-difference source from `scripts/make_latex_tables.py` |

Tables 1 and A.1 (inline LaTeX) and Table C.1 (`relocation_explanations.tex`) are authored summaries, not generated
numeric tables. The [evidence map](../../docs/reproduce.md) identifies their tests and post-hoc status.
Figures 1a and B.1 use the repository's unchanged schematic and execution inputs. The v8 examples use indices
47, 287 / 712, 1308, 3048, 4691 and emit a JSON selection record. Individual builders accept explicit destination
paths; use a temporary directory so authored v8 assets remain unchanged.
