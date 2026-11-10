# Generated LaTeX tables

`tables/` contains the LaTeX exports of the CSV summaries in `../results/`. The directory name is retained because
the reproduction scripts write to this path.

```bash
python scripts/make_latex_tables.py
```

Run the command from the repository root. To rebuild the summaries, figures and LaTeX exports together, use
`bash reproduce/from_outputs.sh`.

| Path | Contents |
| --- | --- |
| `tables/table1.tex` | Main comparisons: object enrichment and ClickMe alignment |
| `tables/table1_full.tex` | All three metrics, intervals, references and correction orders |
| `tables/appendix/*.tex` | Supporting checks, diagnostics and exploratory comparisons |
| [`naming.md`](naming.md) | Internal experiment identifiers and descriptive labels |

Each `.tex` file contains a table body and comments identifying its source CSV files, interval convention and
verdict legend. Include it in a LaTeX document with `booktabs`, `makecell`, `amsmath` and `amssymb`.
