# Paper source and build

`main.pdf` is the supplied final ten-page paper. `main.tex`, `appendix.tex`, `references.bib`,
`figures_own/` and `tables_own/` are imported from the v7 source package without changing the paper.

## Rebuild figures and tables

From the repository root, run `bash reproduce/from_outputs.sh`.
For the four individual commands, see [make/README.md](make/README.md).
Three tables are generated; `tables_own/relocation_explanations.tex` is a hand-written summary.
Its evidence is mapped in [docs/reproduce.md](../docs/reproduce.md).
`tables/appendix/paired_differences.tex` is the original source for the paper's paired table.

## Compile

Get `neurips_2025.sty` from the [official NeurIPS 2025 style ZIP](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip),
linked by the [2025 call for papers](https://neurips.cc/Conferences/2025/CallForPapers).
Place the style file in this directory locally. It is third-party material and is ignored by Git.
Install a LaTeX distribution with the packages requested by `main.tex` and run:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expect 10 pages and no undefined citations or references. Compile in a copy to keep the supplied PDF unchanged.
The source uses the retained repository figures through its existing graphic paths.
