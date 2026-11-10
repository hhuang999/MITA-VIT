# Paper v8 source and build

`main.pdf` is the author's supplied 11-page v8 paper. Its PDF, `main.tex`, `appendix.tex`, bibliography,
figures and tables are synchronised from `paper_draft_v8.zip`. The manuscript text is preserved.
Generator comments and build instructions are maintained with the code.

`bash reproduce/from_outputs.sh` rebuilds summaries, the paired source, generated tables and figures on CPU
into fresh `build/rebuilt/`; it never overwrites this directory's authored assets. See [make/README.md](make/README.md)
and the [v8 evidence map](../docs/reproduce.md). Table 2, Table B.1 and Table B.2 are generated. Tables 1 and A.1
are inline authored summaries; `tables_own/relocation_explanations.tex` (Table C.1) is authored from its source CSVs.

## Optional LaTeX compilation

Obtain `neurips_2025.sty` from the [official NeurIPS 2025 styles](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip)
under its original licence. It is ignored, not redistributed here. In a temporary copy, place it under `paper/`,
install the packages named by `main.tex`, then run:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expect 11 pages and resolved citations/references. The supplied PDF is usable without compiling it.
The repository does not change scientific definitions to remove wording differences: the two proposed Appendix A
precision fixes are recorded for the author in `docs/reproduce.md`.
