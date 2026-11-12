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

## Editorial revision v9

The author-requested revision is under [v9/](v9/README.md), with separate source, presentation builders and
claim/evidence notes. It incorporates the Appendix A wording fixes while retaining the original scientific
definitions and results. The v8 files and the rebuild checks above continue to describe the supplied v8 package.

## Submission manuscript v10

The [v10 source and build notes](v10/README.md) integrate the author-approved diagnostic/correction figure,
preserve the homepage fast-weight schematic, and update all affected definitions, captions and references.
The [complete submission PDF](../output/pdf/attention_fast_weights_paper_v10.pdf) has four main pages,
two reference pages and six appendix pages. Original experiments and registered decision rules are unchanged.

## Reviewed submission manuscript v11

The [v11 source and build notes](v11/README.md) implement the confirmed [v10 review](reviews/v10_submission_review.md),
with unchanged homepage/Figure 2 vector assets, clarified methods and reproduction details, and an isolated
bibliography. Unconfirmed reference entries remain unchanged. The [complete PDF](../output/pdf/attention_fast_weights_paper_v11.pdf)
contains four main pages, two reference pages and eight appendix pages. All original results and registered rules are preserved.

## Framework-focused submission manuscript v12

The [v12 source and build notes](v12/README.md) describe the author-approved rewrite:
*Interpreting ViT Attention Maps Through a Fast-Weight Lens*. A 113-word abstract and
three related-work themes support the analytical framework of reading assumptions,
diagnostic proxies, corrections and matched controls. Parameter randomisation is in
Additional results (B.1/Table B.3); Key parts of the code (E) ends the paper. The
[complete PDF](../output/pdf/attention_fast_weights_paper_v12.pdf) contains four main
pages, two reference pages and nine appendix pages. Original experiments, registration,
homepage/Figure 2 vector assets and the entire v11 bibliography are unchanged.
