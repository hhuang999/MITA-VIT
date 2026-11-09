# Paper materials

Everything needed to write the paper *Attention as Fast Weights: Reading, Diagnosing, and Correcting Attention Maps in Vision Transformers* (HKUST(GZ) DSAA 2012 course project). Format: NeurIPS 2025 preprint, 4 pages of main text plus references and appendix. The paper text itself is not written yet.

## Read first

1. **`naming.md`** — the names used in figures, tables and text. It also maps the internal codes used in `results/`, in the code and in the dossier (R0, C1–C3, M1–M3, P1–P7, …) to those names. Internal codes must not appear in figures or tables.
2. **`key_facts.md`** — every headline number and claim, checked against `results/`, with the correct wording (including corrections of an earlier draft).
3. **`../figures/README.md`** — every figure file: content, print size, data source.
4. **`evidence_dossier.md`** — the complete factual record, for anything not in `key_facts.md`:
   - definitions, data and preprocessing;
   - pre-registered rules and outcomes;
   - every table with intervals;
   - deviations, bugs, caveats;
   - experiments not done.
5. **`../docs/preregistration.md`** — predictions, decision rules and when each was fixed.

## Files

| Path | Content |
| --- | --- |
| `../figures/` | All figures, one panel per file (PDF + PNG); built by `scripts/make_figures.py` |
| `tables/table1.tex` | Main results table: enrichment / ClickMe, layers 9–12, each correction followed by its indented stand-in |
| `tables/table1_full.tex` | All three metrics with 95% intervals, all references, both orders of combining the three corrections |
| `tables/appendix/*.tex` | Further appendix tables (single-layer tests, head vs body, predictions, checks, paired differences, model-side variants, last-layer recovery, exploratory analyses) |
| `references.bib` | BibTeX entries for the related work used so far |

All `.tex` tables are built from `results/` by `scripts/make_latex_tables.py`. Each file contains only the `tabular` (plus font-size commands where needed), with comment lines giving a suggested caption and the data source; captions and labels are written in the paper. They need `booktabs`.

## Rules for figures and tables (from the author)

1. No figure numbers, panel letters or captions inside a figure.
2. One panel per file; every panel has its own axis labels, units and legend.
3. No internal codes; use `naming.md`.
4. Model colours are the same in every data figure (DeiT-S blue, DeiT-B orange, DINO ViT-S green).
5. Minimum font 7 pt at print size (`scripts/make_figures.py` checks this).

## Things the paper must state

- **Pre-registration.** All decision rules, margins and main comparisons were fixed before the main runs. Exploratory analyses (proposed after the main results) had their rules fixed before they ran.
- **ClickMe caveat.** The ClickMe images are very likely ImageNet training images.
- **Attention-removed baseline.** It used a 0.875 centre crop for DeiT-S and DeiT-B (the k sweep used 0.9).
- **Contribution weighting is not new.** It reduces in practice to value-norm weighting (Kobayashi et al., 2020; also SNNA, Zhu et al., 2024); it is not a new method.
- **AI assistance.** Code and experiments were run with an AI coding assistant under the author's design; all claims were checked by the author.

## Compiling

The NeurIPS style file is third-party and not in Git; download `neurips_2025.sty` from the NeurIPS 2025 website into the folder of the `.tex` file. LaTeX build files under `paper/` are ignored by Git.
