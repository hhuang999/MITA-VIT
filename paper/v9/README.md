# Revised manuscript v9

This revision changes the manuscript and its presentation using the existing experiments.
The supplied v8 source/PDF, result CSVs, inference outputs, protocols and registration record
remain unchanged. The v8 reproduction workflow continues to target v8.

- `main.tex`, `appendix.tex`: revised manuscript.
- `build_assets.py`: reads supplied results and builds the revised table, effect-size plot and schematic.
- `asset_sources.json`: exact source rows and input SHA-256 hashes for the new presentation assets.
- `evidence/sanity.csv`: byte-identical copy of the archived historical summary used in Appendix F.
- `revision_notes.md`: outline, paragraph roles, claim/evidence map and five-dimension self-review.
- `verification.json`: PDF checksum, page-count, numeric-cell, input-hash and rendering checks.
- `../../output/pdf/attention_fast_weights_paper_v9.pdf`: compiled deliverable.

From the repository root, use Python 3.12 with Matplotlib:

```powershell
python paper/v9/build_assets.py
```

Obtain `neurips_2025.sty` from the [official NeurIPS 2025 style archive](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip) and place it in this directory.
It is ignored and not redistributed. With Tectonic 0.17.0:

```powershell
New-Item -ItemType Directory -Path build/paper-v9 -Force
tectonic -X compile paper/v9/main.tex --outdir build/paper-v9 --keep-logs --keep-intermediates
New-Item -ItemType Directory -Path output/pdf -Force
Copy-Item build/paper-v9/main.pdf output/pdf/attention_fast_weights_paper_v9.pdf
```

Alternatively run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` from this directory.
The parent bibliography, original diagnosis plot and fixed-seed gallery remain dependencies.
All new table arrows, control diamonds and bold entries use the original verdicts;
the effect-size plot displays original means and confidence intervals without recomputing them.
Appendix F restores the existing randomisation check from `archive/results/sanity.csv`;
its active summary copy allows this manuscript to build without the archive directory.

The historical term `centre-blind` is replaced in the revised prose by `centre-controlled`.
Metric definitions and registered comparisons, margins and outcomes are unchanged.
