# Submission manuscript v10

Complete PDF: [attention_fast_weights_paper_v10.pdf](../../output/pdf/attention_fast_weights_paper_v10.pdf).
It contains four main-content pages, two reference pages and six appendix pages (A–F).

The author approved Candidate 2 from `paper/figure_candidates/` to combine the former conceptual
Table 1 and diagnostic Figure 2. Figure 2 now pairs the three diagnostic profiles with the existing
correction/control effects. The homepage fast-weight schematic and effect plot are byte-identical to v9.
The original numerical main table is now Table 1, with every value and verdict preserved.

`main.tex` contains the adopted figure, compact correction formulas and updated panel references.
`appendix.tex` redirects the displaced ClickMe-curve reference to the preserved per-layer table.
`asset_sources.json` fingerprints the presentation assets and historical registration record.
`revision_notes.md` records editorial decisions and claim/evidence checks. `verification.json` records
the delivered PDF checks. The source-data and figure QA files remain under `output/figure_candidates/`.
The v8 and v9 manuscripts, original results and scientific protocols are preserved.

## Build and check

Use the existing Python environment and Tectonic 0.17.0. Obtain `neurips_2025.sty` from the official
NeurIPS 2025 style archive and place it here (the style is ignored, not redistributed).
From the repository root:

```powershell
New-Item -ItemType Directory -Force build/paper-v10 | Out-Null
tectonic -X compile paper/v10/main.tex --outdir build/paper-v10 --keep-logs --keep-intermediates
if ($LASTEXITCODE -ne 0) { throw 'LaTeX build failed' }
Copy-Item build/paper-v10/main.pdf output/pdf/attention_fast_weights_paper_v10.pdf
python scripts/validate_release.py --strict --report build/paper-v10/original-integrity.json
python paper/figure_candidates/audit_candidates.py --audit-scripts C:/Users/hhuan/.codex/skills/nature-figure/scripts
python paper/v10/verify_build.py
```

Alternatively compile with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` from this directory.
The parent bibliography, authored appendix tables, execution plot and fixed-seed gallery are dependencies.
The verifier checks the page boundaries, resolved references, embedded fonts, original table contents,
approved figure bytes and unchanged registration. Render every page for visual review after edits.
