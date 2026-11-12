# Revised submission manuscript v11

[Complete PDF](../../output/pdf/attention_fast_weights_paper_v11.pdf): four main pages,
two reference pages and eight appendix pages (A–F).

This implements the confirmed findings in [the v10 review](../reviews/v10_submission_review.md).
It clarifies technical definitions, statistical scope, prior-method attribution and reproduction
details using existing experiments. Original results, verdicts, registration and v8–v10 artifacts
remain preserved. The homepage schematic/effect plot and current Figure 2 are unchanged vector
assets; captions and cross-references are updated. Appendix plot/interval typography is enlarged.

- `main.tex`, `appendix.tex`: submission source.
- `references.bib`: isolated bibliography with confirmed metadata edits and two related works.
  The complete Ramsauer/MiTA entries remain unchanged pending author-order/version confirmation.
- `tables/`: unchanged main/sanity tables and a presentation-only copy of paired differences.
- `asset_sources.json`: input and artifact hashes; `verification.json`: actual delivery checks.
- `revision_notes.md`: outline, implementation decisions, claim/evidence map and self-review.
- `verify_build.py`: checks page boundaries, original measurements, preserved figures and deferred references.

## Build

Place the official, ignored `neurips_2025.sty` in this directory. Use Tectonic 0.17.0 and the
project Python environment (PyMuPDF for verification). From the repository root:

```powershell
New-Item -ItemType Directory -Force build/paper-v11 | Out-Null
tectonic -X compile paper/v11/main.tex --outdir build/paper-v11 --keep-logs --keep-intermediates
if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed' }
Copy-Item build/paper-v11/main.pdf output/pdf/attention_fast_weights_paper_v11.pdf
python scripts/validate_release.py --strict --report build/paper-v11/original-integrity.json
python paper/v11/verify_build.py
```

The original execution plot, fixed-seed gallery, diagnostics table and relocation table remain
repository dependencies. Alternatively run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` from this directory.
After edits, render and inspect every page as well as running the verifier. Do not rebuild scientific
outputs or change protocols as part of the manuscript build.
