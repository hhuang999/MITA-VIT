# Framework-focused submission manuscript v12

**Interpreting ViT Attention Maps Through a Fast-Weight Lens**

[Complete PDF](../../output/pdf/attention_fast_weights_paper_v12.pdf): four main pages,
two reference pages and nine appendix pages (A–E). The abstract contains 113 words.

The author-approved revision makes the analytical framework the main contribution:
three assumptions behind spatial attention-map readings, each connected to a diagnostic
proxy, correction and matched control. Related work compares fast-weight foundations,
attention readings/token correspondence, and explanation evaluation. Results retain the
original positive and negative findings, registered decisions and post-hoc scope.

Parameter randomisation is now subsection B.1 and Table B.3 in Additional results.
Appendix E, Key parts of the code, ends the paper. Homepage and Figure 2 vector assets,
all main/paired/sanity tables, stored results and the complete v11 bibliography are
unchanged; unresolved Ramsauer/MiTA entries remain untouched.

- `main.tex`, `appendix.tex`: complete manuscript source.
- `revision_notes.md`: outline, paragraph roles, claim/evidence map and self-review.
- `asset_sources.json`: scientific input and preserved-asset hashes.
- `verify_build.py`, `verification.json`: executable checks and final delivery report.

## Build and verify

Place the official, ignored `neurips_2025.sty` in this directory. From the repository
root, use Tectonic 0.17.0 and Python with PyMuPDF; use the project environment for the
strict release audit:

```powershell
New-Item -ItemType Directory -Force build/paper-v12 | Out-Null
tectonic -X compile paper/v12/main.tex --outdir build/paper-v12 --keep-logs --keep-intermediates
if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed' }
Copy-Item build/paper-v12/main.pdf output/pdf/attention_fast_weights_paper_v12.pdf
python scripts/validate_release.py --strict --report build/paper-v12/original-integrity.json
python paper/v12/verify_build.py
```

The original execution plot, fixed-seed gallery, diagnostics and relocation tables remain
repository dependencies. Alternatively run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`
from this directory. Render and inspect every page after manuscript changes. Scientific
outputs and protocols are preserved; the manuscript build does not run experiments.
