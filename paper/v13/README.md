# How Should We Read ViT Attention Maps? A Fast-Weight Perspective

[Submission PDF](../../output/pdf/attention_fast_weights_paper_v13.pdf) | [Experiment source and results](../../research/v13/) | [Original registration](../../docs/preregistration.md) | [Reviewed baseline](https://github.com/hhuang999/vit-fastweight-lens/tree/35323aa6d4959d465c47044057a0296374374ec7).

The compiled paper uses the v12 schematic, diagnostics figure, original scores and interval tables directly. Each new study has a protocol, analysis code and numeric output under `research/v13/`. The spatial experiment and CLS intervention use the released supervised DeiT-S checkpoint and the public Oxford-IIIT Pet data. The full images, archives, pretrained model and local Python overlay are external to the source package. For the companion source ZIP, extract its contents over a checkout of the linked baseline.

| Folder | How to reproduce the new evidence |
| --- | --- |
| [geometry](../../research/v13/geometry/) | Run `analyze.py`, then `joint.py`, `auroc_sensitivity.py` and `cohort_audit.py` with NumPy and pandas. Categories use model-specific masks. Reports identify follow-ups chosen after the first geometry results. |
| [endpoints](../../research/v13/endpoints/) | Run `analyze.py` from the checkout with NumPy and pandas. `plot_curves.py` reproduces the vector appendix plot. The verification JSON records reconstruction of the historical contrasts. |
| [spatial](../../research/v13/spatial/README.md) | Follow the documented Oxford archive and Hugging Face checkpoint paths, hashes and Python environment. Run `run_spatial.py`, then `audit_results.py` for the reciprocal-donor sensitivity. |
| [cls](../../research/v13/cls/) | Reuse the spatial weights, data and selected filename list. Run `run_cls.py` and then `analyze_endpoints.py` to reproduce the matched mask comparison. Compare original/counterfactual logits using the independent full forward checks recorded with the results. |

## Manuscript build

Use the original official NeurIPS 2025 style. Place a local copy of `neurips_2025.sty` in `paper/v13/`. From the repository root, with the project Python dependencies, Matplotlib and PyMuPDF:

```powershell
python paper/v13/make_tables.py
New-Item -ItemType Directory -Force build/paper-v13 | Out-Null
tectonic -X compile paper/v13/main.tex --outdir build/paper-v13 --keep-logs --keep-intermediates
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
Copy-Item build/paper-v13/main.pdf output/pdf/attention_fast_weights_paper_v13.pdf
$env:PYTHONPATH = 'build/paper-v9/python-deps;src'
python scripts/validate_release.py --strict --report build/paper-v13/baseline-audit.json
python paper/v13/verify_build.py
```

The repository's original release audit checks historical research inputs. The v13 PDF verifier checks pagination, citations, cross-references, fonts, text bounds, the PDF hash and its agreement with the build. Inspect rendered pages after manuscript edits. Earlier reference audits are in [paper/reviews](../reviews/).
