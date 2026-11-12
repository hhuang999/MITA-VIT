# Exploratory analyses

These studies extend the registered ViT attention-map evaluations. They use
frozen model weights and are **exploratory**: their protocols and follow-up
amendments record when each comparison was chosen. The original registration,
decision rules, and outcomes are in [`docs/preregistration.md`](../docs/preregistration.md)
and [`results/`](../results/). Committed CSV/JSON files in each study directory
are the original exploratory results. All commands below write fresh results
under ignored `outputs/rerun/`; they never replace committed results.
The protocol files are preserved verbatim for timestamped provenance, including
their former development paths. Use the commands on this page with the public
directory layout.

Run these commands from the repository root after following the top-level
installation instructions. `python` must use an environment with this package,
PyTorch, timm 1.0.30, NumPy, pandas, SciPy, and Pillow. The CPU-only analyses
below need the saved `outputs/` files but no image download or model forward.

## Saved-output analyses

The geometry analysis reads the committed ImageNet-S per-image exports. Run the
first command before its follow-ups because the latter use the first command's
fixed quartile cuts:

```powershell
python -m experiments.geometry.analyze
python -m experiments.geometry.joint
python -m experiments.geometry.auroc_sensitivity
python -m experiments.geometry.cohort_audit
```

Results appear in `outputs/rerun/geometry/`. Compare `strata.csv`,
`auroc_strata.csv`, `joint_cells.csv`, and `cohort_audit.csv` with the same names
in `experiments/geometry/`. The geometry protocol and its amendments are in
[`geometry/protocol.md`](geometry/protocol.md).

The endpoint analysis reconstructs registered top-1 deletion and ClickMe
contrasts from the saved per-image exports before it reports the exploratory
continuous endpoints and batch bootstrap:

```powershell
python -m experiments.endpoints.analyze
python -m experiments.endpoints.plot_curves --input outputs/rerun/endpoints/budget_differences.csv --output outputs/rerun/endpoints/deletion_curves.pdf
```

`verification.json` reports the reconstruction checks; all 18 historical
paired contrasts must match to floating-point precision. The script produces
`deletion_areas.csv`, `budget_differences.csv`, `clickme_bootstrap.csv`, and a
vector PDF. Add `--preview-formats` to also export SVG and PNG. The original protocol is in
[`endpoints/protocol.md`](endpoints/protocol.md). To rebuild the paper figure
from the committed exploratory results, use:

```powershell
python -m experiments.endpoints.plot_curves --input experiments/endpoints/budget_differences.csv --output paper/figures/deletion_curves.pdf
```

## Oxford-IIIT Pet study and CLS intervention

The Oxford studies require the **official Oxford-IIIT Pet test images and
trimaps** and the frozen supervised DeiT-S checkpoint. The model is
`timm/deit_small_patch16_224.fb_in1k` at Hugging Face revision
`91327a9c99f98fe6b524cd4d397b7226b80e1365`; `load_deit` downloads it on
first use. The expected `model.safetensors` SHA-256 is
`1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b`.
Set `HF_HUB_CACHE` to a writable directory if you want a separate model cache.
Both studies use the pinned timm evaluation transform: bicubic resize of the
short side to 248, 224 × 224 center crop, ImageNet normalization; masks use
nearest-neighbor resizing and the matching crop.

In PowerShell, obtain and verify the two Oxford archives, then extract them
under the default data root:

```powershell
New-Item -ItemType Directory -Force data/oxford_pets | Out-Null
curl.exe -L --fail -o data/oxford_pets/images.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl.exe -L --fail -o data/oxford_pets/annotations.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
(Get-FileHash data/oxford_pets/images.tar.gz -Algorithm SHA256).Hash.ToLower()
(Get-FileHash data/oxford_pets/annotations.tar.gz -Algorithm SHA256).Hash.ToLower()
tar -xzf data/oxford_pets/images.tar.gz -C data/oxford_pets
tar -xzf data/oxford_pets/annotations.tar.gz -C data/oxford_pets
```

The expected archive hashes are, respectively,
`67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d`
and `52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91`.
The scripts expect `images/*.jpg`, `annotations/test.txt`, and
`annotations/trimaps/*.png` under that root. To reuse data elsewhere, set
`OXFORD_PETS_ROOT` to the directory containing `images/` and `annotations/`.

Run the spatial readout and its reciprocal-donor sensitivity check with a CUDA
GPU:

```powershell
python -m experiments.spatial.run_spatial --output-dir outputs/rerun/spatial
python -m experiments.spatial.audit_results --input-dir outputs/rerun/spatial --output-dir outputs/rerun/spatial
```

The spatial run evaluates the first 384 Oxford test names in the fixed SHA-256
order, with 383 eligible masks after cropping. Its primary same-ring enrichment
means are R0 `0.14064`, C3 `0.15964`, and donor-rank control `0.13731`.
`summary.json` and `pair_block_sensitivity.json` carry the full paired
intervals; `per_image.csv` carries each image's scores. The fixed sample is
[`spatial/selected_names.txt`](spatial/selected_names.txt), and its original
choices are in [`spatial/PROTOCOL.md`](spatial/PROTOCOL.md).

Run the separate last-block CLS intervention on the same data and checkpoint:

```powershell
python -m experiments.cls.run_cls --output-dir outputs/rerun/cls
python -m experiments.cls.analyze_endpoints --input-dir outputs/rerun/cls --output-dir outputs/rerun/cls
```

This study takes 128 names from the fixed 384 using a second SHA-256 ordering.
It records 196 patch-source interventions per image, five score maps, and
independent full-forward validation rows. `summary.csv` reports mean per-image
Spearman coefficients of `0.6965` (all-query R0), `0.7055` (all-query C3),
`0.7298` (CLS attention), `0.7466` (CLS summed-head), and `0.7558` (CLS net
term). The `three_endpoint_*.csv` files compare the same 128 images on margin
sensitivity, object enrichment, and object/background AUROC. See
[`cls/protocol.md`](cls/protocol.md) and
[`cls/addendum_protocol.md`](cls/addendum_protocol.md) for the timing of the
original intervention and later mask comparison.

The historical hash prefixes `v13-spatial-2026:` and `v13-cls-2026:` are
preserved **only because they determine the published fixed samples**. Both
scripts check their recomputed selections against the committed name lists
before scoring. No result-based filtering is applied.
