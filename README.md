# How Should We Read ViT Attention Maps? A Fast-Weight Perspective

**Hongbo Huang · DSAA 2012, HKUST(GZ)**

[Read the final paper](paper/fast-weight-attention-maps.pdf) · [Course ZIP (686 KiB)](https://github.com/hhuang999/vit-fastweight-lens/releases/download/paper-final/vit-fast-weight-attention-maps-course-final.zip) · [Full paper source and archived results](https://github.com/hhuang999/vit-fastweight-lens/tree/paper-final)

The paper tests three assumptions about spatial attention maps using matched controls. On the same 128 Oxford-IIIT Pet images, switching from all queries to CLS increased agreement with the measured classifier intervention (+0.041 Spearman), while foreground enrichment and AUROC fell (-0.033 and -0.091).

## Rebuild the main figures and tables

To get the lightweight checkout:

```sh
git clone --depth 1 --single-branch --branch main https://github.com/hhuang999/vit-fastweight-lens.git
cd vit-fastweight-lens
```

Use Python 3.12 in a fresh environment:

```sh
python -m pip install matplotlib==3.11.2 numpy==2.5.4 pandas==3.1.0
python rebuild_main.py
```

Rebuilding these displays uses the included summaries and does not require a GPU, checkpoints or image downloads.

| Paper item | Included inputs | Output |
| --- | --- | --- |
| Figure 1 | Editable vector schematic in `scripts/display/build_static_figures.py` | `paper/figures/fast_weight_lens.pdf` and `.svg` |
| Figure 2 | `results/diagnose.csv`, `results/correct_tests.csv` | `paper/figures/diagnosis_correction_bridge.pdf` |
| Table 1 | `experiments/spatial/summary.json` | `paper/tables/spatial.tex` |
| Table 2 | `experiments/cls/three_endpoint_summary.csv`, `three_endpoint_contrasts.csv` | `paper/tables/cls.tex` |

The figure and table files listed above are generated on demand. The paper directory contains the final PDF.

## Experiment code and provenance

The core attention readouts and evaluation code are in `src/fastweight_vit/`. The original registration is in `docs/preregistration.md`, with configurations in `configs/` and fixed splits in `data/splits/`. The Oxford source-slot and CLS studies include their scripts, exploratory protocols and selected filenames in `experiments/spatial/` and `experiments/cls/`.

For model inference, dataset and weight setup, full experiment dependencies and archived per-image outputs, follow the [full repository at the submitted revision](https://github.com/hhuang999/vit-fastweight-lens/tree/paper-final#run-from-images-and-model-weights).

## Run the core tests

From the repository root, in the Python 3.12 environment:

```sh
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

These tests cover CPU model execution, attention readouts, metrics and experiment configuration. The full artifact audit, raw observations and buildable LaTeX sources are preserved in the [paper-final snapshot](https://github.com/hhuang999/vit-fastweight-lens/tree/paper-final).
