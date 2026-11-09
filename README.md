# Attention as Fast Weights: Reading, Diagnosing, and Correcting Attention Maps in Vision Transformers

Code, fixed image lists and result tables for the paper of the same name (course project, see the note at the end).

## Abstract

Attention maps are the most common way to explain what a vision transformer (ViT) looks at, yet how to read them is
rarely checked. We treat attention as a fast-weight network: for each image, fixed weights write a two-layer network
whose hidden units are the tokens, so an attention map reads hidden activations. On DeiT-S, DeiT-B and DINO ViT-S,
restricting every image region to the 16 units it reads keeps 72%, 83% and 61% of top-1 predictions, against 9%, 30%
and 4% for random units. Of three premises behind the reading, two fail in deep layers: tokens lose their position in
all models, and in supervised models a large activation is not a large contribution. Weighting each unit by its
output contribution is the only correction that improves the map: in supervised ViTs it raises object focus and human
alignment, beats removing high-norm tokens on object focus, and recovers 31–62% of their last-layer drop. Relocating scores with
position probes makes maps worse. Part of what makes deep attention maps look poor is how we read them.

![Execution test: top-1 agreement against the number of units read](figures/execution_k_sweep.png)
![Two ClickMe images: human map and three readings](figures/gallery_layers9_12.png)

*Left: top-1 agreement with the unmodified model when every image region reads only its top-k units in all layers
(solid), random units of the same number (dashed), or no attention (dotted). Right: two randomly drawn ClickMe
images with the human map and three readings of DeiT-S (layers 9–12).*

The paper text is being written; `paper/` holds everything needed for it (naming, checked key facts, LaTeX tables,
bibliography, the full evidence dossier) and `figures/` holds every figure.

## Installation

Python 3.12, one CUDA GPU for the experiments (CPU is enough for the tests and for rebuilding the tables).

```bash
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
pip install gdown  # only to fetch the MiTA-DeiT-S checkpoint (scripts/download_models.py)
```

`requirements.txt` pins every package (timm 1.0.30). Put a Hugging Face token with access to ImageNet in `.env`
(`HF_TOKEN=...`) or in the environment; it is never printed.

## Data and models

The datasets are not distributed here; `scripts/prepare_data.py` downloads and checks them (sources, licences and
counts are listed in its docstring). The fixed image lists are in `data/splits/`: the 5,000 ImageNet validation
images of the execution checks, the 2,000 ImageNet-S images of the deletion test, and the 4,000 held-out images
(2,000 train the position probes, 2,000 give the mean maps and probe accuracy; none is evaluated).

```bash
python scripts/prepare_data.py imagenet imagenet_s clickme partimagenet
python scripts/prepare_data.py --check
python scripts/download_models.py
```

Models: DeiT-S and DeiT-B (timm, pinned revisions), DINO ViT-S/16 with its official linear head, and MiTA-DeiT-S.
The MiTA-DeiT-S weights are not ours and are not redistributed here: they were released by the MiTA authors
(Wen et al., *Mixture-of-top-k attention: Efficient attention via scalable fast weights*, arXiv:2602.01219, 2026),
and `scripts/download_models.py` fetches them from the authors' download link with gdown. Please cite their paper
if you use these weights. Every checkpoint is checked against a SHA-256.
The test-time register uses 8 FFN units of DeiT-S, listed in `configs/table1_register.yaml`; `configs/register.yaml`
shows how they were found.

## Reproduce

| command | what it does | time |
| --- | --- | --- |
| `bash reproduce/from_outputs.sh` | rebuild every summary table (`results/`), figure (`figures/`) and LaTeX table (`paper/tables/`) from the committed per-image outputs of the original runs in `outputs/` (byte for byte) | a few minutes, CPU |
| `bash reproduce/full.sh` | rerun every experiment from scratch into `outputs/rerun/`, then rebuild the tables and figures from the rerun | about 12 hours on one L4 GPU |
| `bash reproduce/smoke.sh` | every experiment on 16 images, then the tables | under 10 minutes on one GPU |

`reproduce/colab.ipynb` runs the same steps on a Colab L4. `full.sh` ends by rewriting the tracked `results/`,
`figures/` and `paper/tables/`; `scripts/run.py` refuses to run on uncommitted changes (every output records its git
commit), so commit them or restore them (`git checkout -- results figures paper/tables`) before further runs. One
result = one config = one command, e.g.
`python scripts/run.py --config configs/table1.yaml`; outputs go to `outputs/<config>/[<dataset>/]<model>/` with the
effective config, the git commit, versions, GPU and log. `outputs/` holds the per-image outputs of the original runs
(about 0.6 GB, see `outputs/README.md`) and `results/` the summary tables built from them.
`docs/reproduce.md` maps every result to its config, summary table and figure or LaTeX table, and records the
reproduction checks (a full rerun from a fresh clone reproduced every table and verdict). `docs/preregistration.md` lists the predictions, decision rules and freeze commits.

```bash
python -m pytest -q
```

## Layout

```
src/fastweight_vit/   readings (R0, C1-C3, stand-ins), references (GAE, centre, random), probes, execution
                      (top-k reading sets), instrument, registers, metrics, enrichment, faithfulness, models, data,
                      experiments/ (one module per experiment)
configs/              one YAML per experiment; the comment at the top says what it produces
scripts/              run.py (run a config), make_tables.py (results/), make_figures.py (figures/),
                      make_latex_tables.py (paper/tables/), prepare_data.py, download_models.py, check_environment.py
reproduce/            from_outputs.sh, full.sh, smoke.sh, colab.ipynb
data/splits/          fixed image lists (IDs only)
outputs/              per-image outputs of the original runs (one folder per config and model)
results/              summary tables (CSV) of the original runs: every mean, interval and verdict
figures/              paper figures, one panel per file (figures/README.md lists them)
paper/                materials for writing the paper: naming, key facts, LaTeX tables, bibliography, evidence dossier
docs/                 preregistration.md, reproduce.md, verification/ (full-rerun comparison)
tests/                pytest suite (CPU)
```

Internal codes (R0 = received attention, C1-C3 = the three corrections, M1-M3 = the three premises, P1-P7 = the
pre-registered predictions) are used in the code and in `results/`; `paper/naming.md` maps them to the names used in
the paper. `AGENTS.md` is a short guide for continuing the work.

## Citation, licence, acknowledgements

Please cite with `CITATION.cff`. Code under the MIT licence (`LICENSE`). We thank the authors of DeiT, DINO, timm,
MiTA (Wen et al., 2026, arXiv:2602.01219), ImageNet-S, ClickMe / Harmonization and PartImageNet for releasing
their models and data. GAE follows Chefer et al. (ICCV 2021); the test-time register follows Jiang et al. (2025).

This is a course project of DSAA 2012 Deep Learning, HKUST(GZ). The code was written with an AI coding assistant
under the author's design; the author checked every conclusion.
