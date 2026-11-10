# Is It the Model or the Reading? A Fast-Weight View of ViT Attention Maps

**Hongbo Huang** · DSAA 2012 Deep Learning course project

[Paper (PDF)](paper/main.pdf) · [Reproduction map](docs/reproduce.md) · [Code branch: release/v1](https://github.com/hhuang999/vit-fastweight-lens/tree/release/v1)

Deep-layer attention maps often look poor. This project asks how much of that comes from how the maps are read.
It uses the long-established fast-weight view of an attention head as a two-layer network whose hidden units
are tokens. The code tests three premises of received-attention maps: specificity, location and contribution.

On DeiT-S, DeiT-B and DINO ViT-S, two premises fail in deep layers. The diagnosis predicts the correction's
outcome for two of the three premises. Contribution weighting, in practice value-norm weighting, improves
object enrichment by 0.038 and 0.065 in the supervised models. Probe relocation hurts all three models.
The paper explains the interpretation, sources and limits of these findings.

![Execution test](figures/execution_k_sweep.png)
![Example maps on two randomly drawn ClickMe images](figures/gallery_layers9_12.png)

## What the code contains

| Part | Implementation |
| --- | --- |
| Check which units the model uses | Restrict attention to top-k or matched random reading sets |
| Measure the three premises | Mean-map variance share, position probes and attention on high-norm tokens |
| Compare readings and controls | Mean-map subtraction, probe relocation and contribution weighting |
| Evaluate maps | Same-ring enrichment, image-specific ClickMe alignment and deletion faithfulness |
| Check the failure mechanisms | Probe errors, MLP probes, globality and contribution-weight shuffling |

## Install

Use Python 3.12. CPU tests and result rebuilding need no datasets, checkpoints or GPU.
For CPU use:

```bash
git clone --branch release/v1 https://github.com/hhuang999/vit-fastweight-lens.git
cd vit-fastweight-lens
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Run shell workflows with Bash, for example in WSL.
For GPU experiments, install the PyTorch requirements from `https://download.pytorch.org/whl/cu128` instead.
Linux with Liberation Sans fonts gives the closest figure rendering to the supplied files.

## Rebuild the paper results

```bash
python -m pytest -q
bash reproduce/from_outputs.sh
```

The rebuild reads the committed `outputs/`. It writes the retained summaries in `results/`, three figures in
`figures/`, the paired-difference source in `paper/tables/`, and the generated paper tables and diagnostic figure.
The fourth paper table is hand-written and stays unchanged. See [the paper build guide](paper/README.md).

Summary values agree within absolute error 1e-12 and all verdicts must match. Floating-point last digits and
font rendering can vary across machines. The supplied results and figures are the original files.
Build in a copy if you want to keep them unchanged.

## Rerun experiments

New inference runs need a CUDA GPU, external datasets and model checkpoints. ImageNet access needs a Hugging Face
token in `.env` as `HF_TOKEN=...`, or in the environment. Keep credentials out of Git.

```bash
python scripts/prepare_data.py imagenet imagenet_s clickme
python scripts/prepare_data.py --check
python -m pip install gdown  # MiTA checkpoint download only
python scripts/download_models.py
python scripts/run.py --config configs/k_sweep.yaml --model deit --output outputs/rerun
# All paper configurations, in dependency order:
bash reproduce/full.sh
```

Data and checkpoints are downloaded separately. Their sources, access terms and hash checks are in the preparation
scripts. Fixed image lists are in `data/splits/`. `reproduce/smoke.sh` and `reproduce/colab.ipynb` provide GPU workflows.
Some shared experiment modules also compute additional analyses; the default builders export only paper results.
The two historical register summaries used in Appendix A are supplied with their provenance in
[`results/register/`](results/register/README.md); the full historical accuracy check is not part of the current rerun.

Runs record their source revision. A Git checkout must have committed source changes. A course ZIP made with
`git archive` carries the revision in `.git_archival.txt` and works without `.git`.

## Layout

| Path | Contents |
| --- | --- |
| `src/fastweight_vit/` | Models, readings, metrics, interventions, data loaders and experiments |
| `configs/` | Fourteen paper configurations |
| `scripts/` | Data preparation, experiment runner and result builders |
| `reproduce/` | CPU rebuild, GPU rerun, smoke workflow and Colab notebook |
| `data/splits/` | Fixed image lists |
| `outputs/` | Original per-image outputs; new runs go to ignored subfolders |
| `results/` | Paper summaries and the two historical register evidence tables |
| `figures/` | Schematic, execution plot and example maps |
| `paper/` | Final PDF, LaTeX source, paper figures, tables and their builders |
| `docs/reproduce.md` | Paper-to-code and result map |
| `tests/` | CPU tests with synthetic data and random model weights |
| `archive/` | Experiments and old materials outside the final paper |

## Export the course code

```bash
git archive --format=zip -o course_code.zip release/v1
```

Git excludes `archive/` automatically. The ZIP contains the paper code and original outputs. External datasets,
checkpoints, local writing materials and credentials are not included. After extraction, follow the install and
rebuild steps above. The archive is not needed to run the exported code.

## Licence and sources

Code is under the [MIT licence](LICENSE). Dataset, model and reused-method sources and terms are recorded beside
their implementations. Follow their original licences when downloading or using those assets.
