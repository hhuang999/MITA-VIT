<h1 align="center">How Should We Read ViT Attention Maps?<br><em>A Fast-Weight Perspective</em></h1>

<p align="center">Hongbo Huang · HKUST(GZ) · DSAA 2012</p>
<p align="center"><a href="paper/fast-weight-attention-maps.pdf"><b>Paper (PDF)</b></a> · <a href="#reproduce-the-paper">Reproduce the paper</a> · <a href="https://github.com/hhuang999/vit-fastweight-lens/tree/paper-final">Full research snapshot</a></p>

**A background patch lights up on an attention map. What does that tell us?** The map shows how strongly queries attend to the token at that grid position. By the final layer, that token has already mixed information from other patches. To interpret the bright spot, we need to know what the token sends, who receives it, and what we want the map to explain.

**Why the receiver matters.** On the same 128 pet images, the CLS-only map better identified which value terms affected the classifier’s margin. The all-query map better located the animal. The choice of receiver changes which question the map answers.

<p align="center"><img src="docs/figures/fast-weight-readout.svg" width="900" alt="For a fixed image and one head, the query and source keys determine softmax activations. Each activation scales a projected value vector. A spatial map can pool either the activations or the lengths of those vector contributions at each source slot."></p>

For a fixed image, one attention head acts like a two-layer network on the **receiving query**. Each source token supplies an input weight (its scaled key) and an output weight (its projected value). The query activates the sources through a joint softmax. An ordinary received-attention map pools those **activations** at each source slot. A value-weighted map pools the **lengths of their vector contributions**. Either map can pool all queries or follow just the classification token (CLS).

## Three assumptions behind a spatial map

We tested each assumption in frozen DeiT-S, DeiT-B and DINO ViT-S/16. Each readout change has a matched control to help identify what accounts for a change in performance.

| Reading assumption | Change to the readout | Matched control |
| --- | --- | --- |
| **Image specificity** · Is this patch bright because of this image? | Subtract the mean map estimated on held-out images. | Subtract a centered Gaussian matched to the mean map’s spatial standard deviation. |
| **Spatial correspondence** · Where should a contextualized token’s score appear? | Redistribute its score using a linear probe’s predicted grid-position probabilities. | Blur the original map by the same mean squared displacement. |
| **Output contribution** · How much does an activated source send? | Weight attention by the length of each source’s projected value. | Zero received attention at patches with unusually large block-input norms. |

Relocating scores by decoded position lowered all nine checkpoint–metric scores relative to received attention. In supervised DeiT-S/B, value weighting improved foreground and human-map alignment over received attention and met the registered improvement criterion in three of four corresponding matched-control contrasts. [Original protocol and decision rules](docs/preregistration.md) · [Full contrasts in Figure 2 and the appendix](paper/fast-weight-attention-maps.pdf)

## One layer, two different rankings

In an **exploratory** follow-up, we kept the checkpoint (DeiT-S), layer (12), images (128 Oxford-IIIT Pet test images) and value weighting fixed. We changed only which queries contributed to the map. To measure the effect on CLS, we held attention fixed, removed each patch’s value term from CLS in turn, and measured the absolute change in the original top-two logit margin.

| What should the map rank highly? | All queries | CLS only | Paired change (95% CI) |
| --- | ---: | ---: | ---: |
| Patches with larger margin effects · Spearman ρ | 0.705 | 0.747 | **+0.041** [0.026, 0.057] |
| Foreground patches at matched radii · enrichment | 0.127 | 0.095 | **−0.033** [−0.043, −0.023] |
| Foreground patches across radii · AUROC | 0.769 | 0.678 | **−0.091** [−0.110, −0.074] |

Each row averages per-image scores. Differences use the unrounded paired observations. The foreground comparison was added after observing the margin result. The intervention measures one final-layer path in this checkpoint. It answers a narrower question than how much an input pixel matters to the prediction. [Per-image results](experiments/cls/three_endpoint_per_image.csv) · [Paired contrasts](experiments/cls/three_endpoint_contrasts.csv) · [Follow-up protocol](experiments/cls/addendum_protocol.md)

## Reproduce the paper

**Start with the published results.** In a fresh Python 3.12 environment, rebuild both main figures and both tables on CPU. The inputs are included in the lightweight checkout.

```sh
git clone --depth 1 --single-branch --branch main https://github.com/hhuang999/vit-fastweight-lens.git
cd vit-fastweight-lens
python -m pip install matplotlib==3.11.2 numpy==2.5.4 pandas==3.1.0
python rebuild_main.py
```

| Paper item | Source | Generated file |
| --- | --- | --- |
| Figure 1 · Fast-weight schematic | [Vector drawing code](scripts/display/build_static_figures.py) | `paper/figures/fast_weight_lens.pdf` and `.svg` |
| Figure 2 · Diagnostics and matched controls | [Diagnostics](results/diagnose.csv) · [Contrasts](results/correct_tests.csv) | `paper/figures/diagnosis_correction_bridge.pdf` |
| Table 1 · Assignment of value lengths | [Spatial summary](experiments/spatial/summary.json) | `paper/tables/spatial.tex` |
| Table 2 · Choice of receiver | [Endpoint means](experiments/cls/three_endpoint_summary.csv) · [Paired contrasts](experiments/cls/three_endpoint_contrasts.csv) | `paper/tables/cls.tex` |

**Run from images and weights.** The [full research snapshot](https://github.com/hhuang999/vit-fastweight-lens/tree/paper-final#run-from-images-and-model-weights) gives dataset and checkpoint setup, preprocessing and inference commands, archived outputs and the LaTeX source. On `main`, browse the [readout implementation](src/fastweight_vit/readings.py), [experiment configurations](configs), [fixed sample splits](data/splits), and the exploratory [source-slot](experiments/spatial) and [CLS](experiments/cls) studies.

**Run the CPU tests.** From the repository root in the same Python environment:

```sh
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

[Download the course submission ZIP](https://github.com/hhuang999/vit-fastweight-lens/releases/download/paper-final/vit-fast-weight-attention-maps-course-final.zip) · [Citation metadata](CITATION.cff)
