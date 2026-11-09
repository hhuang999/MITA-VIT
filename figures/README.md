# Figures

Every figure of the paper, one panel per file, as vector PDF and 300-dpi PNG at print size.

- **Rebuild:** `python scripts/make_figures.py`. Inputs are `results/` and, for the image panels, `outputs/`.
- **Inside each figure:** no figure numbers, panel letters or captions; the paper adds them with `subcaption`.
- **Names:** no internal codes; names follow `paper/naming.md`.
- **Fonts:** at least 7 pt at the size below.
- **Model colours** in every data figure: DeiT-S blue, DeiT-B orange, DINO ViT-S green. The schematic uses purple, dark red, greys and the inferno colour map.

| File | Content | Size (in) | Source |
| --- | --- | --- | --- |
| `schematic_fast_weight_network` | An attention head as a two-layer network written for each image: keys = first-layer weights, softmax = activation, values = second-layer weights, tokens = hidden units | 2.60 × 1.90 | none |
| `schematic_reading` | Received attention draws the activation of unit j at patch j | 1.45 × 1.90 | none |
| `schematic_premises` | The three premises of the reading → how each is measured → its correction → the stand-in the correction must beat | 5.50 × 1.50 | none |
| `execution_k_sweep` | Top-1 agreement with the unmodified model (%) against the number of units each image region reads (k = 4–128). Solid: top-k reading sets; dashed: random sets of the same size; dotted: attention removed. Three models | 2.60 × 1.90 | `results/k_sweep.csv` |
| `gallery_layers9_12` | Two ClickMe images (indices 47 and 287, the two lowest indices of a seeded random draw, not selected). Rows: image, human (ClickMe) map, received attention, contribution weighting, probe relocation; DeiT-S, layers 9–12 summed. Each map is min-max normalised, with one shared colour bar | 2.80 × 1.90 | `outputs/table1/clickme/deit/gallery_*.npz` |
| `gallery_layer12` | The same at layer 12 alone | 2.80 × 1.90 | the same (`*_l12` maps) |
| `premise_mean_map_share` | Share of variance explained by the mean map, per layer; band = 95% interval | 1.35 × 1.50 | `results/diagnose.csv` (panel M1, variant R0) |
| `premise_probe_accuracy` | Position-probe accuracy per layer | 1.35 × 1.50 | `results/diagnose.csv` (M2, probe) |
| `premise_high_norm_share` | Share of received attention on high-norm tokens per layer | 1.35 × 1.50 | `results/diagnose.csv` (M3, R0) |
| `received_attention_vs_clickme` | Image-specific ClickMe Spearman of received attention per layer | 1.35 × 1.50 | `results/diagnose.csv` (quality, R0) |
| `c2_analysis_distance_error` | Expected distance (patches) between the true patch and the position probe's distribution (solid) or the matched blur (dashed), layers 6–12 | 2.60 × 1.90 | `results/c2_analysis/errors.csv` (distance) |
| `c2_analysis_far_error_share` | Share of the error mass farther than 3 patches; grey line = pre-registered threshold 0.5 | 2.60 × 1.90 | `results/c2_analysis/errors.csv` (far_share) |
| `c2_analysis_shift_by_distance` | Change in the share of the score at each distance from the image centre caused by probe relocation (solid) and the matched blur (dashed); layers 9–12, ImageNet-S | 2.60 × 1.90 | `results/c2_analysis/shift_bins.csv` |
| `c2_analysis_mean_prediction_deit_s` | DeiT-S, layer 12: the probe's mean prediction over the 2,000 held-out test images, relative to uniform (= 1). Not min-max normalised, because the point is that it is close to uniform | 1.80 × 1.30 | `outputs/c2_analysis/deit/mean_prediction.npz` |
| `c2_analysis_mean_prediction_dino` | The same for DINO ViT-S | 1.80 × 1.30 | `outputs/c2_analysis/dino/mean_prediction.npz` |

The LaTeX tables are in `paper/tables/`; see `paper/README.md`.
