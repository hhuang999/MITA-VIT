# Reproduction map

For every result of the paper: the config that produces it, the summary table it ends up in, the figure or LaTeX table built from it, and the GPU time of the original run. Names follow `paper/naming.md`.

## How results are produced

| Step | Command | Output |
| --- | --- | --- |
| Run one experiment | `python scripts/run.py --config configs/<config>.yaml` | `outputs/<config>/[<dataset>/]<model>/`: per-image tables and arrays, the effective config, git commit, versions, GPU, log |
| Summary tables | `python scripts/make_tables.py` | `results/*.csv` (every mean, interval and verdict) |
| Figures | `python scripts/make_figures.py` | `figures/*.pdf, *.png` (see `figures/README.md`) |
| LaTeX tables | `python scripts/make_latex_tables.py` | `paper/tables/*.tex` |
| Everything | `bash reproduce/from_outputs.sh` (CPU, a few minutes, from the committed `outputs/`), or `bash reproduce/full.sh` (rerun from scratch into `outputs/rerun/`, about 12 GPU hours on one L4) | all of the above |

- **Committed files.** `outputs/` holds the per-image outputs of the original runs (about 0.6 GB; `outputs/README.md`), and `results/`, `figures/` and `paper/tables/` are built from them. `from_outputs.sh` rebuilds all three byte for byte.
- **Clean code before a run.** `scripts/run.py` refuses to run on uncommitted code changes, because every output records its git commit.

## Result → config → files

| Result | Config(s) | Summary table(s) in `results/` | Figure / LaTeX table | GPU time of the original run |
| --- | --- | --- | --- | --- |
| Execution test, k sweep; attention removed | `k_sweep`, `attention_off` | `k_sweep.csv`, `k_sweep_tests.csv` | `execution_k_sweep` | 0.60 h + 0.16 h |
| One layer restricted or without attention | `single_layer`, `attention_off` | `single_layer.csv` | `appendix/single_layer.tex` | 1.60 h |
| Head vs body patches (PartImageNet) | `part_reliance` | `part_reliance.csv` | `appendix/head_vs_body.tex` | 1.37 h |
| Top-k map vs full map | `map_similarity` | `map_similarity.csv` | `appendix/head_vs_body.tex` | 1.08 h |
| Mean map, share of variance explained | `diagnose_probe`, `diagnose_mean_maps` | `diagnose.csv` (M1), `radial.csv` | `premise_mean_map_share` | 0.09 h + 0.33 h |
| Position probes and their accuracy | `diagnose_probe` | `diagnose.csv` (M2), `probe_accuracy.csv`, `numbers.csv` | `premise_probe_accuracy` | (above) |
| Received attention on high-norm tokens | `high_norm` | `diagnose.csv` (M3) | `premise_high_norm_share` | 2.08 h |
| Value-norm ratio of high-norm tokens | `value_norm` | `value_norm.csv` | — (text) | 0.90 h |
| Per-layer ClickMe alignment of received attention | `clickme_layers` | `diagnose.csv` (quality) | `received_attention_vs_clickme` | 2.58 h |
| Main results: readings, stand-ins, references; negative controls; last-layer recovery; ClickMe gallery | `table1` | `table1_long.csv`, `table1.csv`, `table1.md`, `correct_tests.csv`, `negative_controls.csv`, `last_layer.csv`, `numbers.csv` | `table1.tex`, `table1_full.tex`, `appendix/paired_differences.tex`, `appendix/last_layer_recovery.tex`, `gallery_layers9_12`, `gallery_layer12` | 3.05 h |
| Model-side variants | `table1_register`, `table1_mita` (units from `register`) | `variants.csv` | `table1.tex`, `appendix/model_side_variants.tex` | 0.13 h + 0.13 h (+ 0.02 h) |
| Order B of the three corrections | `order` | `order.csv` | `table1_full.tex`, `appendix/combination_orders.tex` | 0.56 h |
| Sanity check (model randomisation) | `sanity` | `sanity.csv` | `appendix/sanity_check.tex` | 0.39 h |
| All pre-registered predictions | — | `predictions.csv`, `diagnose_tests.csv` | `appendix/predictions.tex` | — |
| Why probe relocation fails (exploratory) | `c2_analysis` | `c2_analysis/*.csv` | `c2_analysis_*`, `appendix/probe_errors.tex`, `appendix/score_shift.tex` | 0.32 h |
| Exploratory checks (combinations, contribution checks, dose, globalisation, MLP probe, GAE gap) | `supplementary_heldout`, `supplementary_eval` | `supplementary/*.csv` | `appendix/*.tex` | 0.19 h + 2.46 h |
| Probability discount (exploratory) | `d2` | `d2/*.csv` | `appendix/probability_discount.tex` | 0.67 h |
| Earlier reproduction check (pre-registered) | — | `history/reproduction_check.csv` | `appendix/negative_controls.tex` | — |

GPU times are from the original runs: NVIDIA L4, except parts of the single-layer restriction (DeiT-B), head versus body and the centre-Gaussian ClickMe number, which ran on an NVIDIA A100.

## Reproduction checks

- **Full rerun.** On 2026-11-08 every experiment was rerun from a fresh clone, in a fresh environment, on one NVIDIA L4 (11.5 GPU hours), following the README.
  - All 31 summary tables agreed with the committed ones within tolerance, and every verdict was identical.
  - The largest difference of a mean was 2.5e-5 (map similarity); 26 tables were bit-identical.
  - Tolerances: means 1e-4; readings with a retrained probe 2e-3; execution agreement 0.1 points; probe accuracy 0.5 points.
  - Report: `docs/verification/rerun_2026-11-08.csv`; script: `docs/verification/compare_tables.py`.
- **Batch size is part of one definition.** The image-specific ClickMe control pairs each image with the previous image of its batch, so the batch size matters. `clickme_layers` uses batch 128 / 64 / 128 (DeiT-S / DeiT-B / DINO ViT-S), as in the original runs.
- **Tests.** `pytest` runs 78 tests on CPU. `bash reproduce/smoke.sh` runs every experiment on 16 images (about 4 minutes on an L4).
