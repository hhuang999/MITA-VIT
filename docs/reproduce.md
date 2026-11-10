# Reproduce the paper

The final paper is `paper/main.pdf`. All paths below are relative to the repository root.
The map covers the main text and Appendices A-E. R0 is received attention; C1 is mean-map subtraction;
C2 is probe relocation; C3 is contribution weighting; R_all applies all three.

## Commands

| Code | Command | Use |
| --- | --- | --- |
| R | `bash reproduce/from_outputs.sh` | Rebuild the paper summaries, figures and generated tables on CPU |
| E | `python scripts/run.py --config configs/<name>.yaml --output outputs/rerun` | Rerun one config, with external data, checkpoints and a CUDA GPU |
| F | `bash reproduce/full.sh` | Rerun the fourteen paper configs in dependency order, then R |
| P | `python -m pytest -q` | CPU tests; fake data and random weights, no downloads |

E writes `outputs/rerun/<name>/[<dataset>/]<model>/`. To summarise a rerun, use
`bash reproduce/from_outputs.sh outputs/rerun`. R without an argument uses the original committed outputs.
Never overwrite those original outputs with a new experiment run.

## Figures, tables and empirical claims

| Paper item / claim | Config and dependencies | Original outputs | Result or source file | Command |
| --- | --- | --- | --- | --- |
| Figure 1: fast-weight network | Static drawing | None | `figures/schematic_fast_weight_network.{pdf,png}` | `python scripts/make_figures.py` |
| Figure 2; Table B.1; main text premise and map-quality percentages | `diagnose_probe`, `diagnose_mean_maps` (needs probes), `high_norm`, `clickme_layers` | Corresponding folders in `outputs/` | `results/diagnose.csv`; `paper/figures_own/diagnosis.pdf`; `paper/tables_own/diagnostics_per_layer.tex` | E for each config, then R |
| Execution: k=16 agreement 72/83/61%, random 9/30/4%; Figure B.1 and every-k comparison | `k_sweep`, `attention_off` | `outputs/k_sweep/`, `outputs/attention_off/` | `results/k_sweep.csv`, `results/k_sweep_tests.csv`; `figures/execution_k_sweep.{pdf,png}` | E for both, then R |
| Mean maps 82-98% radial in deep layers | `diagnose_mean_maps`, needs `diagnose_probe` | `outputs/diagnose_mean_maps/<model>/radial.csv` | `results/radial.csv` | E, then R |
| High-norm value-vector ratios 0.29 and 0.23 | `value_norm` | `outputs/value_norm/<model>/high_norm.csv.gz` | `results/value_norm.csv`, rows with config=value_norm and layer=9-12 mean | E, then R |
| Table 1: readings, controls, GAE and paired verdicts | `table1`, needs probes and mean maps | `outputs/table1/{imagenet_s,clickme}/` | `results/table1_long.csv`, `results/correct_tests.csv`; `paper/tables_own/main_results.tex` | E, then R |
| Table 1 model-side rows; Appendix B variant differences and intervals | `table1_register`, `table1_mita`; register search and lens dependencies | `outputs/table1_register/`, `outputs/table1_mita/` | `results/variants.csv` | E for both, then R |
| Table B.2: paired differences and intervals | `table1`, lens dependencies | `outputs/table1/` | `results/correct_tests.csv` -> `paper/tables/appendix/paired_differences.tex` -> `paper/tables_own/paired_differences.tex` | R |
| Table 2: does diagnosis predict correction? | Diagnostics plus `table1` | Diagnostic and main-result folders above | `results/diagnose.csv`, `results/correct_tests.csv`; the interpretive table is written in `paper/main.tex` | R; inspect the named diagnostics and paired comparisons |
| Last-layer recovery: enrichment 62/33%, ClickMe 31/45%, deletion 8/29% | `table1` | `outputs/table1/` | `results/last_layer.csv`, reading=C3 | E, then R |
| DINO changes at most 0.004; early relocation changes at most 0.004 | `table1` | `outputs/table1/` | `results/negative_controls.csv`, tests P4a/P4b | E, then R |
| Contribution gains track high-norm attention: 0.075/0.132 enrichment, 0.045/0.158 ClickMe | `supplementary_eval`, needs lens and `supplementary_heldout` | `outputs/supplementary_eval/` | `results/supplementary/dose.csv`, improvement=C3 - R0 | E, then R |
| Shuffled-weight comparisons; value-only weighting differences -0.010 to 0.000 | `supplementary_eval`, same dependencies | `outputs/supplementary_eval/` | `results/supplementary/c3_mechanism.csv`, tests C3 - C3shuffle and C3 - C3v | E, then R |
| Table C.1: 2-29% far error mass; layer-12 neighbourhood probabilities 45/66/81% | `c2_analysis`, needs probes and mean maps | `outputs/c2_analysis/<model>/probe_errors.csv.gz` | `results/c2_analysis/errors.csv`, source=probe, metric=far_share/near3x3 | E, then R |
| Table C.1: probe mean prediction peak 1.1-1.3, weak relation to mean map | `c2_analysis`, same dependencies | `outputs/c2_analysis/<model>/mean_prediction.npz` and held-out mean maps | `results/c2_analysis/mean_prediction.csv`, max_ratio and correlations | E, then R |
| Table C.1: MLP accuracy gains 1.8/3.8/6.7 points, relocation lower in six cells | `supplementary_heldout`, `supplementary_eval`, lens dependencies | Corresponding folders in `outputs/` | `results/supplementary/mlp_probe.csv`; `results/supplementary/c2_variants.csv`, test=C2MLP - R0 | E for both, then R |
| Table C.1: globality and probe accuracy/entropy at layers 10-12 | `supplementary_heldout`, needs probes | `outputs/supplementary_heldout/<model>/global.csv` | `results/supplementary/globalisation.csv` | E, then R |
| Table C.1: hand-written summary | The four analyses above | As above | `paper/tables_own/relocation_explanations.tex` | R rebuilds its evidence; preserve this table as supplied |
| Figure D.1: two randomly drawn example images | `table1`, ClickMe dataset | `outputs/table1/clickme/deit/gallery_*.npz` | `figures/gallery_layers9_12.{pdf,png}` | E, then R |
| Appendix A: centre-Gaussian raw ClickMe correlation 0.41; two excluded duplicates | `table1`, `diagnose_probe` | `outputs/table1/clickme/`, `outputs/diagnose_probe/<model>/heldout.json` | `results/numbers.csv`; fixed held-out list in `data/splits/heldout_4000.json` | E for both, then R |
| Appendix A: random/radial controls near zero and Gaussian deletion 0.110/0.099/0.140 | `table1` | `outputs/table1/` | `results/table1_long.csv`, methods random/centre/mean_map | E, then R |
| Appendix A: blur widths 0.7-2.9 patches | `diagnose_probe` | `outputs/diagnose_probe/<model>/probe_accuracy.csv` | `results/probe_accuracy.csv`, column sigma | E, then R |
| Appendix A: selected register units; 92.75% reduction, 99.0% agreement; DeiT-B 44.9% | `register`; selected units in `table1_register` | Historical pilot summary, with provenance in `results/register/README.md` | `results/register/pilot_gate.csv` | E repeats the DeiT-S search; inspect n8/n32 columns of the supplied summary |
| Appendix A: formal register reduction 91.5%, accuracy 79.8% | Historical formal register check | Historical summary, same provenance guide | `results/register/register_deit.csv`, R0 vs R2 | Inspect the supplied CSV; this historical full-validation check is not rebuilt by the current runner |
| Sample counts, splits, models, learning rate, epochs, seeds and bootstrap rules | The named configs, `src/fastweight_vit/models/`, `src/fastweight_vit/stats.py`, `src/fastweight_vit/probes.py` | Counts in per-image outputs and held-out lists | `data/splits/*.json`; n columns in the result tables; `configs/*.yaml` | P checks the fixed lists; E records effective settings |
| Appendix E: received/contribution reading and relocation | Implementations, no separate run | Covered by main-result runs | `src/fastweight_vit/readings.py` and `src/fastweight_vit/probes.py` | P |

Appendix A's early top-k comparison (S -0.011, Tiny +0.037) comes from `study/v5`, commit `1d66c0f7b39119e3ee7777ad7d086df68769d427`, `docs/analysis/v5_report.md` (the original B32 judgment); that earlier study is not included here.

## Statistics and validation

One value per image; paired bootstrap over images, 1,000 resamples, seed 0, 95% percentile intervals.
A difference is comparable if its interval contains zero or its magnitude is below 0.02
(0.05 for deletion); otherwise its sign gives higher or lower. These rules are in `scripts/make_tables.py`.
Batch size matters for the shuffled ClickMe pairing; keep the config values.

The cleanup keeps the original results, outputs and figure files unchanged. CPU rebuilds must agree within
absolute error 1e-12, with identical verdicts and figure content. The paired source and three generated paper
tables must agree byte for byte. Fonts and rasterisation can cause small rendering differences.
The supplied paper PDF is preserved exactly. No GPU rerun was part of this cleanup.

The historical register summaries above preserve cited evidence. Their old formal accuracy measurement is
not claimed to be reproduced by the current map-evaluation pipeline. The other historical top-k comparison
has the explicit external source given above.
