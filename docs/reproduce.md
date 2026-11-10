# Reproduce the paper

The supplied paper is v8, `paper/main.pdf` (11 pages). All paths below are relative to the repository root.
The map covers the main text and Appendices A-E. R0 is received attention; C1 is mean-map subtraction;
C2 is probe relocation; C3 is contribution weighting; R_all applies all three.

## Commands

| Code | Command | Use |
| --- | --- | --- |
| R | `bash reproduce/from_outputs.sh` | Rebuild to a fresh build directory on CPU; preserve originals |
| E | `python scripts/run.py --config configs/<name>.yaml --output outputs/rerun` | Rerun one config, with external data, checkpoints and a CUDA GPU |
| F | `bash reproduce/full.sh outputs/rerun build/gpu-rerun` | Optional future GPU pathway, outside release acceptance |
| P | `python -m pytest -q` | CPU tests; fake data and random weights, no downloads |

E writes `outputs/rerun/<name>/[<dataset>/]<model>/`. To summarise an optional future rerun, use
`bash reproduce/from_outputs.sh outputs/rerun build/gpu-rerun`. R without an argument reads the original committed outputs and writes only to fresh `build/rebuilt/`.
Never overwrite those original outputs with a new experiment run.

## Figures, tables and empirical claims

| Paper item / claim | Config and dependencies | Original outputs | Result or source file | Command |
| --- | --- | --- | --- | --- |
| Figure 1a: fast-weight network | Static drawing | None | `figures/schematic_fast_weight_network.{pdf,png}` | R (static schematic and execution source unchanged) |
| Figure 2; Table B.1; main text premise and map-quality percentages | `diagnose_probe`, `diagnose_mean_maps` (needs probes), `high_norm`, `clickme_layers` | Corresponding folders in `outputs/` | `results/diagnose.csv`; `paper/figures_own/diagnosis.pdf`; `paper/tables_own/diagnostics_per_layer.tex` | E for each config, then R |
| Execution: k=16 agreement 72/83/61%, random 9/30/4%; Figure B.1 and every-k comparison | `k_sweep`, `attention_off` | `outputs/k_sweep/`, `outputs/attention_off/` | `results/k_sweep.csv`, `results/k_sweep_tests.csv`; `figures/execution_k_sweep.{pdf,png}` | E for both, then R |
| Mean maps 82-98% radial in deep layers | `diagnose_mean_maps`, needs `diagnose_probe` | `outputs/diagnose_mean_maps/<model>/radial.csv` | `results/radial.csv` | E, then R |
| High-norm value-vector ratios 0.29 and 0.23 | `value_norm` | `outputs/value_norm/<model>/high_norm.csv.gz` | `results/value_norm.csv`, rows with config=value_norm and layer=9-12 mean | E, then R |
| Table 2: readings, controls, GAE and paired verdicts | `table1`, needs probes and mean maps | `outputs/table1/{imagenet_s,clickme}/` | `results/table1_long.csv`, `results/correct_tests.csv`; `paper/tables_own/main_results.tex` | E, then R |
| Table 2 model-side rows; Appendix B variant differences and intervals | `table1_register`, `table1_mita`; register search and lens dependencies | `outputs/table1_register/`, `outputs/table1_mita/` | `results/variants.csv` | E for both, then R |
| Table B.2: paired differences and intervals | `table1`, lens dependencies | `outputs/table1/` | `results/correct_tests.csv` -> `paper/tables/appendix/paired_differences.tex` -> `paper/tables_own/paired_differences.tex` | R |
| Table 1: premises, proxies, corrections, controls and post-hoc outcomes | Diagnostics plus `table1` | Diagnostic and main-result folders above | `results/diagnose.csv`, `results/correct_tests.csv`; the interpretive table is written in `paper/main.tex` | R; inspect the named diagnostics and paired comparisons |
| Last-layer recovery: enrichment 62/33%, ClickMe 31/45%, deletion 8/29% | `table1` | `outputs/table1/` | `results/last_layer.csv`, reading=C3 | E, then R |
| DINO changes at most 0.004; early relocation changes at most 0.004 | `table1` | `outputs/table1/` | `results/negative_controls.csv`, tests P4a/P4b | E, then R |
| Contribution gains track high-norm attention: 0.075/0.132 enrichment, 0.045/0.158 ClickMe | `supplementary_eval`, needs lens and `supplementary_heldout` | `outputs/supplementary_eval/` | `results/supplementary/dose.csv`, improvement=C3 - R0 | E, then R |
| Shuffled-weight comparisons; value-only weighting differences -0.010 to 0.000 | `supplementary_eval`, same dependencies | `outputs/supplementary_eval/` | `results/supplementary/c3_mechanism.csv`, tests C3 - C3shuffle and C3 - C3v | E, then R |
| Table C.1: 2-29% far error mass; layer-12 neighbourhood probabilities 45/66/81% | `c2_analysis`, needs probes and mean maps | `outputs/c2_analysis/<model>/probe_errors.csv.gz` | `results/c2_analysis/errors.csv`, source=probe, metric=far_share/near3x3 | E, then R |
| Table C.1: probe mean prediction peak 1.1-1.3, weak relation to mean map | `c2_analysis`, same dependencies | `outputs/c2_analysis/<model>/mean_prediction.npz` and held-out mean maps | `results/c2_analysis/mean_prediction.csv`, max_ratio and correlations | E, then R |
| Table C.1: MLP accuracy gains 1.8/3.8/6.7 points, relocation lower in six cells | `supplementary_heldout`, `supplementary_eval`, lens dependencies | Corresponding folders in `outputs/` | `results/supplementary/mlp_probe.csv`; `results/supplementary/c2_variants.csv`, test=C2MLP - R0 | E for both, then R |
| Table C.1: globality and probe accuracy/entropy at layers 10-12 | `supplementary_heldout`, needs probes | `outputs/supplementary_heldout/<model>/global.csv` | `results/supplementary/globalisation.csv` | E, then R |
| Table C.1: hand-written summary | The four analyses above | As above | `paper/tables_own/relocation_explanations.tex` | R rebuilds its evidence; preserve this table as supplied |
| Figures 1b / D.1: first two / next four fixed-seed examples | `table1`, ClickMe dataset | `outputs/table1/clickme/deit/gallery_*.npz` | `paper/figures_own/example_maps.pdf`, `example_maps_more.pdf`; `figures/example_maps{,_more}.png`; `figures/gallery_selection.json` | E, then R |
| Appendix A: centre-Gaussian raw ClickMe correlation 0.41; two excluded duplicates | `table1`, `diagnose_probe` | `outputs/table1/clickme/`, `outputs/diagnose_probe/<model>/heldout.json` | `results/numbers.csv`; fixed held-out list in `data/splits/heldout_4000.json` | E for both, then R |
| Appendix A: random/radial controls near zero and Gaussian deletion 0.110/0.099/0.140 | `table1` | `outputs/table1/` | `results/table1_long.csv`, methods random/centre/mean_map | E, then R |
| Appendix A: blur widths 0.7-2.9 patches | `diagnose_probe` | `outputs/diagnose_probe/<model>/probe_accuracy.csv` | `results/probe_accuracy.csv`, column sigma | E, then R |
| Appendix A: selected register units; 92.75% reduction, 99.0% agreement; DeiT-B 44.9% | `register`; selected units in `table1_register` | Historical pilot summary, with provenance in `results/register/README.md` | `results/register/pilot_gate.csv` | E repeats the DeiT-S search; inspect n8/n32 columns of the supplied summary |
| Appendix A: formal register reduction 91.5%, accuracy 79.8% | Historical formal register check | Historical summary, same provenance guide | `results/register/register_deit.csv`, R0 vs R2 | Inspect the supplied CSV; this historical full-validation check is not rebuilt by the current runner |
| Sample counts, splits, models, learning rate, epochs, seeds and bootstrap rules | The named configs, `src/fastweight_vit/models/`, `src/fastweight_vit/stats.py`, `src/fastweight_vit/probes.py` | Counts in per-image outputs and held-out lists | `data/splits/*.json`; n columns in the result tables; `configs/*.yaml` | P and the strict audit check lists/protocols; historical records preserve effective settings |
| Appendix E: received/contribution reading and relocation | Implementations, no separate run | Covered by main-result runs | `src/fastweight_vit/readings.py` and `src/fastweight_vit/probes.py` | P |

Appendix A's early top-k comparison (S -0.011, Tiny +0.037) comes from `study/v5`, commit `1d66c0f7b39119e3ee7777ad7d086df68769d427`, `docs/analysis/v5_report.md` (the original B32 judgment); that earlier study is not included here.

## Registered outcomes (Table A.1)

| Paper label | Historical rule | Numeric evidence | Observed outcome |
| --- | --- | --- | --- |
| R1 | P4a/P4b | `results/negative_controls.csv` | 3/3 DINO cells and 6/6 early-layer cells comparable |
| R2 | P3 | `results/correct_tests.csv`, R_all - R0 | 1/4 supervised centre-blind cells higher: DeiT-B ClickMe; partly holds |
| R3 | P6 | `results/variants.csv`, variant - DeiT-S R0 | 4/4 centre-blind cells not higher |
| Execution / diagnostics | P1/P2 | `results/k_sweep_tests.csv`, `results/diagnose.csv` | Registered replications; earlier observations had been seen |

The historical record and dates are in [preregistration.md](preregistration.md). Single-correction directions were
not registered; diagnosis–correction correspondence is post hoc. Table 1, Table A.1 and Table C.1 are authored
interpretations. The numerical Table 2, Table B.1 and Table B.2 are generated from reference CSVs. The strict audit
checks the 1/4 outcome, negative controls, practical margins, eight register units and the source evidence for
the manual interpretations. The complete machine-readable mapping is [evidence.json](evidence.json).

## Statistics and validation

One value per image; paired bootstrap over images, 1,000 resamples, seed 0, 95% percentile intervals.
A difference is comparable if its interval contains zero or its magnitude is below 0.02
(0.05 for deletion); otherwise its sign gives higher or lower. These rules are in `scripts/make_tables.py`.
Batch size matters for the shuffled ClickMe pairing; keep the config values.

The cleanup keeps the original results, outputs and figure files unchanged. CPU rebuilds must agree within
absolute error 1e-12, with identical verdicts and figure content. The paired source and three generated paper
tables must agree byte for byte. Fonts and rasterisation can cause small rendering differences.
The supplied paper PDF is preserved exactly. No GPU rerun or probe training is part of this release preparation; acceptance uses CPU artifact verification.

The historical register summaries above preserve cited evidence. Their old formal accuracy measurement is
not claimed to be reproduced by the current map-evaluation pipeline. The other historical top-k comparison
has the explicit external source given above.


## Release checks and known manuscript wording differences

`python scripts/validate_release.py --strict --rebuild --archive-check --report build/verification.json`
audits the artifacts, rebuilds in a temporary directory, compares every generated table and verifies a real ZIP.
Originals must not change. Missing/corrupt/duplicate data and changed protocols fail. CPU numeric tolerance is
absolute 1e-12; discrete fields and verdicts are exact. Cross-platform rendering differences are separate from
numeric equivalence. See [provenance](provenance/README.md) for original-run records, import transformations and
which earlier revisions the authentic GPU verification covers. The optional GPU scripts were not executed here.

Two author wording changes remain proposed, without changing the supplied manuscript or algorithms:
(1) Appendix A object patches use at least half of the **non-ignored pixels**, and require at least one valid pixel;
(2) the supporting value-norm ratio uses uniform crop 0.875, unlike the main DeiT crop 0.9.
