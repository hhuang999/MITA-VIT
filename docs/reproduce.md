# Reproduction map

**What this file holds:** for every table, figure and number of the paper, the command that produces it, the
output file, and the result of the three reproduction checks; then the map from the names used during the study
(branch `study/v5`) to the names used here.

- Runs: `python scripts/run.py --config configs/<config>.yaml` (all models and datasets of the config), outputs in
  `outputs/<config>/[<dataset>/]<model>/`. Tables: `python scripts/make_tables.py` → `results/`. Figures:
  `python scripts/make_figures.py` → `figures/`. `reproduce/full.sh` runs everything in dependency order.
- **Layer 1** (rebuild from saved outputs): the per-image outputs of the original study runs, laid out as release
  outputs (`docs/release/import_study_outputs.py`), rebuilt with `make_tables.py` / `make_figures.py` and compared
  with the study's tables (`docs/release/compare_layer1.py`). "Identical" = same rows, largest absolute difference 0.
- **Layer 2** (full rerun from a fresh clone with `reproduce/full.sh`): tolerances — means 1e-4; every verdict
  identical; execution agreement 0.1 percentage points; retrained probe accuracy 0.5 points; means of readings that
  use the probe (C2 and its combinations) 0.002.
- **Layer 3** (fresh environment): `pytest` and `reproduce/smoke.sh` (16 images).
- GPU time: of the original runs (NVIDIA L4, some on A100/H100 for the older ones), from `elapsed_seconds`.

## Paper elements

| paper element | config(s) | output file(s) in `results/` | layer 1 | layer 2 (largest difference) | GPU time of the original runs |
| --- | --- | --- | --- | --- | --- |
| Figure 2(a), section 4.2 (P1) | fig2a_k_sweep, attention_off | k_sweep.csv, k_sweep_tests.csv; figures/figure_2a | identical | in progress | 0.60 h + 0.16 h |
| Figure 2(b); appendix figure (layer 12) | table1 (ClickMe gallery of DeiT-S) | figures/figure_2b, figure_A_2b_layer12 | identical (pixel) | in progress | (with table1) |
| Figure 3(a), M1; mean-map ring share | diagnose_probe, diagnose_mean_maps | diagnose.csv (panel M1), radial.csv | identical | in progress | 0.09 h + 0.33 h |
| Figure 3(b), M2; probe accuracy, top probability | diagnose_probe | diagnose.csv (M2), probe_accuracy.csv, numbers.csv | identical | in progress | (above) |
| Figure 3(c), M3 | high_norm | diagnose.csv (M3) | identical (from the earlier per-layer run, see note 1) | in progress | 2.08 h (earlier run) |
| Figure 3(d), R0 vs ClickMe per layer | clickme_layers | diagnose.csv (quality) | identical (from the earlier ClickMe run, note 1) | in progress | 2.58 h (earlier run) |
| P2 | (figure 3) | diagnose_tests.csv | identical | in progress | — |
| Value-norm ratio (section 4.3) | value_norm | value_norm.csv | identical (0.292 / 0.230) | in progress | 0.90 h (earlier run) |
| Table 1, sections 4.2 and 4.4, appendix D tables | table1 | table1.md, table1.csv, table1_long.csv, correct_tests.csv | identical | in progress | 3.05 h |
| Table 1 "change the model" rows, P6 | table1_register, table1_mita | variants.csv | identical | in progress | 0.13 h + 0.13 h |
| Last-layer recovery (P5) | table1 | last_layer.csv | identical | in progress | — |
| The other combination order | order | order.csv | identical | in progress | 0.56 h |
| Negative controls (P4) | table1 | negative_controls.csv, correct_tests.csv | identical | in progress | — |
| All predictions | — | predictions.csv | identical | in progress | — |
| Sanity check (appendix C) | sanity | sanity.csv | identical | in progress | 0.39 h |
| Appendix B: one layer restricted / attention off | single_layer, attention_off | single_layer.csv | means identical; intervals differ by ≤ 0.0012 (note 2) | in progress | 1.60 h (earlier run) |
| Appendix B: head vs body patches | part_reliance | part_reliance.csv | identical | in progress | 1.37 h (earlier run) |
| Appendix B: top-k map vs full map | map_similarity | map_similarity.csv | identical (recomputed from saved scores) | in progress | 1.08 h (earlier run) |
| Register units of "+register" | register | outputs/register/deit/gate.json | not rebuilt (units copied into configs/table1_register.yaml) | in progress | 0.02 h |
| Appendix E: why C2 fails; appendix figure | c2_analysis | c2_analysis/*.csv; figures/figure_A_c2 | identical (figure: same data, visually identical) | in progress | 0.32 h |
| Appendices D and E: supplementary analyses | supplementary_heldout, supplementary_eval | supplementary/*.csv | identical | in progress | 0.19 h + 2.46 h |
| Appendix E: discount without relocation | d2 | d2/*.csv | identical | in progress | 0.9 h (2026-11-08 runs) |
| Introduction: centre Gaussian vs ClickMe (0.41) | table1 | numbers.csv | identical to the earlier ClickMe run (0.4141) | in progress | — |
| Section 4.1: ClickMe near-duplicates among held-out candidates | diagnose_probe | numbers.csv, data/splits/heldout_4000.json | identical | in progress | — |
| Historical reproduction check (appendix C) | — | history/reproduction_check.csv | kept as a record (the earlier code is not here) | — | — |

Notes.
1. In the study, figure 3(c) and 3(d) came from earlier runs that computed dense attention mass per layer; here
   they come from the R0 reading per layer (`high_norm`, `clickme_layers`). Dense attention mass and R0 differ by a
   positive factor only, so the shares and rank correlations are the same; layer 1 rebuilt the tables from the
   earlier runs' saved per-image values, layer 2 reruns them with the new code.
2. The study took these intervals from an earlier analysis that resampled the images in another row order; the
   means are identical and the intervals differ by at most 0.0012.

## Old names → new names

| study/v5 | here |
| --- | --- |
| package `mita_probe` | `fastweight_vit` |
| `mita_probe.probe` (top-k reading sets) | `fastweight_vit.execution` |
| `identity.py` (position probe), probe classes in `readings.py` and the MLP probe | `probes.py` |
| `explain.py` (GAE, centre, random, attention mass, probe map) | `references.py` (`topk_map` = the old probe 'mass' map) |
| `plausibility.py` | `enrichment.py` |
| `interventions.sink_tokens` | `tokens.high_norm_tokens` |
| `budget.instrument` | `instrument.instrument` |
| `experiments/datasets.py` | `data/build.py` |
| readings `blur`, `no_high`, `centre_sub` | `stand_in_blur`, `stand_in_no_high`, `stand_in_centre` |
| supplementary maps `no_high_l<l>`, `blurMLP` | `stand_in_no_high_l<l>`, `stand_in_blur_MLP` |
| experiment `anatomy` stage `rq2` (k sweep); runs `read_ksweep_*` | `execution` stage `k_sweep`; config `fig2a_k_sweep` |
| experiment `anatomy` stage `rq1` (setting L3 only); runs `v2_rq1`, `v2_dino_rq1`, `v3_deit_b_rq1` | `execution` stage `single_layer`; config `single_layer` |
| experiment `attention_off` (T7); runs `ab_t7_*` | `execution` stage `attention_off`; config `attention_off` |
| `diagnose` stages `probe`, `m1`; runs `diagnose_probe_*`, `diagnose_m1_*` | stages `probe`, `mean_maps`; configs `diagnose_probe`, `diagnose_mean_maps` |
| `identity.csv`, `identity_images.csv` | `probe_accuracy.csv`, `probe_accuracy_images.csv` |
| `layer_reading` (S3; family 'attention'); runs `s_layers_*` | `diagnose` stage `high_norm`; config `high_norm` |
| value-norm ratio of the budget study (T3, `heads.csv`); runs `ab_t3_*` | `diagnose` stage `high_norm`; config `value_norm` |
| `human_alignment` (E7/S2, 'attention_layer<l>'); runs `s_e7_clickme_*` | `correct` with R0_l1–R0_l12; config `clickme_layers` |
| `correct` runs `correct_ablation_*`, `correct_register_*`, `correct_mita_*`, `correct_order_*` | configs `table1`, `table1_register`, `table1_mita`, `order` |
| `sanity` runs `correct_sanity_*` | config `sanity` |
| `parts` (E8, reliance part); runs `v3_e8_parts_*` | `part_reliance` |
| S1 similarity (`explanations`, probe_mass vs attention_mass); runs `s_e4_imagenet_s_*` | `map_similarity` |
| `register` stage `search` (T2); run `ab_t2_deit`, `gate.json` | `register_search`; config `register`; units in `table1_register.yaml` |
| `explore_c2` (E-C2); runs `explore_c2_*`, `explore_c2_probefix_*` | `c2_analysis` |
| `supp` (X1–X5) stages `heldout`, `eval`; runs `supp_heldout_*`, `supp_eval_*` | `supplementary`; configs `supplementary_heldout`, `supplementary_eval` |
| D2 (B50); runs `explore_d2_*` | `correct` with D2 readings; config `d2` |
| analysis folders `v5_analysis` (A/), `explore_c2_analysis` (E/), `v5_supp_analysis` (S/) | `results/`, `results/c2_analysis/`, `results/supplementary/` |
| `fig2a_ksweep.csv`, `p1_tests.csv`, `fig3_diagnose.csv`, `a2_single_layer.csv`, `a6_last_layer.csv`, `a7_order.csv` | `k_sweep.csv`, `k_sweep_tests.csv`, `diagnose.csv`, `single_layer.csv`, `last_layer.csv`, `order.csv` |
| `x1.csv` … `x6.csv`, `x2_seeds.csv`, `x3_layers.csv`, `x3_dose.csv`, `x4_layers.csv`, `x5_probe.csv`, `x5.csv` | `combinations.csv`, `c3_mechanism.csv`, `c3_mechanism_seeds.csv`, `dose_layers.csv`, `dose.csv`, `globalisation.csv`, `mlp_probe.csv`, `c2_variants.csv`, `gae_gap.csv` |
| branch labels X1-A/B, X3-A/B, M-weak, H-global | A/B, A/B, weak, holds / does not hold |
| figures `figure_v5_A_c2` | `figure_A_c2` |
| decision ids B05–B50, predictions P1–P7, extensions B/C/E, appendix ids A2/A6/A7 | `docs/preregistration.md` (P1–P7 kept) |
| `colab/` scripts, `configs/v5.yaml` and its base chain | `reproduce/`, flattened configs |
