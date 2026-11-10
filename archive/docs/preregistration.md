# Pre-registration

**What this file holds:** the predictions P1–P7 of the paper with their decision rules, when each rule was fixed
(date and commit), and the outcome. Exploratory analyses, proposed after the main results, are listed separately
with the rules registered before they were run. This file replaces the full Chinese decision log of the study,
which stays on the `study/v5` branch.

Statistics for every comparison: one value per image, paired bootstrap over images (1,000 resamples, seed 0),
95% percentile intervals. Verdict: "comparable" if the interval contains 0 or |mean| < margin; otherwise "higher" or
"lower". Margins: SRG 0.05; same-ring enrichment 0.02; ClickMe image-specific Spearman 0.02.

## Registered predictions

The full registration log (`docs/decisions.md`, Chinese) and the execution specification (`docs/experiment_plan_v5.md`) are on the `study/v5` branch; the commits below are part of this branch's history. Each entry was committed before the runs it governs. Commit times and run start times are UTC (`git log`; `timestamp` in each run's `metadata.json`).

### Fixed when What was fixed when

| Date (UTC) | Commit | Content fixed | Runs governed |
| --- | --- | --- | --- |
| 2026-10-29 07:38 | `90e78d5` | Earlier study: SRG and same-ring enrichment definitions and margins; ClickMe protocol (pixel-level Spearman, other-image control); PartImageNet head-vs-body prediction; sanity-check procedure and threshold (median < 0.5); qualitative-gallery rule; statistics | ClickMe run from 2026-10-29 11:56; head-vs-body runs from 12:40 |
| 2026-10-31 06:32 | `a96a7aa` | Full-attention map (R0) vs top-k map comparison and its margins (0.05 SRG, 0.02 enrichment and ClickMe); per-layer curves; high-norm share definition | Runs from 2026-10-31 07:24 |
| 2026-11-03 09:37 / 11:35 | `2cae9af`, `966cdb3` | Attention-budget study: register unit search gate, uniform preprocessing, value-norm ratio threshold (< 0.5); attention-removal test | Runs from 2026-11-03 11:21 / 12:21 |
| 2026-11-05 10:02 | `b8bc945` | v5 positioning (read → diagnose → correct) | — |
| 2026-11-05 10:07 | `c55feed` | v5 proposals: readings, stand-ins, references, data and held-out set, diagnostics, predictions P1–P6, order of runs and stop rule | — |
| 2026-11-05 11:19 | `e5e9b90` | Smoke-run findings (32–64 images per run), compute estimate, ten questions to the author | — |
| 2026-11-05 15:31 | `94bf426` | Author confirmation and freeze: margins and statistics, primary comparisons (P7), blur matching, held-out set, SRG for +register and none for MiTA, layer 1–4 control without SRG, zeroing as the C3 stand-in, ClickMe source to be checked | First formal run 2026-11-05 15:53 (code `94bf426`) |
| 2026-11-05 16:42 (committed 17:02) | `5faa503` | Implementation-check tolerance 1e-4 → 5e-4 (no research definition changed) | Remaining formal runs from 17:12 (code `5faa503`) |
| 2026-11-05 17:07 | `7584ed2` | ClickMe image-source correction (no experiment changed) | — |
| 2026-11-05 22:24 | `b35fcf6` | Formal results and verdict ledger | — |
| 2026-11-07 09:43 | `65aba6c` | Exploratory analysis of C2 (probe errors, mean prediction, where C2 moves mass; ClickMe smoothing compilation): definitions and descriptive criteria | Runs 2026-11-07 10:17 (code `ebfd25f`) |
| 2026-11-07 10:30 | `e5894a5` | Exploratory supplementary analyses (pairwise combinations, C3 checks, dose–response, globalisation, MLP probe and C2 variants, GAE gap, figures): definitions and branch rules | Runs 2026-11-07 11:08–13:49 (code `534e333`) |
| 2026-11-07 10:54 | `7015d89` | Fix of the float32 far-share computation; probe-only rerun | Rerun 2026-11-07 13:48 |
| 2026-11-07 14:41 | `9453e46` | Exploratory results and branch outcomes | — |
| 2026-11-08 08:12 | `81bad1c` | Author decisions after the exploratory results (Section 4.5 of the paper as "version B"; Figure 2(b) on layers 9–12) | — |
| 2026-11-08 09:19 | `d9bae2d` | Discount without relocation: definition, stand-in, branch rule | Running at the time of writing |

Code commits of the formal runs (`git_commit` in `metadata.json`): `94bf426` for the 3 probe runs, 3 mean-map runs and the DeiT-S and DINO k sweeps; `5faa503` for all other formal runs. The two commits differ only in the implementation-check tolerance.

### Results seen before registration Results seen before registration (disclosed in the plan)

- The DeiT-S k sweep (same setting, k = 4–128) had been seen in an earlier run; P1 is a replication for DeiT-S.
- The earlier full-attention, per-layer and high-norm-share results and the earlier position probe (400 training images) had been seen; the M2 and M3 parts of P2 are replications.
- The attention-budget results (including the value-norm ratios on the pilot set, before its threshold was registered) had been seen.
- Smoke runs of 32–64 images per run were inspected for code checks.
- All exploratory analyses were proposed after the formal results.

### Predictions P1–P7 and outcomes Predictions P1–P7 (registered text, translated) and results

Text from `docs/experiment_plan_v5.md` §5 and the author confirmation in `docs/decisions.md` (2026-11-05), translated from Chinese.

| Prediction | Registered text | Registered decision rule | Type |
| --- | --- | --- | --- |
| P1 | Under the main setting, execution agreement rises monotonically with k; at every k the reading sets are above the random sets. | Per model: the difference between adjacent k (larger − smaller) has interval lower bound > 0 (5 tests); at every k, (reading − random) has lower bound > 0 (6 tests). All satisfied: holds. | Replication (DeiT-S seen earlier) |
| P2 | M2 becomes more severe with depth; M3 concentrates in the middle and late layers of supervised models and is close to 0 in DINO; M1 is described only. | M2: layer-12 accuracy interval upper bound < layer-1 interval lower bound, and Spearman(layer index, accuracy) ≤ −0.8 (each model). M3 (R0 share from the earlier run): DINO's high-norm share has interval upper bound < 0.01 at every layer; for DeiT-S and DeiT-B the layer with the largest share lies in layers 6–12. | Replication + description |
| P3 (main verdict) | With layers 9–12 combined, R_all is above R0 on same-ring enrichment and image-specific Spearman in the supervised models. | Four cells (DeiT-S, DeiT-B × two metrics), R_all − R0: all four "higher" = holds; 1–3 = partly holds; 0 = does not hold. DINO and SRG described only. Each correction's own contribution (C − R0) and its comparison with its stand-in (C − stand-in) are also reported. | Directional |
| P4 | Negative controls: C3 hardly changes the results on DINO; C2 hardly changes the results at layers 1–4. | P4a: DINO, layers 9–12, C3 − R0 "comparable" on all three metrics. P4b: three models, layers 1–4, C2 − R0 "comparable" on enrichment and ClickMe (SRG not used, author decision). Must hold; otherwise stop the queue and report. | Must hold |
| P5 | The share of the supervised models' last-layer drop that the corrections recover. | Recovered share = (corrected layer 12 − R0 layer 12) ÷ (R0 at the best of layers 8–11 by mean − R0 layer 12), paired bootstrap. | Descriptive |
| P6 | The "+register" and "top-k training" rows raise object enrichment by no more than the margin. | (1) Report the original verdicts of the attention-budget study as they are (register: holds, +0.006; top-k training: does not hold, Small −0.011, Tiny +0.037). (2) Under the Table 1 protocol: variant R0 − DeiT-S R0 is not "higher" on same-ring enrichment and on ClickMe. | Earlier result + check under the Table 1 protocol |
| P7 (primary comparisons) | The pre-specified primary comparisons are only two kinds: R_all − R0, and each correction − its stand-in (C1 − centred Gaussian subtraction, C2 − matched blur, C3 − zero high-norm tokens), judged for three models and three metrics (higher / comparable / lower), with no direction predicted. All other comparisons are descriptive. | Verdict rule of 2.9 | Primary, no direction |

Also registered: reproduction check (new R0 vs the earlier full-attention runs, |difference of means| < 0.01 on all three metrics; reported, no stop); sanity check (median Spearman < 0.5 after full randomisation = passes; R0, C1, C2, C3, R_all predicted to pass, centre as the failing example; mean maps and probes not re-estimated at each level); order check ("robust" if (other order − R0) and (R_all − R0) have the same verdict in every cell; descriptive).

**Pre-agreed reading of P3** (`docs/experiment_plan_v5.md` §5): (1) holds (4/4): the corrections improve the reading; a correction is said to exceed its stand-in only if correction − stand-in is also "higher". (2) partly holds (1–3/4): state where the corrections work and where not; do not generalise to "the corrections work". (3) does not hold (0/4): the lens diagnoses misreadings but reading corrections have limited effect.

| Prediction | Model(s) | Outcome | Detail |
| --- | --- | --- | --- |
| P1 | DeiT-S | holds | 11/11 tests (`results/k_sweep_tests.csv`) |
| P1 | DeiT-B | holds | 11/11 tests (`results/k_sweep_tests.csv`) |
| P1 | DINO | holds | 11/11 tests (`results/k_sweep_tests.csv`) |
| P2 (M2) | DeiT-S | holds | layer 12 upper 0.1830 < layer 1 lower 1.0000; Spearman(layer, accuracy) = −1.0000 |
| P2 (M2) | DeiT-B | holds | layer 12 upper 0.3203 < layer 1 lower 1.0000; Spearman(layer, accuracy) = −1.0000 |
| P2 (M2) | DINO | holds | layer 12 upper 0.3600 < layer 1 lower 0.9993; Spearman(layer, accuracy) = −0.9930 |
| P2 (M3) | DeiT-S | holds | largest share at layer 8 (0.0929) |
| P2 (M3) | DeiT-B | holds | largest share at layer 11 (0.4354) |
| P2 (M3) | DINO | holds | largest upper bound over layers = 0.0003 |
| P3 | DeiT-S, DeiT-B | partly holds | 1/4 cells higher; case: partial; pre-agreed reading (2) |
| P4a | DINO | holds | 3/3 comparable |
| P4b | DeiT-S,DeiT-B,DINO | holds | 6/6 comparable |
| P5 | DeiT-S, DeiT-B | descriptive | `results/last_layer.csv` |
| P6 | +register, top-k training | holds | 4/4 not higher |
| P7 | DeiT-S | no direction predicted | C1 vs stand-in: enrichment comparable, SRG comparable, ClickMe comparable |
| P7 | DeiT-S | no direction predicted | C2 vs stand-in: enrichment lower, SRG lower, ClickMe lower |
| P7 | DeiT-S | no direction predicted | C3 vs stand-in: enrichment higher, SRG comparable, ClickMe higher |
| P7 | DeiT-B | no direction predicted | C1 vs stand-in: enrichment higher, SRG higher, ClickMe higher |
| P7 | DeiT-B | no direction predicted | C2 vs stand-in: enrichment higher, SRG comparable, ClickMe comparable |
| P7 | DeiT-B | no direction predicted | C3 vs stand-in: enrichment higher, SRG comparable, ClickMe comparable |
| P7 | DINO | no direction predicted | C1 vs stand-in: enrichment comparable, SRG comparable, ClickMe comparable |
| P7 | DINO | no direction predicted | C2 vs stand-in: enrichment lower, SRG lower, ClickMe lower |
| P7 | DINO | no direction predicted | C3 vs stand-in: enrichment comparable, SRG comparable, ClickMe comparable |

Source: `results/predictions.csv`, `results/diagnose_tests.csv` (rebuilt by `scripts/make_tables.py`).

P2 detail values come from `results/diagnose.csv` (the rule is in `scripts/make_tables.py`, function `diagnose`). P6 part (1): the earlier verdicts (attention-budget study, on the `study/v5` branch: register on DeiT-S: 0.0057 [0.0047, 0.0068], "holds"; top-k training, Small: −0.0107 [−0.0131, −0.0084]; Tiny: 0.0374 [0.0347, 0.0402]; the hypothesis needed both pairs to hold and was recorded as "does not hold"; these intervals are Bonferroni 99.5% intervals with 10,000 resamples, on the 11,919-image formal split with uniform preprocessing).

## Exploratory analyses (proposed after the main results; rules registered before running)

None of these changes a verdict above. Results: `results/c2_analysis/`, `results/supplementary/`, `results/d2/`.

| analysis | registered (UTC, commit) | rule |
| --- | --- | --- |
| Why C2 fails: geometry of the probe's errors, its mean prediction, where C2 moves scores | 2026-11-07 09:43, `65aba6c`; far-share computation fixed before the probe rerun, `7015d89` | "Probe errors are mostly far": share of error mass farther than 3 patches > 0.5 (interval) at layers 9–12 in all models. "Fixed position preference": mean prediction max ≥ 2 × uniform and Pearson with the model's mean map ≥ 0.5 (lower bound > 0) in every model-layer cell, layers 9–12. |
| Pairwise combinations (C1 + C3, C2 + C3, C1 + C2) | 2026-11-07 10:30, `e5894a5` | Branch A if C1C3 − C3 is "higher" in ≥ 2 of the 4 supervised cells (enrichment, ClickMe) and "lower" in none; otherwise B. |
| C3 mechanism (‖v‖ only, weights shuffled within image, per-head mean weights) | same | "Expected" if C3 − shuffle and C3 − head-mean are "higher" in at least one cell of each supervised model and "lower" nowhere. |
| C3 dose–response and per-layer effects | same | Branch A if Spearman(share of R0 on high-norm tokens, C3 − R0) on ClickMe has interval lower bound > 0 in both supervised models. |
| Within-layer globalisation of tokens vs probe uncertainty | same | Holds if, at layers 10–12, the token-globality groups correlate negatively with probe accuracy and positively with prediction entropy (overall interval and within-ring means) in ≥ 2 models. |
| Two-layer MLP probe; C2 with confident tokens only | same | MLP "strong" if its layer-12 accuracy exceeds the linear probe's by > 10 points in some model; C2 rescued if C2-MLP − R0 is not "lower" in ≥ 4 of 6 cells. C2-confident: not "lower" in ≥ 4 of 6 cells. |
| Share of the R0–GAE gap closed by C3 | same | Descriptive. |
| Discount without relocation (D2 = R0 × P(j \| x_j); stand-in: the same with P(j \| x_j) shuffled within image and layer, seeds 0–4) | 2026-11-08 09:19, `d9bae2d` | Branch A if D2 − R0 is "higher" in ≥ 2 of 6 cells (3 models × enrichment, ClickMe), "lower" in none, and D2 − shuffle is "higher" in each of those cells; then C3D2 − C3 is reported. Otherwise B (report only). |

## Historical reproduction check

The main runs were checked against an earlier implementation of R0 (`results/history/reproduction_check.csv`:
new R0 vs the earlier full-attention runs on the same images; registered criterion |difference of means| < 0.01 on
each metric). The earlier implementation is not part of this repository; the check is kept as a record.
