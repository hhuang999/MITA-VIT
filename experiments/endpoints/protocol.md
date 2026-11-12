# Exploratory endpoint analysis protocol (v13)

Date: 2026-11-12. This protocol was written before calculating the endpoints below. It uses the unchanged v12 per-image outputs and does not alter any registered analysis or verdict.

## Question and comparisons

Does C3's spatial readout advantage persist when deletion is measured continuously rather than only by top-1 agreement? Use the same fixed, paired 2,000 ImageNet-S images, all three models (`deit`, `deit_b`, `dino`), the original nine positive deletion budgets (1, 2, 4, 8, 16, 32, 64, 98, 147 of 196 tokens), and both original deletion orders. Preselect C3 minus R0 and C3 minus `stand_in_no_high` (the C3 control). The latter contrast tests whether weighting by actual value norms adds beyond removing high-norm contributions.

For image *i*, method *m*, and budget *b*, compute:

- Probability separation: `p_target(drop_bottom) - p_target(drop_top)`; positive means the ranking better retains the original top-1 prediction when low-ranked tokens are removed and damages it when high-ranked tokens are removed.
- KL separation: `KL(drop_top) - KL(drop_bottom)`; positive has the analogous direction for the divergence of the original output distribution from the reduced output distribution. Use the stored KL direction in the original outputs.

Integrate each separation over the full budget range by the trapezoid rule, with `x=[0,1,2,4,8,16,32,64,98,147]/196` and zero separation at zero budget. The last point is 147/196, so these values are area over the evaluated range, not average separation over [0,1]. The baseline zero difference is justified because the same unmodified model output applies to both deletion orders. No fabricated zero-budget absolute `p_target` is needed. Probability area has units of probability × fraction of tokens; KL area has units of nats × fraction of tokens. Both are exploratory and have **no registered practical threshold**.

Report all per-budget paired contrasts as well, retaining the original nine budgets without selecting a favourable range. Report the sign pattern and magnitude across the complete curve rather than introducing a post hoc cut-off.

## Uncertainty, checks and ClickMe dependence

For each model and endpoint, form one paired difference per image and resample those images 1,000 times using NumPy `default_rng(0)` with replacement; report the mean and 95% percentile interval. This matches the v12 bootstrap convention but does not convert the new endpoints into registered outcomes.

For ClickMe, recompute the original image-specific score as pixel Spearman against its own human map minus pixel Spearman against the previous human map in the 64-image inference batch. For C3 minus R0, C3 minus `stand_in_no_high`, and `stand_in_no_high` minus R0, report the original image-level paired bootstrap and a cluster bootstrap drawing entire original 64-image batches. All 17,408 records are 272 complete batches, indexed in their original DataLoader order; `index // 64` identifies a batch. Clustering reflects dependence from the batch-local shuffled map. It does not account for uncertainty in a different shuffle permutation. Preserve all 1,000 resamples, seed 0, and the historical margin and verdict separately for verification; the cluster interval is a sensitivity analysis, not a replacement registration rule.

Before interpretation, verify complete image/method/condition/budget grids, no duplicate keys, finite values, identical paired image sets, and exact reconstruction (up to floating-point precision) of the original top-1 deletion and ClickMe paired contrasts from the saved unrounded data and original statistic implementation.

## Input and output

Inputs are `outputs/table1/imagenet_s/{deit,deit_b,dino}/faithfulness.csv.gz`, `outputs/table1/clickme/{deit,deit_b,dino}/alignment.csv.gz`, `results/correct_tests.csv`, and `configs/table1.yaml`. Run `python research/v13/endpoints/analyze.py` from the repository root. Outputs stay in this directory. No original data, registration record, or table is rewritten.
