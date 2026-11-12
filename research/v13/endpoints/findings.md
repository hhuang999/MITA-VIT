# Continuous deletion endpoints and ClickMe batch bootstrap

Exploratory analysis of unchanged v12 outputs, 2026-11-12. See [protocol.md](protocol.md) and `analyze.py` for fixed design and regeneration. The 1,000-resample intervals describe variation over the original paired image set. The new probability and KL endpoints have no registered practical margin or prediction. All units and directions are in the protocol; positive values favour the first method.

## Main result

| Contrast | Model | Probability separation area, mean [95% CI] | KL separation area, mean [95% CI] |
| --- | --- | ---: | ---: |
| C3 − R0 | DeiT-S | 0.01357 [0.01193, 0.01518] | 0.05803 [0.04971, 0.06563] |
| C3 − no-high control | DeiT-S | 0.01040 [0.00880, 0.01199] | 0.04095 [0.03272, 0.04870] |
| no-high control − R0 | DeiT-S | 0.00317 [0.00237, 0.00395] | 0.01708 [0.01345, 0.02065] |
| C3 − R0 | DeiT-B | 0.02163 [0.01951, 0.02386] | 0.10700 [0.09625, 0.11819] |
| C3 − no-high control | DeiT-B | 0.01030 [0.00840, 0.01229] | 0.04794 [0.03826, 0.05781] |
| no-high control − R0 | DeiT-B | 0.01133 [0.00986, 0.01297] | 0.05906 [0.05085, 0.06654] |
| C3 − R0 | DINO | 0.00040 [−0.00065, 0.00151] | −0.00277 [−0.00864, 0.00339] |
| C3 − no-high control | DINO | 0.00040 [−0.00065, 0.00151] | −0.00277 [−0.00864, 0.00339] |

Across the fixed 0–147/196 budget range, the two supervised models show a C3 gain over both received attention and the no-high control on the continuous output measures. The latter control sets the layer's high-norm patches to zero in R0; it does not apply C3's output-projected value length. Thus the continuous deletion advantage is not explained solely by removing those patches. The no-high control itself also improves the continuous area relative to R0, especially in DeiT-B, so both parts contribute to the observed C3–R0 difference. DINO does not show an integrated gain on either continuous endpoint. These are ranking and intervention-on-input measurements, not an identification of why value weighting helps or a claim about causal information flow through the model.

The original registered top-1 agreement area is unchanged: C3−R0 is 0.01731 [0.01298, 0.02133] in DeiT-S, 0.02874 [0.02402, 0.03378] in DeiT-B, and 0.00028 [−0.00321, 0.00395] in DINO. The 0.05 historical practical margin still governs its original verdicts; none of the continuous endpoint intervals changes those verdicts.

## Dependence on deletion budget

`budget_differences.csv` gives all nine original budgets, each model, all three metrics, all three contrasts, and paired image-bootstrap intervals. For the central C3−no-high control contrast, the mean probability separation is positive in 8/9 DeiT-S and 9/9 DeiT-B budgets; the individual 95% intervals are above zero in 7/9 and 6/9, respectively. Mean KL separation is positive in 7/9 DeiT-S and 6/9 DeiT-B budgets, with intervals above zero in 7/9 and 5/9. The small-budget KL contrasts can be negative: at 1 and 2 tokens, DeiT-S means are −0.00280 and −0.00319 nats, while at 2 tokens DeiT-B is −0.00355 nats. The full-range positive area is therefore a statement about the evaluated curve as a whole, not every perturbation size. Top-1 agreement is coarser and less regular: C3−control means are positive at 7/9 DeiT-S and 8/9 DeiT-B budgets. These nine single-budget intervals are descriptive and are not multiplicity-adjusted hypothesis tests.

## ClickMe batch sensitivity

The shuffled map pairs each image to the preceding human map inside its original 64-image batch, so we repeated the paired bootstrap at the level of the 272 original batches. The intervals below are 95% percentiles; the historical verdict uses the original 0.02 practical margin and is left unchanged.

| Contrast | Model | Mean | Image CI | 64-image batch CI | Original verdict |
| --- | --- | ---: | ---: | ---: | --- |
| C3 − R0 | DeiT-S | 0.03333 | [0.03249, 0.03418] | [0.03244, 0.03423] | higher |
| C3 − no-high control | DeiT-S | 0.02203 | [0.02125, 0.02280] | [0.02117, 0.02284] | higher |
| no-high control − R0 | DeiT-S | 0.01130 | [0.01076, 0.01181] | [0.01081, 0.01182] | comparable |
| C3 − R0 | DeiT-B | 0.05757 | [0.05627, 0.05882] | [0.05626, 0.05886] | higher |
| C3 − no-high control | DeiT-B | 0.01707 | [0.01605, 0.01809] | [0.01593, 0.01810] | comparable |
| no-high control − R0 | DeiT-B | 0.04051 | [0.03912, 0.04189] | [0.03914, 0.04195] | higher |
| C3 − R0 | DINO | −0.00073 | [−0.00095, −0.00052] | [−0.00095, −0.00050] | comparable |

Batch resampling changes the intervals only slightly. In the marginal DeiT-B C3−control case, both intervals are wholly positive but the mean remains under the preregistered 0.02 practical margin; the original verdict remains comparable. This makes a precise paper statement possible: DeiT-B has a positive measured C3−control ClickMe difference, but it does not clear that practical threshold. Cluster resampling treats the stored previous-image pairing as fixed; it cannot estimate variability over hypothetical alternative pairings.

## Audit and suggested manuscript use

`verification.json` confirms that all 18 selected original SRG and ClickMe paired contrast means and percentile limits reconstruct from the unrounded saved observations, with maximum absolute error `1.01e-16` and matching sample counts. The deletion data contain exactly 2,000 paired images, 21 methods, both conditions, and all nine budgets for each model. ClickMe contains 17,408 images and 272 complete batches per model.

The most useful v13 addition is a compact plot or table of probability separation across the nine budgets, accompanied by the integrated C3−control probability and KL results. It shows that the top-1 deletion metric hid continuous output differences, while the model and control heterogeneity bounds the interpretation. Under the four-page limit, these results could replace a weaker post-hoc summary rather than adding another full table. The batch bootstrap belongs in the appendix as an uncertainty check. A new independent dataset or perturbation family would still be needed to establish that the gain generalizes beyond this deletion protocol.
