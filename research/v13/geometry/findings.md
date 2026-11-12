# ImageNet-S geometry and endpoint sensitivity: exploratory findings

Run `D:/anaconda/envs/vb/python.exe research/v13/geometry/analyze.py`, then `joint.py`, `auroc_sensitivity.py`, and `cohort_audit.py` from the repository root. All scripts read the original archived per-image outputs and write only under this directory. `protocol.md` records the frozen first-stage choices and the separately labeled follow-ups. The registered original statistics are unchanged.

## Main empirical pattern

On ImageNet-S, C3's centre-ring enrichment gain over R0 is concentrated in small foreground masks and appears larger when the object centroid is farther from the crop centre. All values below are paired means; the full bootstrap intervals and denominators are in `strata.csv` and `extreme_contrasts.csv`.

| Model | C3−R0, all | Area Q1 → Q4 | Centre-distance Q1 → Q4 | C3−zero-highnorm, all |
| --- | ---: | ---: | ---: | ---: |
| DeiT-S | +0.038 | +0.092 → +0.002 | +0.015 → +0.072 | +0.026 |
| DeiT-B | +0.065 | +0.165 → +0.008 | +0.027 → +0.122 | +0.025 |
| DINO | −0.004 | −0.003 → −0.003 | −0.003 → −0.002 | −0.004 |

The same shape appears for C3 versus head-mean and within-image average shuffled weights in DeiT-S/B; DINO stays near zero. Thus the observed enrichment pattern is not explained solely by replacing C3 with its high-norm-zeroed control or by shuffling weights. These are associations within saved maps, not interventions on object geometry.

`joint_cells.csv` and `joint_contrasts.csv` show the area-by-distance follow-up. Within the smallest-mask quartile, the C3−R0 difference rises from +0.062 to +0.103 in DeiT-S (Q4−Q1 +0.041, 95% CI [0.028, 0.053]) and +0.129 to +0.175 in DeiT-B (+0.047 [0.028, 0.066]) across distance quartiles. The trend is weaker in larger masks. The large-mask/farthest-distance corner has only 11 DeiT images (8 DINO), so its cell does not warrant interpretation. Object area and displacement are strongly linked in this selected/cropped dataset.

## Endpoint sensitivity changes the interpretation

Patch AUROC is calculated from the same saved images and maps but ranks foreground against background **without selecting the known foreground-count top-k**. Its DeiT C3−R0 gains persist overall: +0.0278 [0.0274, 0.0283] in DeiT-S and +0.0457 [0.0452, 0.0462] in DeiT-B, with C3−zero-highnorm +0.0204 and +0.0227 respectively. Yet the strong small-mask/off-centre enrichment gradient disappears and the distance trend modestly reverses: C3−R0 patch AUROC Q1→Q4 distance is +0.0297→+0.0248 in DeiT-S (Q4−Q1 −0.0049 [−0.0061, −0.0038]) and +0.0483→+0.0417 in DeiT-B (−0.0066 [−0.0080, −0.0050]). Area Q1→Q4 is +0.0241→+0.0264 and +0.0419→+0.0457. Corresponding controls show the same lack of the enrichment gradient. Full results are in `auroc_strata.csv` and `auroc_extreme_contrasts.csv`.

The defensible conclusion is that C3 improves **foreground/background patch ranking** in these supervised checkpoints, while the magnitude of **known-size top-k centre-ring enrichment** is heavily conditioned on object geometry. The specific enrichment gradient should not be described as an inherent advantage of norm weighting for small or eccentric objects. The metric mathematically gives zero enrichment when the foreground occupies every valid patch: selecting the known foreground count selects all valid patches. Near-full masks also have little headroom. AUROC's reversed distance trend emphasizes the need to name the endpoint. Neither endpoint alone establishes model-causal importance or explains why C3 helps.

DINO patch AUROC is slightly negative overall (C3−R0 −0.0050 [−0.0052, −0.0048]) and varies with mask geometry more than its near-zero enrichment. This is a descriptive, post-registration endpoint difference, **not** a reclassification of the registered DINO verdict. DINO uses a different model-specific crop; even the same image IDs have different mask geometry from DeiT. Compare within-model quartiles only.

## Historical denominator audit

`cohort_audit.csv` verifies all six historical paired means exactly against `results/correct_tests.csv`. The nominal finite enrichment cohort contains 241 all-foreground DeiT images and 265 all-foreground DINO images. These yield enrichment 0 but undefined AUROC. The 269 DeiT / 275 DINO area-zero images yield undefined enrichment and AUROC. Consequently, the finite historical enrichment n is 12,150 for DeiT-S/B and 12,144 for DINO; the foreground/background pair cohort is 11,909 and 11,879. If an appendix sentence says every 12,150 / 12,144 image has both classes, it is incorrect. A corrected sentence can state: “The enrichment mean uses 12,150 DeiT or 12,144 DINO images with at least one foreground patch; among them, 241 or 265 all-foreground crops contribute zero enrichment. Patch AUROC uses the 11,909 or 11,879 images with both classes.”

Excluding all-foreground cases from the old enrichment results would make C3−R0 +0.037566→+0.038327 (DeiT-S), +0.065389→+0.066712 (DeiT-B), and −0.003507→−0.003585 (DINO). These are **audit-only sensitivity estimates**; the historical results and registration remain as archived. The original method has a seeded random tie-break for equal map scores. Main versus supplementary exports agree exactly except two DeiT-S R0 image rows (indices 943 and 1136; maximum difference 0.006757); all new control comparisons use the same supplementary export, so that tiny mismatch does not mix runs.

## Suggested v13 integration

In four-page main text, one concise sentence after the primary table could state that a post hoc patch-AUROC check confirms DeiT spatial ranking gains (+0.028/+0.046 versus R0) while the stronger small-mask enrichment gradient is endpoint-sensitive. Place the quartile and denominator details in Additional Results (one compact table or figure plus a short caveat). This adds a useful boundary on how localisation improvement should be interpreted, without presenting object geometry as a causal mechanism or expanding the original registered claim. If the main v13 work adds a stronger new controlled experiment, this geometry analysis can remain in the appendix.
