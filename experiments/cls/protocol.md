# Last-block CLS value-path experiment (exploratory protocol)

Written before computing model scores, 2026-11-13. This is a new exploratory v13 experiment; it does not alter the v12 registration, outputs, or verdicts.

## Research question

Do all-query fast-weight spatial readouts rank source patches by their *direct effect on the final classifier*, or does a CLS-specific reading do better? The test isolates the final block's value path to the CLS query, so it can bound the interpretation of the existing R0 and C3 maps without changing their definition.

## Fixed model and images

Use the same frozen supervised DeiT-S checkpoint and exact evaluation preprocessing as the spatial Oxford-IIIT Pet test experiment (`research/v13/spatial/run_spatial.py`), eval mode, no training or checkpoint changes. Start from its fixed 384 selected Oxford test names; sort them by SHA-256 digest of `v13-cls-2026:` plus the filename and take the first 128 unique names. This hash is independent of the spatial-study selection hash. No class-correctness, mask-quality, or score-based filtering. Save the selected names and source/setting hashes. The Pet trimaps are not used for the classification-path endpoint.

## Intervention and direct-effect endpoint

Only the final (twelfth, one-based) transformer block is changed. For each image, keep the original block input, Q/K/V, and attention softmax. For each of the 196 patch source positions `j`, set that patch's value contribution to **zero for the CLS query only**, in every head: `u_cls(-j) = u_cls - concat_h[a_{h,CLS,j} v_{h,j}]`. Do not remove a source token or renormalize attention; other queries remain untouched. Run the original output projection with its bias and dropout in eval mode on all 196 modified CLS attention vectors. Then recompute the CLS residual, norm2, MLP, second residual, final norm, pooling and classifier head using the unchanged model modules. The last block has token-wise MLP and residual operations, so these 196 variants can be evaluated vectorially without recomputing other token states. Check the zero-intervention reconstruction against the ordinary model's original logits and compare a fixed 4-image × 3-patch subset (`j=0,97,195`) to independent full-model forwards with a final-block attention value hook. Require maximum absolute logit and fixed-class margin discrepancies at most `2e-3`; record the actual discrepancies and latency.

For each original image, fix the top-1 class `t` and runner-up class `r` from its unmodified output logits. Let `m = logit[t] - logit[r]`. For source patch `j`, the outcome is `abs((logit_{-j}[t] - logit_{-j}[r]) - m)`: the absolute change in the fixed original logit margin. This is a local, classifier-path sensitivity to last-block CLS values, not a full-image perturbation effect or a Shapley/causal attribution across layers. Save all 196 outcomes per image.

## Five prespecified ranking scores

At layer 12 only, for each patch `j`, compare five maps with the direct-effect outcome, using per-image Spearman rank correlation across all 196 patches:

1. **All-query R0:** `sum_{h,i} a_{h,i,j}` over all 197 queries.
2. **All-query C3:** `sum_{h,i} a_{h,i,j} ||W_O^h v_{h,j}||`.
3. **CLS attention:** `sum_h a_{h,CLS,j}`.
4. **CLS summed-head magnitude:** `sum_h ||a_{h,CLS,j} W_O^h v_{h,j}||`.
5. **CLS net-term magnitude:** `||sum_h a_{h,CLS,j} W_O^h v_{h,j}||`, retaining cross-head cancellation.

The outcome is an absolute margin change, so no score signs are fitted to it. Do not select layers, budgets, classes, or samples using the observed correlations. Record per-image Spearman coefficients and their paired contrasts, especially CLS net-term versus all-query C3 and CLS summed-head magnitude versus all-query C3. Save per-source scores to permit audit.

## Statistics and interpretation

For every score, report mean per-image Spearman, 95% percentile CI from 10,000 image bootstrap draws, seed 2026; use shared resample indices for paired method contrasts. Report the exact sample count, tied/undefined rank handling, selection and verification metadata, compute device, and elapsed time. Correlation with this endpoint cannot establish how earlier layers contribute or whether a score maps annotated objects; disagreement between object localization and direct CLS sensitivity is informative. Use the result to narrow or redirect the v13 paper claim, not to revise the old registered tests.
