# Receiver comparison in DeiT-B: reviewer-requested exploratory replication

Fixed on 2026-11-13 before loading DeiT-B weights or computing its receiver
scores. This follow-up is separate from the historical registration and from
the original DeiT-S CLS/mask protocols. Neither is amended.

## Question and scope

Does the direction of the all-query versus CLS-only comparison persist when
model width changes from supervised DeiT-S to DeiT-B, holding images,
preprocessing, layer, intervention, scores and endpoints fixed? This tests
robustness across these two checkpoints, not independent images, earlier
layers, full-image causality or a general localisation/faithfulness trade-off.
CLS is favoured structurally by the intervention target; the useful question
is how the mask comparison behaves on those same images.

## Model, sample and execution

Use `deit_base_patch16_224.fb_in1k`, pinned revision
`b78cc5532a69df6bcad9c3a8d76653fd20b31ac6`, eval mode, no fitting.
Reuse the exact 128 names in `experiments/cls/selected_names.txt` from the
original fixed 384-image Oxford test cohort. Do not select by model
correctness, intervention response or mask scores. Copy selected names into
this new study directory. Bicubic resize short side 248, centre crop 224,
ImageNet normalisation; nearest-neighbour masks, boundary ignored, patch
foreground if at least half of valid pixels are foreground.

Run `experiments.cls.run_cls --model base` with this file as `--protocol` and
a fresh output directory. Keep all five original scores and 196 last-block
CLS-only value interventions per image. Attention is frozen, no
renormalisation, output bias retained; recompute the remaining CLS path.
Response is the absolute change in the original top-two logit margin.
Validate the no-intervention reconstruction and four images by patches
0, 97, 195 against full-model forwards; maximum absolute logit and margin
errors must each be at most 0.002. Save source scores, per-image rankings,
full-forward checks, hashes, model revision, device and runtime.

## Endpoints and interpretation

Use the existing mask analysis: mean per-image margin Spearman, patch AUROC
and same-ring enrichment, on one common cohort eligible for both foreground
and background. Report any exclusions and preserve the full intervention
sample separately. AUROC is the main mask endpoint because enrichment's
scale depends on object size; AUROC does not control for radius.

Primary comparison: CLS sum of head magnitudes minus all-query value
weighting. Secondary: CLS net magnitude minus CLS sum of head magnitudes.
Use 10,000 paired image-bootstrap draws, NumPy seed 2026, 95% percentile
intervals. Report effect sizes and intervals without the original registered
margins or a binary replication verdict. Publish all results, including
opposite directions or intervals containing zero. Compare signs and paired
effects within endpoints, not magnitudes across endpoints.

## ImageNet-S extension assessment

Independently assess whether raw images and masks are locally available for
a score-independent subset with these same operations. Saved aggregate or
per-image map-quality scores cannot substitute for the new interventions.
If the matched inputs are absent and access cannot be completed, report the
concrete data limitation and retain the pet-domain restriction. Do not infer
this extension from the original registered ImageNet-S experiments.
