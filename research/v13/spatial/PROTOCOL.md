# Exploratory frozen ViT spatial readout study (specified before scoring)

## Question

Does output projected value length help a received attention map identify the
image's annotated object, and which part of its spatial arrangement matters?
This probes a model reading, not causal image evidence or classification
faithfulness. The experiment is post hoc relative to the v12 registration.

## Model and independent data

- Use the exact supervised DeiT-S checkpoint pinned in
  `src/fastweight_vit/models/deit.py`:
  `timm/deit_small_patch16_224.fb_in1k` at
  `91327a9c99f98fe6b524cd4d397b7226b80e1365`; eval mode, no training.
- Use Oxford-IIIT Pet **test** images with their pixel trimaps from the Oxford
  VGG release. This is an independent, class-narrow external domain check,
  not an ImageNet-S replication. If downloaded successfully, deterministically
  select 384 test file names by SHA-256 ordering of `v13-spatial-2026:` plus
  the filename. Pair each image with its own trimap. Do not change selection
  based on results.
- Apply DeiT-S evaluation resize, center crop and normalization to images;
  align trimaps with nearest-neighbor resize and the same center crop. Use the
  repository's `token_foreground`: label 1 object, label 2 background, label 3
  ignored border, object if at least half valid pixels in a 16x16 token are
  object. Exclude images with no valid object or background patches, record
  exclusions. No class targets or model correctness filter.

## Scores and controls

Capture all-query attention `a[h,i,j]`, values `v[h,j]`, and output projection
head slices `W_O[h]` in frozen blocks 9–12 (1-based). Only patch destinations
`j=1..196` are scored; all 197 query positions contribute. Sum each score
across four layers, then rank patches per image. For each block define
`A[h,j]=sum_i a[h,i,j]`, `L[h,j]=||W_O[h]v[h,j]||`, `V[h,j]=||v[h,j]||`.

1. `R0=sum_h A[h,j]` (attention received).
2. `C3=sum_h A[h,j] L[h,j]` (the existing readout).
3. `C3v=sum_h A[h,j] V[h,j]` (unprojected value length).
4. `norm_only=sum_h mean_j(A[h,j]) L[h,j]` (head mass retained, source-specific
   received attention removed).
5. `donor_rank`: for every layer and head, replace each recipient `L` with the
   recipient's *same sorted values* arranged at positions according to a
   different image's length ranks. Donor pairs are a fixed permutation with
   no self pair. This preserves each recipient's marginal norm distribution
   and total scale while disrupting recipient-specific spatial alignment.
6. `ring_shuffle`: permute the recipient `L` among patches at the same squared
   distance from the 14x14 grid center, using a fixed seed. The same position
   permutation is used for all heads within a layer, preserving joint head
   norm vectors and radial profile; it disrupts within-ring alignment.
7. `net_output=sum_i ||sum_h a[h,i,j] W_O[h]v[h,j]||`, aggregated across layers.
   This retains cross-head cancellation for each query/destination, but is
   still a local output contribution magnitude, not causal attribution.

Checks: ensure C3 from capture equals the existing `received * length` formula
within floating-point tolerance; `net_output` cannot exceed the sum of
per-head magnitudes at the same query/destination (triangle inequality).

## Analysis

Use exactly the same images, masks, and layer set for every score. Primary
metric is per-image same-distance-ring object enrichment from the repository:
top-K where K equals the valid object patch count, minus the object's rate
within selected patches' rings. Report paired mean differences versus R0 and
C3 with image-level bootstrap 95% intervals (seed 2026, 10,000 draws).
Secondary: per-image object/background patch AUROC, precision at object area,
and object mass; report as context rather than selection criteria. Also report
object area distribution and sample count. If Oxford cannot be downloaded,
use a clearly labelled smaller ClickMe gallery fallback; do not claim the
384-image design was executed.

Interpretation was fixed without seeing scores: a C3 gain coupled to loss
under donor/ring shuffles supports the role of image-specific spatial norm
alignment. Similar gain after shuffles suggests a generic norm or radial
prior. `norm_only` close to C3 suggests the attention factor adds little.
`net_output` differs from C3 when cancellation matters; better object
enrichment would support it as a spatial readout, but never prove causal
faithfulness. A negative/heterogeneous result narrows the v13 claim.
