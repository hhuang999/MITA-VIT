# Evidence dossier

**Paper:** *Attention as Fast Weights: Reading, Diagnosing, and Correcting Attention Maps in Vision Transformers* (HKUST(GZ) DSAA 2012 course project).

**Purpose.** This document lists, for an external reviewer, what was run, on which data, with which definitions, what was fixed before the runs, and the results as stored in the result files. It contains no interpretation.

**Conventions.**

- Every number is read from a result file (CSV or JSON) in the repository. The file is named under each table ("Source"). Paths are relative to the repository root.
- An interval is written `mean [lower, upper]`: the mean over images and the 95% percentile interval of a per-image paired bootstrap (1,000 resamples, seed 0), unless the table says otherwise.
- Agreement rates, accuracies and shares are written as proportions (0.7152 = 71.52%).
- "Higher / comparable / lower" is the verdict rule of Section 2.9, applied by the analysis script and stored in the CSV files.
- Model names: DeiT-S, DeiT-B, DINO (= DINO ViT-S/16 with its official linear head), MiTA-DeiT-S.
- Run-folder names and other internal labels appear only in file paths; Section 8 maps them to the names used here.

## 0. Overview

Locations refer to the final version of the paper (`paper/review/main_review.pdf`; the main text follows `paper_review_text.md` with the corrections from the number check), by its section numbers. "§" = section of the main text, "App." = appendix A–E of that PDF. "Pre-registered" means that the definition, the decision rule and (where stated) the predicted outcome were committed to the decision log before the formal runs (Section 3). "Exploratory, post hoc" means proposed after the formal results were seen; definitions were committed before these runs, and they do not change any registered verdict.

| # | Experiment | Question | Registration status | Location in the paper |
| --- | --- | --- | --- | --- |
| 1 | Execution test and k sweep | Are the units read from attention used by the model? Agreement with the unmodified model when every query reads only its region's top-k units in all 12 layers | Pre-registered (prediction P1; replication for DeiT-S) | §3.2, §4.2, Fig. 2(a); App. C |
| 2 | Single-layer restriction and single-layer removal of attention | Which single layers can be restricted to the reading sets, or lose attention, without changing the prediction | Earlier runs, reported per model as registered (descriptive) | §4.2; App. B |
| 3 | Head vs body patches | Does removing 4 head patches hurt the prediction more than removing 4 body patches? | Pre-registered in the earlier study (PartImageNet prediction 1) | App. B |
| 4 | Top-k map vs full map | Per-image Spearman between the top-k reading-set map and the full attention map (R0) | Pre-registered in the earlier study (median > 0.5) | App. B |
| 5 | M1 (image-specific reading) | Share of within-image spatial variance of R0 explained by the model's mean map, per layer | Pre-registered as descriptive | §3.3, §4.3, Fig. 3(a) |
| 6 | M2 (unit j stands for patch j) | Position-probe accuracy per layer | Pre-registered (P2; replication) | §3.3, §4.3, Fig. 3(b) |
| 7 | M3 (activation = contribution) and value-norm ratio | Share of R0 on high-norm tokens per layer; value-vector norm of high-norm vs ordinary patches | P2 (replication) for the share; ratio from the earlier attention-budget study | §3.3, §4.3, Fig. 3(c) |
| 8 | Per-layer quality of R0 | ClickMe image-specific Spearman of single-layer R0 maps | Earlier run, descriptive | §4.3, Fig. 3(d) |
| 9 | Table 1: readings, stand-ins, references | Three metrics for R0, C1–C3, R_all, stand-ins and references, layers 9–12 | Pre-registered (P3, P7) | §4.2, §4.4, Table 1; App. D |
| 10 | Corrections vs R0 and vs stand-ins | Paired differences with verdicts | Pre-registered: R_all − R0 (P3) and correction − stand-in (P7) are primary; correction − R0 and stand-in − R0 are descriptive | §4.4; App. D |
| 11 | Change the model | +register (DeiT-S) and top-k training (MiTA-DeiT-S) on the Table 1 protocol | Pre-registered (P6) | §4.4, Table 1 |
| 12 | Combination orders | R_all (C3→C2→C1) vs the other order (C3→C1→C2) | Pre-registered as a descriptive order check | §4.4; App. D |
| 13 | Last-layer recovery | Share of the R0 drop from the best of layers 8–11 to layer 12 that each correction recovers | Pre-registered as descriptive (P5) | Abstract, §1, §4.4; App. D |
| 14 | Sanity check | Model-randomisation test of every reading | Pre-registered threshold (median < 0.5) | §4.1; App. C |
| 15 | Negative controls | C3 on DINO; C2 at layers 1–4 | Pre-registered (P4, stop rule) | §4.1; App. C |
| 16 | Reproduction check | New R0 vs earlier full-attention runs on the same images | Pre-registered (difference < 0.01) | App. C |
| 17 | Qualitative gallery | Two ClickMe images: human map, R0, C3, C2 | Image choice by a rule fixed in the earlier study (seeded draw, smallest indices) | §4.2, Fig. 2(b); App. B (layer 12) |
| 18 | Probe error geometry, mean prediction, where C2 moves mass | Why does C2 lower the metrics? | Exploratory, post hoc; definitions and descriptive criteria written before running | §4.5; App. E |
| 19 | ClickMe and smoothing | Compilation of existing differences for blur, C2, the other order | Exploratory, post hoc (no new runs) | §5 (practice point ii); App. E |
| 20 | Pairwise combinations | C1+C3, C2+C3, C1+C2 | Exploratory, post hoc; branch rule written before running | §4.4; App. D |
| 21 | C3 mechanism checks | C3 with ‖v‖, with shuffled weights, with head-mean weights | Exploratory, post hoc; rule written before running | §4.4; App. D |
| 22 | C3 dose–response and per-layer | Across images, does the C3 gain grow with R0's high-norm share? | Exploratory, post hoc; branch rule written before running | §4.4; App. D |
| 23 | Within-layer globalisation | Are tokens closer to the image mean token harder to locate? | Exploratory, post hoc; rule written before running | §4.5; App. E |
| 24 | MLP probe and C2 variants | Does a two-layer probe, or moving only confident tokens, change C2? | Exploratory, post hoc; branch rule written before running | Abstract, §4.5; App. E |
| 25 | Share of the GAE gap closed by C3 | (C3 − R0) ÷ (GAE − R0) | Exploratory, post hoc (descriptive) | §4.4; App. D |
| 26 | Discount without relocation | R0 × probability that token j is still at patch j | Exploratory, post hoc; definition and branch rule registered before running | §4.5; App. E |

## 1. Models, data, preprocessing and compute

### 1.1 Models

| Paper name | Weights | Classifier | Notes |
| --- | --- | --- | --- |
| DeiT-S | timm `deit_small_patch16_224.fb_in1k`, revision `91327a9c99f98fe6b524cd4d397b7226b80e1365` | timm classifier head | Supervised ImageNet-1k |
| DeiT-B | timm `deit_base_patch16_224.fb_in1k`, revision `b78cc5532a69df6bcad9c3a8d76653fd20b31ac6` | timm classifier head | Supervised ImageNet-1k |
| DINO | timm `vit_small_patch16_224.dino`, revision `10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4` | Official DINO linear head `dino_deitsmall16_linearweights.pth` (SHA-256 `23250c67b1ae117ef8149c8afd9c212390bdb6f00e33f92be8c17cc306dc6c1b`): CLS token of the last 4 blocks, each through the final norm, concatenated (1,536-d), one linear layer (facebookresearch/dino `eval_linear.py`) | Self-supervised backbone, frozen |
| +register | DeiT-S as above | timm head | Test-time register (Jiang et al. 2025): the activations of 8 FFN hidden units, (layer, unit) = (5, 684), (4, 1361), (4, 375), (5, 1274), (5, 36), (5, 333), (5, 83), (5, 1500), are shifted to one added register token inserted at `norm_pre` (after CLS, no positional embedding). Units found on 500 ImageNet-S pilot images (`experiments/2026-11-03_ab_t2_deit/gate.json`) |
| MiTA-DeiT-S (top-k training) | Official MiTA checkpoint `best_checkpoint.pth` (SHA-256 `d45434671d1096c6d7bfcba38407eb6a632049adf1b1137d8ea85fa638778ac2`), loaded into the timm DeiT-S skeleton; dense re-implementation of MiTA attention (25 summary slots from a 5×5 landmark grid, k = 25) | Trained head | Trained with top-k attention |

Software and numerics in all v5 runs (from each run's `metadata.json`): PyTorch `2.7.0+cu128`, timm `1.0.30`, float32 without autocast, deterministic algorithms. Code: `src/mita_probe/models/`.

### 1.2 Data

| Data | Images | Used for | Selection |
| --- | --- | --- | --- |
| ImageNet validation, fixed subset | 5,000 | Execution test, k sweep, single-layer tests, no-attention runs | Seed-0 draw from the 50,000 validation images (`data/imagenet_val_indices_seed0.json`) |
| ImageNet-S919 validation | 12,419 | Same-ring enrichment (all images); M1 share; where C2 moves mass | All images with pixel masks (Gao et al. 2022). Enrichment is defined for 12,150 images (DeiT-S, DeiT-B) and 12,144 (DINO); the others have no valid object patch |
| ImageNet-S, fixed SRG subset | 2,000 | SRG for every map | Seed-0 random subset of the 12,419 (the subset of the earlier DeiT-B run) |
| ImageNet-S, fixed sanity subset | 1,000 | Model-randomisation sanity check | Seed-0 random subset |
| ClickMe (Harmonization release) | 17,408 | ClickMe image-specific Spearman | 17 validation TFRecord shards of the Harmonization release (Fel et al. 2022), each record a 256×256 image, a 256×256 click heatmap and the label. Source of the images: see Section 5.3 |
| Held-out ImageNet validation set | 4,000 = 2,000 probe-train + 2,000 test | Position probes (train); mean maps, probe test accuracy, blur widths (test). Never used for evaluation | Validation rows drawn in seed-0 order; ImageNet-S validation images skipped by file name (1,296 skipped); candidates whose 32×32 grey thumbnail correlates ≥ 0.9 with a same-class ClickMe image skipped (whole-image and centre-square geometries). 4,002 candidates examined, 2 ClickMe near-duplicates found (correlations 0.974 and 0.921, centre-square geometry). Identical list for the three models: yes (`heldout.json` of each probe run) |
| Held-out, MLP probe split (exploratory) | 1,800 + 200 | Two-layer MLP probe training and early stopping | The 2,000 probe-train images, last 10% (200 images) held out for early stopping |
| PartImageNet_Seg | 24,080 (DeiT-S, DINO); 3,614 (DeiT-B, val + test parts only) | Head vs body patches | Images analysed: 24,058 / 3,609 / 24,057 (DeiT-S / DeiT-B / DINO). These are ImageNet training images |
| ImageNet-S, attention-budget split | 500 pilot + 11,919 formal | Register unit search (pilot); value-norm ratio (formal) | ImageNet-S validation shuffled with seed 0; first 500 = pilot, rest = formal |

### 1.3 Preprocessing

| Protocol | Transform | Used in |
| --- | --- | --- |
| DeiT-S, DeiT-B (own) | timm default for the checkpoint: Resize 248 (bicubic), centre crop 224 (crop 0.9) | k sweep; ImageNet-S evaluation; probes; mean maps (ImageNet-S geometry); PartImageNet; per-layer M3 run |
| DINO (own) | DINO linear-evaluation transform: Resize 256 (bicubic), centre crop 224 (crop 0.875) | All DINO runs |
| MiTA-DeiT-S (own) | MiTA evaluation: Resize 256 (bicubic), centre crop 224 (crop 0.875) | Top-k training row of Table 1 |
| Uniform | Resize 256 (bicubic), centre crop 224 (crop 0.875) for every model | Attention-removal runs (all layers and single layers), register unit search, value-norm ratio (attention-budget study) |
| ClickMe geometry | Image: 256×256 resized bilinearly to 224×224, no crop. Human map: 256×256 heatmap resized to 64×64 (bilinear), blurred with a 10×10 Gaussian (σ = 10), resized to 224×224 (bilinear). Normalisation mean/std of each model | All ClickMe evaluations |
| ClickMe-geometry mean maps | Held-out images converted to the matched ClickMe storage geometry (centre square, 256×256), then resized to 224 as ClickMe | C1, R_all, the other order and the centred Gaussian subtraction on ClickMe |
| Masks | Object mask resized with the same Resize + centre crop (nearest neighbour). Patch valid if it has ≥ 1 non-ignored pixel; object patch if ≥ half of its valid pixels are the target class | Enrichment; PartImageNet |

Sources: `src/mita_probe/models/__init__.py`, `src/mita_probe/data/clickme.py`, `src/mita_probe/data/masks.py`; `preprocessing: uniform` in `experiments/2026-11-03_ab_t{2,3,7}_*/config.yaml`.

### 1.4 Image-list splits

- The held-out set excludes every ImageNet-S validation image by file name, and is not used for any evaluation metric.
- The SRG subset (2,000) and the sanity subset (1,000) are subsets of the 12,419 ImageNet-S images.
- The same 5,000 ImageNet validation images are used for the k sweep, the single-layer restriction and the attention-removal runs.
- Readings, stand-ins and references are scored on the same images within each metric (paired comparisons).
- ClickMe and the held-out set: 2 of 4,002 examined held-out candidates matched a ClickMe image (Section 5.3).

### 1.5 Compute

GPU time = sum of `elapsed_seconds` in the `checks.json` of every run in the group (whole runs; reused runs also computed quantities that the paper does not use). "First start" is the earliest `timestamp` in the runs' `metadata.json` (UTC). The failed first DeiT-B k-sweep run (Section 5.1) was not retrieved and is not counted.

| Group | Runs | GPU hours | GPU | Code commit | First start (UTC) |
| --- | --- | --- | --- | --- | --- |
| Execution test, k sweep (3 models) | 3 | 0.60 | NVIDIA L4 | 5faa503, 94bf426 | 2026-11-05 16:19 |
| Position probes and held-out set (3 models) | 3 | 0.09 | NVIDIA L4 | 94bf426 | 2026-11-05 15:53 |
| Mean maps and M1 share (3 models) | 3 | 0.33 | NVIDIA L4 | 94bf426 | 2026-11-05 15:59 |
| Table 1 readings, stand-ins, references (3 models × 2 datasets) | 6 | 3.05 | NVIDIA L4 | 5faa503 | 2026-11-05 17:34 |
| Change the model: +register (2 datasets) | 2 | 0.13 | NVIDIA L4 | 5faa503 | 2026-11-05 20:38 |
| Change the model: top-k training (2 datasets) | 2 | 0.13 | NVIDIA L4 | 5faa503 | 2026-11-05 20:44 |
| Combination orders (3 models × 2 datasets) | 6 | 0.56 | NVIDIA L4 | 5faa503 | 2026-11-05 20:54 |
| Sanity check (3 models) | 3 | 0.39 | NVIDIA L4 | 5faa503 | 2026-11-05 21:28 |
| Exploratory: probe error geometry, mean prediction, where C2 moves mass (3 models) | 3 | 0.31 | NVIDIA L4 | ebfd25f | 2026-11-07 10:17 |
| Exploratory: far-share rerun after the float32 fix (3 models) | 3 | 0.01 | NVIDIA L4 | 534e333 | 2026-11-07 13:48 |
| Exploratory: held-out pass (MLP probes, globalisation, probe tables; 3 models) | 3 | 0.19 | NVIDIA L4 | 534e333 | 2026-11-07 11:08 |
| Exploratory: evaluation pass (pairwise combinations, C3 checks, C2 variants; 3 models × 2 datasets) | 6 | 2.46 | NVIDIA L4 | 534e333 | 2026-11-07 11:20 |
| Reused: single-layer restriction (whole earlier runs, 3 models) | 3 | 1.60 | NVIDIA A100-SXM4-80GB, NVIDIA L4 | 08367cc, 28eba6f, 5699c62 | 2026-10-28 03:56 |
| Reused: attention removed, all layers and single layers (3 models) | 3 | 0.16 | NVIDIA L4 | efe6287 | 2026-11-03 11:33 |
| Reused: head vs body patches, PartImageNet (3 models) | 3 | 1.37 | NVIDIA A100-SXM4-80GB | 08367cc, 2f5aac9 | 2026-10-29 12:40 |
| Reused: top-k map vs full map, ImageNet-S (3 models) | 3 | 1.08 | NVIDIA L4 | 0269534 | 2026-10-31 11:27 |
| Reused: per-layer high-norm share (M3), ImageNet-S (3 models) | 3 | 2.08 | NVIDIA L4 | 0269534 | 2026-10-31 12:32 |
| Reused: per-layer ClickMe quality of R0 and reproduction reference (3 models) | 3 | 2.58 | NVIDIA L4 | 490f05a | 2026-10-31 07:24 |
| Reused: register unit search (DeiT-S, DeiT-B, DINO) | 3 | 0.07 | NVIDIA L4 | efe6287 | 2026-11-03 11:29 |
| Reused: value-norm ratio (budget runs, DeiT-S, DeiT-B) | 2 | 0.90 | NVIDIA L4 | 4671c79 | 2026-11-03 12:21 |
| Reused: centred Gaussian vs ClickMe, Spearman 0.41 (one run) | 1 | 0.28 | NVIDIA A100-SXM4-80GB | 6304f52 | 2026-10-29 11:56 |
| Exploratory: discount without relocation (3 models × 2 datasets) | 6 | 0.67 | NVIDIA L4 | 7f75fa1 | 2026-11-08 10:00 |

Total: 19.05 GPU hours, of which 8.93 in the v5 formal and exploratory runs (all on NVIDIA L4 GPUs).

## 2. Definitions

### 2.1 Notation and the reading R0

Block ℓ (1–12) receives 197 tokens: CLS and 196 patches (14×14 grid). a<sup>ℓh</sup><sub>ij</sub> is the softmax attention of query i (all 197 tokens, CLS included) to key j in head h. Maps are computed for the 196 patch keys only.

- **R0** (attention received), per layer: s<sup>ℓ</sup><sub>j</sub> = Σ<sub>h</sub> Σ<sub>i</sub> a<sup>ℓh</sup><sub>ij</sub>. A multi-layer reading is the sum of its per-layer maps. Unless stated otherwise, layers 9–12.
- All three evaluation metrics depend only on the ranking of patches within an image, so a constant offset or a positive factor does not change a result.

Code: `src/mita_probe/readings.py`.

### 2.2 Corrections

| Name | Definition per layer ℓ (then summed over layers) | Premise |
| --- | --- | --- |
| C3 (contribution weighting) | c<sub>j</sub> = Σ<sub>h</sub> Σ<sub>i</sub> a<sub>hij</sub> · ‖W<sub>O</sub><sup>h</sup> v<sub>hj</sub>‖<sub>2</sub>; W<sub>O</sub><sup>h</sup> = the columns of the output projection for head h, bias excluded | M3 |
| C2 (relocation by probe) | r<sub>p</sub> = Σ<sub>j</sub> s<sub>j</sub> · P<sub>ℓ</sub>(p \| x<sub>j</sub>); P<sub>ℓ</sub> = 196-way softmax of the layer-ℓ linear position probe applied to x<sub>j</sub>, the block-ℓ token after norm1 (the input that produces keys and values) | M2 |
| C1 (subtract mean map) | s<sub>j</sub> − s̄<sub>j</sub>; s̄ = mean of R0 at layer ℓ over the 2,000 held-out test images, in the geometry of the evaluation set | M1 |
| R_all (combined, C3→C2→C1) | relocate c with P<sub>ℓ</sub>, then subtract the held-out mean of relocated c | All three |
| Other order (C3→C1→C2) | subtract the held-out mean of c from c, then relocate with P<sub>ℓ</sub> | All three |

Position probe: linear 196-way classifier on standardised tokens (training mean and standard deviation), Adam, learning rate 0.01, batch 4,096, 30 epochs, seed 0, cross-entropy with target = original patch index; one probe per layer and model (36 probes), trained on the 2,000 held-out probe-train images (all 196 patch tokens per image).

### 2.3 Stand-ins (one per correction)

| Stand-in | Definition | Paired with |
| --- | --- | --- |
| Matched blur | R0 redistributed with a Gaussian kernel on the 14×14 grid, each row normalised over the grid (total kept, as in C2). σ<sub>ℓ</sub> is chosen by bisection so that the kernel's mean squared displacement equals C2's mean squared displacement Σ<sub>p</sub> P<sub>ℓ</sub>(p \| x<sub>j</sub>) ‖p − j‖² on the held-out test tokens of layer ℓ | C2 |
| Zero high-norm tokens | R0 with the layer-ℓ high-norm patches set to 0. High-norm patch at layer ℓ: block-ℓ input norm (before norm1) > 3 × the median patch norm of that image and layer (midpoint median; CLS excluded); judged separately per layer | C3 |
| Centred Gaussian subtraction | R0 (layers 9–12) minus a centred isotropic Gaussian (σ = 14/4 = 3.5 patches) scaled to the spatial standard deviation of the summed layer 9–12 mean map that C1 subtracts | C1 |

Mean squared displacement of C2 on held-out test tokens / matched σ (patches), layers 9–12:

| Model | Layer 9 | Layer 10 | Layer 11 | Layer 12 |
| --- | --- | --- | --- | --- |
| DeiT-S | 5.001 / 1.685 | 8.845 / 2.294 | 12.031 / 2.723 | 13.439 / 2.900 |
| DeiT-B | 0.877 / 0.685 | 1.596 / 0.927 | 2.988 / 1.284 | 6.479 / 1.936 |
| DINO | 1.159 / 0.787 | 1.586 / 0.925 | 2.189 / 1.092 | 2.487 / 1.167 |

Source: `experiments/2026-11-05_diagnose_probe_{deit,deit_b,dino}/identity.csv`.

### 2.4 References

| Reference | Definition |
| --- | --- |
| Random | Independent Uniform(0, 1) score per patch, seeded (seed 0 combined with the batch start) |
| Centre | Centred isotropic Gaussian, σ = 3.5 patches (image-independent) |
| Mean map | The summed layer 9–12 held-out mean map of R0 (image-independent) |
| GAE | Chefer et al. 2021: relevance R ← R + mean<sub>h</sub>[(∂y/∂A ⊙ A)<sup>+</sup>] R through all layers, starting from the identity; CLS row, patch columns; y = logit of the model's own top-1 class. Uses gradients and the class; reported as a reference |

### 2.5 Change-the-model rows

- **+register:** R0 of DeiT-S with the test-time register (Section 1.1). The register token is excluded from the map like CLS. In SRG the register token is kept together with CLS when patches are removed.
- **Top-k training:** R0 of MiTA-DeiT-S; weight a query puts on a summary slot is passed back to tokens by S = softmax(landmark · key). No SRG: the summary slots pool fixed regions of the full 14×14 grid, which are undefined after patches are removed.
- Both rows are paired with DeiT-S R0 on the same image indices. MiTA-DeiT-S uses its own crop (0.875), DeiT-S uses 0.9 (Section 5.3).

### 2.6 Execution test (reading sets)

The 14×14 grid is pooled into 16 regions (4×4 adaptive average pooling; windows [0,4), [3,7), [7,11), [10,14) per axis, so neighbouring windows overlap). In every head, each region's mean query (adaptive average of its patch queries) is scored against all 197 keys (scaled dot product); its top-k keys form the region's reading set. Each patch query is routed to one region by position (row r, column c → region (r·4//14)·4 + c·4//14); CLS keeps its own top-k set. Reading sets and routing are computed in a forward pass of the unmodified model and frozen; the model is then rerun with every query allowed to attend only to its region's set, in all 12 layers at once (boolean attention mask; Q, K, V from the intervened forward). Outcome: top-1 agreement with the unmodified model. Random sets: same size, same routing, sets drawn uniformly without replacement from the 197 tokens, seeds 0–4 averaged per image before the bootstrap. No attention: the attention branch of all 12 blocks set to zero (only residual and FFN). k ∈ {4, 8, 16, 32, 64, 128}. Before execution each run checks that the explicit-softmax attention reproduces the model (same top-1 classes; maximum logit difference < 5e-4; Section 5.1).

### 2.7 Diagnostics

- **M1 share** (per layer, R0): 1 − Σ<sub>x</sub> Σ<sub>j</sub> (r − r̄)² / Σ<sub>x</sub> Σ<sub>j</sub> (s − s̄<sub>x</sub>)², where r = s − s̄ (s̄ = held-out mean map), r̄ and s̄<sub>x</sub> are each map's own spatial mean, summed over the 12,419 ImageNet-S images. Interval: bootstrap of the ratio of sums over images. It can be negative.
- **Radial share of the mean map:** share of the mean map's spatial variance explained by its ring means (rings = patches at equal distance from the grid centre).
- **M2** (per layer): top-1 accuracy of the linear position probe on all 196 patch tokens of each of the 2,000 held-out test images, averaged per image, then bootstrapped over images.
- **M3** (per layer): share of R0 that falls on that layer's high-norm patches (definition in 2.3), ImageNet-S, 12,419 images.
- **Value-norm ratio** (attention-budget study): per image and head, mean ‖v<sub>j</sub>‖ over high-norm patches ÷ median ‖v<sub>j</sub>‖ over the other patches (‖v‖, not ‖W<sub>O</sub>v‖), averaged over the images that have high-norm patches and over heads, per layer.
- **Per-layer quality:** ClickMe image-specific Spearman of the single-layer R0 map.

### 2.8 Metrics

**SRG (deletion faithfulness; Blücher et al. 2024).** Patches are ranked by the map (ties broken by a seeded random permutation, seed 0 per batch). Removing a patch = dropping its token after the positional embedding (equivalent to masking it from every layer's attention; checked by a unit test); CLS (and the register token) are always kept. At removal budgets 0, 1, 2, 4, 8, 16, 32, 64, 98, 147 patches (x = budget / 196; agreement 1 at budget 0), agreement = top-1 equals the unmodified model's top-1. SRG = area (trapezoid rule) under the least-important-first curve − area under the most-important-first curve. 2,000 ImageNet-S images.

**Same-ring enrichment.** Per image: select the top-|O| valid patches by score (|O| = number of object patches; ties broken by seeded noise of size 1e-6 × max |score|). Rings = sets of patches with equal distance from the grid centre (6.5, 6.5). For every ring that contains selected patches: (object rate among the selected patches of the ring − object rate among all valid patches of the ring). Enrichment = average of these differences weighted by the number of selected patches per ring. A map that depends only on distance from the centre scores 0 in expectation. ImageNet-S, all images with a defined value.

**ClickMe image-specific Spearman.** Spearman correlation with this image's human map minus Spearman correlation with another image's human map. Both correlations are at pixel level: the 14×14 map is upsampled bilinearly to 224×224 and correlated over the 224×224 pixels with the 224×224 human map (geometry in 1.3). The "other image" is the previous image in the batch: the human maps of a batch are rolled by one position along the batch dimension (`torch.roll(human, 1, 0)`), so image i is paired with image i − 1 and the first image of a batch with the last image of the same batch. Batch size 64; 17,408 = 272 × 64, so every batch is full. Any map that does not depend on the image scores exactly 0 per image. Code: `src/mita_probe/experiments/human_alignment.py` (`alignment`), `scripts/analyze_v5.py` (`per_image`).

### 2.9 Statistics and verdict rule

- One value per image (random-seed controls averaged over seeds 0–4 first). Paired comparisons use the images where both values exist.
- Bootstrap: 1,000 resamples of images with replacement, `numpy.random.default_rng(0)`; 95% percentile interval of the resampled means. Ratios (M1, last-layer recovery, share of the GAE gap) resample the numerator and denominator together and take the ratio of means.
- No multiple-comparison correction.
- **Verdict rule** (`scripts/analyze_v5.py`, function `verdict`; decision log, author confirmation of 2026-11-05):

  ```
  comparable  if  lower <= 0 <= upper   or   |mean| < margin
  higher      otherwise, if lower > 0
  lower       otherwise (upper < 0)
  ```

  Margins: SRG 0.05; same-ring enrichment 0.02; ClickMe image-specific Spearman 0.02.
- Other rules: execution-test predictions use "interval lower bound > 0" without a margin; the head-vs-body comparison of the earlier study uses "higher if lower > 0, lower if upper < 0, otherwise uncertain"; the attention-budget verdicts use Bonferroni 99.5% intervals with 10,000 resamples and a margin of 0.03.

### 2.10 Properties stated before the runs (not findings)

Recorded in `docs/experiment_plan_v5.md` §2.4: (1) all three metrics are rank-based per image; if all ‖W<sub>O</sub>v‖ are equal, C3 equals R0 (unit test). (2) C2 with an identity probe returns R0 and keeps the total (unit test); early-layer probes are close to identity, so the layer 1–4 control is an implementation check. (3) Image-independent references score about 0 on enrichment and ClickMe by construction. (4) C1 can change enrichment only to the extent that the mean map is not radially symmetric; the centred Gaussian subtraction is exactly radial. (5) C1 averages to 0 on the held-out images (unit test), not on the evaluation sets. (6) C2 changes SRG only through the probe's notion of position.

## 3. Pre-registration

The registration log is `docs/decisions.md` (Chinese); the execution specification is `docs/experiment_plan_v5.md`. Each entry was committed to git before the runs it governs. Commit times and run start times are UTC (`git log`; `timestamp` in each run's `metadata.json`).

### 3.1 What was fixed when

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
| 2026-11-08 09:19 | `d9bae2d` | Discount without relocation: definition, stand-in, branch rule | Runs 2026-11-08 10:00–10:58 (code `7f75fa1`) |

Code commits of the formal runs (`git_commit` in `metadata.json`): `94bf426` for the 3 probe runs, 3 mean-map runs and the DeiT-S and DINO k sweeps; `5faa503` for all other formal runs. The two commits differ only in the implementation-check tolerance.

### 3.2 Results seen before registration (disclosed in the plan)

- The DeiT-S k sweep (same setting, k = 4–128) had been seen in an earlier run; P1 is a replication for DeiT-S.
- The earlier full-attention, per-layer and high-norm-share results and the earlier position probe (400 training images) had been seen; the M2 and M3 parts of P2 are replications.
- The attention-budget results (including the value-norm ratios on the pilot set, before its threshold was registered) had been seen.
- Smoke runs of 32–64 images per run were inspected for code checks.
- All exploratory analyses (Section 4.18) were proposed after the formal results.

### 3.3 Predictions P1–P7 (registered text, translated) and results

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
| P1 | DeiT-S | holds | 11/11 tests (`p1_tests.csv`, Section 4.1) |
| P1 | DeiT-B | holds | 11/11 tests (`p1_tests.csv`, Section 4.1) |
| P1 | DINO | holds | 11/11 tests (`p1_tests.csv`, Section 4.1) |
| P2 (M2) | DeiT-S | holds | layer 12 upper 0.1830 < layer 1 lower 1.0000; Spearman(layer, accuracy) = −1.0000 |
| P2 (M2) | DeiT-B | holds | layer 12 upper 0.3203 < layer 1 lower 1.0000; Spearman(layer, accuracy) = −1.0000 |
| P2 (M2) | DINO | holds | layer 12 upper 0.3600 < layer 1 lower 0.9993; Spearman(layer, accuracy) = −0.9930 |
| P2 (M3) | DeiT-S | holds | largest share at layer 8 (0.0929) |
| P2 (M3) | DeiT-B | holds | largest share at layer 11 (0.4354) |
| P2 (M3) | DINO | holds | largest upper bound over layers = 0.0003 |
| P3 | DeiT-S, DeiT-B | partly holds | 1/4 cells higher; case: partial; pre-agreed reading (2) (Section 4.10) |
| P4a | DINO | holds | 3/3 comparable (Section 4.15) |
| P4b | DeiT-S,DeiT-B,DINO | holds | 6/6 comparable (Section 4.15) |
| P5 | DeiT-S, DeiT-B | descriptive | Section 4.13 |
| P6 | +register, top-k training | holds | 4/4 not higher (Section 4.11) |
| P7 | DeiT-S | no direction predicted | C1 vs control: enrichment comparable, SRG comparable, ClickMe comparable |
| P7 | DeiT-S | no direction predicted | C2 vs control: enrichment lower, SRG lower, ClickMe lower |
| P7 | DeiT-S | no direction predicted | C3 vs control: enrichment higher, SRG comparable, ClickMe higher |
| P7 | DeiT-B | no direction predicted | C1 vs control: enrichment higher, SRG higher, ClickMe higher |
| P7 | DeiT-B | no direction predicted | C2 vs control: enrichment higher, SRG comparable, ClickMe comparable |
| P7 | DeiT-B | no direction predicted | C3 vs control: enrichment higher, SRG comparable, ClickMe comparable |
| P7 | DINO | no direction predicted | C1 vs control: enrichment comparable, SRG comparable, ClickMe comparable |
| P7 | DINO | no direction predicted | C2 vs control: enrichment lower, SRG lower, ClickMe lower |
| P7 | DINO | no direction predicted | C3 vs control: enrichment comparable, SRG comparable, ClickMe comparable |

Source: `experiments/2026-11-05_v5_analysis/predictions.csv`; `experiments/2026-11-05_v5_analysis/fig3_diagnose.csv`.

P2 detail values are recomputed from `fig3_diagnose.csv` with the code of `scripts/analyze_v5.py` (function `p2`). P6 part (1): the earlier verdicts are in `experiments/2026-11-03_ab_analysis_formal/judgements.csv` (register on DeiT-S: 0.0057 [0.0047, 0.0068], "holds"; top-k training, Small: −0.0107 [−0.0131, −0.0084]; Tiny: 0.0374 [0.0347, 0.0402]; the hypothesis needed both pairs to hold and was recorded as "does not hold"; these intervals are Bonferroni 99.5% intervals with 10,000 resamples, on the 11,919-image formal split with uniform preprocessing).

## 4. Experiments

### 4.1 Execution test and k sweep

**Question.** When every query may attend only to its region's k reading-set units, in all 12 layers at once, how often does the top-1 prediction match the unmodified model?

**Setup.** DeiT-S, DeiT-B, DINO; fixed 5,000 ImageNet validation images; own preprocessing; 16 regions, spatial routing, CLS with its own set (Section 2.6); k ∈ {4, 8, 16, 32, 64, 128}. Controls: random sets of the same size (seeds 0–4, averaged per image); no attention in any layer (separate runs with uniform preprocessing, Section 5.3).

**Registration.** P1 (rule: lower bound > 0 for every adjacent-k difference and every reading − random difference). Replication for DeiT-S.

Top-1 agreement with the unmodified model (n = 5,000 per cell):

| Model | Sets | k = 4 | k = 8 | k = 16 | k = 32 | k = 64 | k = 128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeiT-S | reading sets | 0.2584 [0.2464, 0.2696] | 0.5198 [0.5058, 0.5340] | 0.7152 [0.7030, 0.7268] | 0.8230 [0.8126, 0.8330] | 0.8922 [0.8838, 0.9008] | 0.9608 [0.9552, 0.9658] |
| DeiT-S | random sets (seed mean) | 0.0054 [0.0044, 0.0063] | 0.0183 [0.0164, 0.0203] | 0.0919 [0.0865, 0.0974] | 0.3032 [0.2936, 0.3130] | 0.6060 [0.5956, 0.6158] | 0.8469 [0.8392, 0.8542] |
| DeiT-B | reading sets | 0.5128 [0.4986, 0.5272] | 0.7296 [0.7170, 0.7416] | 0.8276 [0.8174, 0.8384] | 0.8894 [0.8810, 0.8982] | 0.9364 [0.9294, 0.9432] | 0.9758 [0.9714, 0.9800] |
| DeiT-B | random sets (seed mean) | 0.0193 [0.0172, 0.0216] | 0.0807 [0.0754, 0.0859] | 0.2958 [0.2860, 0.3054] | 0.6201 [0.6100, 0.6294] | 0.8139 [0.8056, 0.8224] | 0.9185 [0.9132, 0.9244] |
| DINO | reading sets | 0.2508 [0.2388, 0.2620] | 0.4652 [0.4522, 0.4796] | 0.6148 [0.6010, 0.6286] | 0.7164 [0.7048, 0.7286] | 0.8012 [0.7908, 0.8122] | 0.9292 [0.9226, 0.9358] |
| DINO | random sets (seed mean) | 0.0038 [0.0030, 0.0046] | 0.0096 [0.0082, 0.0111] | 0.0424 [0.0391, 0.0460] | 0.1967 [0.1893, 0.2046] | 0.4792 [0.4690, 0.4897] | 0.7528 [0.7446, 0.7612] |

No attention in any of the 12 layers (uniform preprocessing; n = 5,000):

| Model | Top-1 agreement |
| --- | --- |
| DeiT-S | 0.0012 [0.0004, 0.0022] |
| DeiT-B | 0.0016 [0.0006, 0.0028] |
| DINO | 0.0008 [0.0002, 0.0016] |

Source: `experiments/2026-11-05_v5_analysis/fig2a_ksweep.csv`.

P1 tests (each: lower bound > 0):

| Model | Tests holding | Reading − random, range of means over k | Adjacent k difference, range of means |
| --- | --- | --- | --- |
| DeiT-S | 11/11 | 0.1139–0.6233 (smallest lower bound 0.1068) | 0.0686–0.2614 (smallest lower bound 0.0608) |
| DeiT-B | 11/11 | 0.0573–0.6489 (smallest lower bound 0.0516) | 0.0394–0.2168 (smallest lower bound 0.0334) |
| DINO | 11/11 | 0.1764–0.5724 (smallest lower bound 0.1675) | 0.0848–0.2144 (smallest lower bound 0.0752) |

Source: `experiments/2026-11-05_v5_analysis/p1_tests.csv`. Verdict: P1 holds for all three models.

Implementation check (maximum logit difference between explicit-softmax and fused attention, same classes required): DeiT-S 3.39e-05, DeiT-B 1.40e-04, DINO 2.67e-05 (`experiments/2026-11-05_read_ksweep_*/checks.json`).

### 4.2 Single-layer restriction and single-layer removal of attention

**Question.** Restricting only one layer to the reading sets (k = 16), or removing only one layer's attention branch: how much top-1 agreement is kept?

**Setup.** Same 5,000 ImageNet validation images. Restriction: earlier runs (DeiT-S and DINO on an L4, DeiT-B on an A100), own preprocessing, 16 regions, k = 16. Removal: attention-budget runs, uniform preprocessing (Section 5.3). Not rerun for v5 (Section 6).

**Registration.** Reported per model, descriptively, as decided before the formal runs.

| Layer | DeiT-S restricted | DeiT-S attention removed | DeiT-B restricted | DeiT-B attention removed | DINO restricted | DINO attention removed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9452 [0.9386, 0.9516] | 0.6876 [0.6748, 0.7010] | 0.9710 [0.9662, 0.9752] | 0.8778 [0.8688, 0.8870] | 0.7670 [0.7556, 0.7782] | 0.0016 [0.0006, 0.0028] |
| 2 | 0.9194 [0.9118, 0.9264] | 0.9278 [0.9208, 0.9352] | 0.9642 [0.9592, 0.9692] | 0.9732 [0.9688, 0.9772] | 0.9492 [0.9428, 0.9548] | 0.9502 [0.9444, 0.9564] |
| 3 | 0.9168 [0.9092, 0.9242] | 0.9370 [0.9302, 0.9440] | 0.9658 [0.9608, 0.9708] | 0.9776 [0.9732, 0.9818] | 0.9486 [0.9428, 0.9546] | 0.9384 [0.9318, 0.9452] |
| 4 | 0.9448 [0.9386, 0.9508] | 0.9424 [0.9360, 0.9490] | 0.9630 [0.9580, 0.9680] | 0.9658 [0.9606, 0.9706] | 0.9162 [0.9084, 0.9240] | 0.9470 [0.9408, 0.9532] |
| 5 | 0.9544 [0.9488, 0.9604] | 0.9452 [0.9386, 0.9516] | 0.9570 [0.9514, 0.9626] | 0.9622 [0.9568, 0.9674] | 0.9404 [0.9340, 0.9466] | 0.9306 [0.9240, 0.9378] |
| 6 | 0.9342 [0.9280, 0.9404] | 0.9432 [0.9368, 0.9494] | 0.9634 [0.9580, 0.9684] | 0.9654 [0.9606, 0.9704] | 0.8914 [0.8824, 0.8998] | 0.9380 [0.9312, 0.9442] |
| 7 | 0.9430 [0.9360, 0.9492] | 0.9334 [0.9266, 0.9408] | 0.9572 [0.9512, 0.9626] | 0.9612 [0.9558, 0.9666] | 0.9050 [0.8970, 0.9128] | 0.9272 [0.9202, 0.9340] |
| 8 | 0.9500 [0.9438, 0.9556] | 0.9382 [0.9314, 0.9448] | 0.9554 [0.9498, 0.9606] | 0.9544 [0.9488, 0.9604] | 0.9054 [0.8972, 0.9132] | 0.9196 [0.9114, 0.9270] |
| 9 | 0.9518 [0.9460, 0.9576] | 0.9392 [0.9328, 0.9456] | 0.9514 [0.9452, 0.9568] | 0.9444 [0.9380, 0.9506] | 0.8886 [0.8792, 0.8972] | 0.8922 [0.8834, 0.9008] |
| 10 | 0.9602 [0.9548, 0.9654] | 0.9396 [0.9332, 0.9468] | 0.9620 [0.9566, 0.9672] | 0.9368 [0.9298, 0.9440] | 0.8676 [0.8588, 0.8772] | 0.8574 [0.8478, 0.8668] |
| 11 | 0.9546 [0.9492, 0.9600] | 0.9574 [0.9514, 0.9632] | 0.9640 [0.9590, 0.9688] | 0.9418 [0.9352, 0.9482] | 0.8330 [0.8218, 0.8436] | 0.8714 [0.8630, 0.8804] |
| 12 | 0.9460 [0.9398, 0.9518] | 0.9718 [0.9674, 0.9762] | 0.9648 [0.9596, 0.9696] | 0.9400 [0.9336, 0.9466] | 0.8090 [0.7980, 0.8194] | 0.9166 [0.9088, 0.9242] |
| all 12 | 0.7152 [0.7030, 0.7268] | 0.0012 [0.0004, 0.0022] | 0.8276 [0.8174, 0.8384] | 0.0016 [0.0006, 0.0028] | 0.6148 [0.6010, 0.6286] | 0.0008 [0.0002, 0.0016] |

Source: `experiments/2026-11-05_v5_analysis/a2_single_layer.csv`. Built by `scripts/analyze_v5.py` (`extension_b`) from `experiments/2026-10-28_v2_analysis/rq1_anatomy.csv`, `experiments/2026-10-28_v2_dino_analysis/rq1_anatomy.csv`, `experiments/2026-10-29_v3_deit_b_analysis/rq1_anatomy.csv` (restriction) and `experiments/2026-11-03_ab_t7_*/attention_off.csv` (removal). The "all 12" restriction row is the k = 16 cell of 4.1.

### 4.3 Head vs body patches

**Question.** On animal images, does removing r = 4 head patches lower the log-probability of the original top-1 class more than removing r = 4 body patches?

**Setup.** PartImageNet_Seg; DeiT-S and DINO on all 24,080 images, DeiT-B on the val + test parts (3,614) for compute reasons; own preprocessing; patches removed from all layers (token dropping); within one part r patches drawn at random, seeds 0–4 averaged; outcome log p(original top-1) − log p(after removal); controls: r random object patches, r background patches. Paired comparison on images with both a removable animal head and body (super-categories Quadruped, Biped, Fish, Bird, Snake, Reptile). These are ImageNet training images.

**Registration.** Earlier study, prediction 1 ("the model relies more on the head: removing head patches hurts more than removing body patches"); rule of that study: higher if the interval lower bound > 0.

Head − body, log-probability drop (nats), r = 4:

| Model | Images | Head − body | Verdict (earlier rule) |
| --- | --- | --- | --- |
| DeiT-S | 14693 | 0.0169 [0.0142, 0.0199] | higher |
| DeiT-B | 2234 | 0.0075 [0.0041, 0.0107] | higher |
| DINO | 14815 | 0.0213 [0.0186, 0.0239] | higher |

Mean log-probability drop per part (nats), r = 4; the ratio is the ratio of the two means (no interval):

| Model | Super-category | Head | Body | Head ÷ body |
| --- | --- | --- | --- | --- |
| DeiT-S | Quadruped | 0.0326 [0.0278, 0.0382] (n = 6598) | 0.0150 [0.0114, 0.0188] (n = 6488) | 2.18 |
| DeiT-S | Biped | 0.0374 [0.0286, 0.0480] (n = 2146) | 0.0116 [0.0075, 0.0164] (n = 2187) | 3.21 |
| DeiT-S | Bird | 0.0123 [0.0074, 0.0178] (n = 1564) | 0.0101 [0.0064, 0.0145] (n = 2043) | 1.22 |
| DeiT-S | Fish | 0.0152 [0.0075, 0.0276] (n = 1041) | 0.0148 [0.0088, 0.0216] (n = 1239) | 1.03 |
| DeiT-S | Reptile | 0.0198 [0.0146, 0.0252] (n = 2856) | 0.0226 [0.0172, 0.0281] (n = 3291) | 0.88 |
| DeiT-S | Snake | 0.0239 [0.0186, 0.0301] (n = 1762) | 0.0250 [0.0191, 0.0321] (n = 2684) | 0.96 |
| DeiT-S | control: random object patches | 0.0212 [0.0195, 0.0230] (n = 23471) |  |  |
| DeiT-S | control: random background patches | 0.0012 [0.0011, 0.0014] (n = 23054) |  |  |
| DeiT-B | Quadruped | 0.0090 [0.0054, 0.0132] (n = 1001) | 0.0053 [0.0027, 0.0085] (n = 970) | 1.69 |
| DeiT-B | Biped | 0.0138 [0.0044, 0.0245] (n = 316) | 0.0025 [0.0000, 0.0054] (n = 328) | 5.45 |
| DeiT-B | Bird | 0.0062 [0.0019, 0.0104] (n = 238) | 0.0069 [0.0026, 0.0116] (n = 312) | 0.89 |
| DeiT-B | Fish | 0.0127 [0.0033, 0.0272] (n = 160) | 0.0007 [−0.0092, 0.0109] (n = 182) | 17.23 |
| DeiT-B | Reptile | 0.0092 [0.0046, 0.0158] (n = 451) | 0.0110 [0.0053, 0.0183] (n = 494) | 0.83 |
| DeiT-B | Snake | 0.0134 [0.0063, 0.0219] (n = 265) | 0.0129 [0.0045, 0.0234] (n = 400) | 1.04 |
| DeiT-B | control: random object patches | 0.0115 [0.0086, 0.0154] (n = 3525) |  |  |
| DeiT-B | control: random background patches | 0.0006 [0.0003, 0.0009] (n = 3456) |  |  |
| DINO | Quadruped | 0.0367 [0.0315, 0.0426] (n = 6636) | 0.0105 [0.0080, 0.0131] (n = 6469) | 3.48 |
| DINO | Biped | 0.0327 [0.0259, 0.0403] (n = 2161) | 0.0119 [0.0068, 0.0176] (n = 2197) | 2.75 |
| DINO | Bird | 0.0248 [0.0183, 0.0323] (n = 1602) | 0.0130 [0.0089, 0.0174] (n = 2059) | 1.91 |
| DINO | Fish | 0.0158 [0.0111, 0.0213] (n = 1062) | 0.0155 [0.0107, 0.0217] (n = 1247) | 1.02 |
| DINO | Reptile | 0.0183 [0.0137, 0.0232] (n = 2865) | 0.0174 [0.0135, 0.0217] (n = 3299) | 1.05 |
| DINO | Snake | 0.0278 [0.0213, 0.0355] (n = 1782) | 0.0242 [0.0183, 0.0308] (n = 2682) | 1.15 |
| DINO | control: random object patches | 0.0209 [0.0192, 0.0229] (n = 23505) |  |  |
| DINO | control: random background patches | −0.0001 [−0.0002, 0.0001] (n = 23019) |  |  |

Source: `experiments/2026-10-29_v3_e8_parts_{deit,deit_b,dino}/analysis_comparisons.csv`; `experiments/2026-10-29_v3_e8_parts_{deit,deit_b,dino}/analysis_reliance.csv`. Columns produced by `scripts/analyze_parts.py` (`reliance_log_drop_r4`: paired head − body per image).

### 4.4 Top-k map vs full map

**Question.** How similar is the top-k reading-set map (16 regions, k = 16, layers 9–12) to the full attention map R0 (layers 9–12)?

**Setup.** Per-image Spearman over the 196 patches between the two saved maps; all 12,419 ImageNet-S images; own preprocessing; L4.

**Registration.** Earlier study: median > 0.5 in every model.

| Model | Images | Median | 25th percentile | 75th percentile | Median > 0.5 |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | 12419 | 0.8974 | 0.8687 | 0.9188 | holds |
| DeiT-B | 12419 | 0.9118 | 0.8860 | 0.9331 | holds |
| DINO | 12419 | 0.9017 | 0.8756 | 0.9228 | holds |

Source: `experiments/2026-10-31_s_analysis/s1_similarity.csv`. No bootstrap interval was computed for the median.

### 4.5 M1: share of spatial variance explained by the mean map

**Question.** How much of each layer's R0 map is image-independent? **Setup.** R0 per layer; mean maps from the 2,000 held-out test images (own crop); evaluated on all 12,419 ImageNet-S images (definition in 2.7). **Registration.** Descriptive (part of P2).

| Layer | DeiT-S | DeiT-B | DINO |
| --- | --- | --- | --- |
| 1 | 0.0679 [0.0658, 0.0697] | 0.1239 [0.1213, 0.1263] | 0.1120 [0.1096, 0.1144] |
| 2 | 0.2115 [0.2098, 0.2132] | 0.1402 [0.1385, 0.1420] | 0.0740 [0.0728, 0.0753] |
| 3 | 0.1372 [0.1358, 0.1384] | 0.1499 [0.1482, 0.1515] | 0.1966 [0.1946, 0.1984] |
| 4 | 0.0686 [0.0674, 0.0698] | 0.0747 [0.0736, 0.0757] | 0.1462 [0.1440, 0.1483] |
| 5 | 0.0780 [0.0768, 0.0791] | 0.0580 [0.0570, 0.0590] | 0.1463 [0.1440, 0.1485] |
| 6 | 0.0520 [0.0510, 0.0530] | 0.0624 [0.0613, 0.0634] | 0.1195 [0.1173, 0.1216] |
| 7 | 0.0440 [0.0428, 0.0452] | 0.0541 [0.0529, 0.0552] | 0.1235 [0.1215, 0.1255] |
| 8 | 0.0422 [0.0408, 0.0435] | 0.0542 [0.0529, 0.0557] | 0.0659 [0.0644, 0.0675] |
| 9 | 0.0423 [0.0410, 0.0435] | 0.0571 [0.0556, 0.0587] | 0.0946 [0.0921, 0.0968] |
| 10 | 0.0432 [0.0419, 0.0446] | 0.0551 [0.0537, 0.0566] | 0.1521 [0.1492, 0.1549] |
| 11 | 0.0411 [0.0400, 0.0422] | 0.0576 [0.0561, 0.0591] | 0.1670 [0.1640, 0.1699] |
| 12 | 0.0327 [0.0316, 0.0337] | 0.0495 [0.0484, 0.0507] | 0.1355 [0.1326, 0.1382] |

Source: `experiments/2026-11-05_v5_analysis/fig3_diagnose.csv (panel M1, variant R0)`.

Radial share of the R0 mean map (share of its spatial variance explained by distance from the centre), layers 9–12:

| Model | Geometry | Layer 9 | Layer 10 | Layer 11 | Layer 12 |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | ImageNet-S (own crop) | 0.8561 | 0.8417 | 0.8326 | 0.8239 |
| DeiT-S | ClickMe | 0.8681 | 0.8542 | 0.8492 | 0.8203 |
| DeiT-B | ImageNet-S (own crop) | 0.8366 | 0.8293 | 0.8307 | 0.8203 |
| DeiT-B | ClickMe | 0.8327 | 0.8256 | 0.8252 | 0.8182 |
| DINO | ImageNet-S (own crop) | 0.9238 | 0.9700 | 0.9804 | 0.9507 |
| DINO | ClickMe | 0.9350 | 0.9756 | 0.9812 | 0.9580 |

Source: `experiments/2026-11-05_diagnose_m1_{deit,deit_b,dino}/radial.csv`.

### 4.6 M2: position-probe accuracy

**Question.** Does token j still encode patch position j? **Setup.** Linear probe per layer (Section 2.2) trained on 2,000 held-out images, tested on the other 2,000 (all 196 patch tokens). The last column of each model gives the earlier probe (400 training images, earlier run; no interval stored). **Registration.** P2 (replication).

| Layer | DeiT-S | DeiT-S earlier probe | DeiT-B | DeiT-B earlier probe | DINO | DINO earlier probe |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0000 [1.0000, 1.0000] | 0.9992 | 1.0000 [1.0000, 1.0000] | 0.9997 | 0.9994 [0.9993, 0.9995] | 0.9951 |
| 2 | 1.0000 [1.0000, 1.0000] | 0.9996 | 1.0000 [1.0000, 1.0000] | 1.0000 | 0.9996 [0.9995, 0.9997] | 0.9758 |
| 3 | 0.9995 [0.9994, 0.9996] | 0.9829 | 1.0000 [1.0000, 1.0000] | 0.9999 | 0.9988 [0.9985, 0.9990] | 0.9744 |
| 4 | 0.9904 [0.9901, 0.9908] | 0.8839 | 0.9999 [0.9999, 1.0000] | 0.9993 | 0.9982 [0.9979, 0.9984] | 0.9603 |
| 5 | 0.9746 [0.9739, 0.9752] | 0.8114 | 0.9994 [0.9993, 0.9994] | 0.9947 | 0.9956 [0.9952, 0.9959] | 0.9362 |
| 6 | 0.8534 [0.8514, 0.8553] | 0.6288 | 0.9947 [0.9944, 0.9950] | 0.9659 | 0.9890 [0.9885, 0.9896] | 0.8457 |
| 7 | 0.6505 [0.6477, 0.6529] | 0.4529 | 0.9683 [0.9677, 0.9690] | 0.8812 | 0.9587 [0.9573, 0.9599] | 0.7152 |
| 8 | 0.4390 [0.4364, 0.4414] | 0.3174 | 0.9184 [0.9171, 0.9197] | 0.7556 | 0.8562 [0.8536, 0.8586] | 0.5834 |
| 9 | 0.3101 [0.3079, 0.3124] | 0.2331 | 0.8365 [0.8345, 0.8384] | 0.6170 | 0.6738 [0.6707, 0.6768] | 0.4467 |
| 10 | 0.2315 [0.2294, 0.2335] | 0.1817 | 0.6683 [0.6658, 0.6708] | 0.4547 | 0.5048 [0.5019, 0.5078] | 0.3379 |
| 11 | 0.1937 [0.1918, 0.1956] | 0.1531 | 0.4915 [0.4886, 0.4940] | 0.3353 | 0.3923 [0.3896, 0.3950] | 0.2726 |
| 12 | 0.1812 [0.1794, 0.1830] | 0.1459 | 0.3177 [0.3148, 0.3203] | 0.2212 | 0.3575 [0.3549, 0.3600] | 0.2544 |

Source: `experiments/2026-11-05_v5_analysis/fig3_diagnose.csv (panel M2)`; `per image: experiments/2026-11-05_diagnose_probe_*/identity_images.csv`.

Mean of the largest probe probability per token (held-out test tokens):

| Model | Layer 6 | Layer 9 | Layer 10 | Layer 11 | Layer 12 |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | 0.8155 | 0.3960 | 0.3199 | 0.2791 | 0.2657 |
| DeiT-B | 0.9910 | 0.8807 | 0.7761 | 0.6500 | 0.5004 |
| DINO | 0.9580 | 0.6636 | 0.5438 | 0.4617 | 0.4355 |

Source: `experiments/2026-11-05_diagnose_probe_{deit,deit_b,dino}/identity.csv (column top_probability)`. No interval stored.

### 4.7 M3: share of R0 on high-norm tokens, and value-norm ratio

**Question.** How much of R0 falls on high-norm tokens, and how long are their value vectors? **Setup.** Share of single-layer R0 on that layer's high-norm patches; all 12,419 ImageNet-S images; own preprocessing; earlier run on an L4. **Registration.** P2 (replication).

| Layer | DeiT-S | DeiT-B | DINO |
| --- | --- | --- | --- |
| 1 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0002 [0.0002, 0.0003] |
| 3 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0001] |
| 4 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 5 | 0.0010 [0.0009, 0.0010] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 6 | 0.0264 [0.0260, 0.0267] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 7 | 0.0742 [0.0737, 0.0747] | 0.0354 [0.0350, 0.0358] | 0.0000 [0.0000, 0.0000] |
| 8 | 0.0929 [0.0922, 0.0936] | 0.1290 [0.1283, 0.1297] | 0.0000 [0.0000, 0.0000] |
| 9 | 0.0928 [0.0921, 0.0935] | 0.2153 [0.2144, 0.2163] | 0.0000 [0.0000, 0.0000] |
| 10 | 0.0911 [0.0903, 0.0918] | 0.2810 [0.2796, 0.2824] | 0.0000 [0.0000, 0.0000] |
| 11 | 0.0560 [0.0552, 0.0567] | 0.4354 [0.4339, 0.4368] | 0.0000 [0.0000, 0.0000] |
| 12 | 0.0723 [0.0712, 0.0733] | 0.2221 [0.2203, 0.2240] | 0.0000 [0.0000, 0.0000] |

Source: `experiments/2026-11-05_v5_analysis/fig3_diagnose.csv (panel M3, from experiments/2026-10-31_s_analysis/s3_layers.csv)`.

**Value-norm ratio** (definition in 2.7). Attention-budget study, formal split of 11,919 ImageNet-S images, uniform preprocessing (Resize 256, crop 0.875), all heads; layers without high-norm patches have no value. Per-layer values recomputed from the run files with the code of `scripts/analyze_budget.py` (sum over heads of the per-image ratios ÷ number of image–head pairs); the layer 9–12 mean is the stored value. No interval was computed.

| Model | L6 | L7 | L8 | L9 | L10 | L11 | L12 | Mean L9–12 (stored) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeiT-S | 0.870 | 0.609 | 0.381 | 0.214 | 0.295 | 0.273 | 0.387 | 0.2920 |
| DeiT-B | — | 0.605 | 0.502 | 0.268 | 0.192 | 0.210 | 0.248 | 0.2296 |

Source: `experiments/2026-11-03_ab_t3_{deit,deit_b}/heads.csv`; `experiments/2026-11-03_ab_analysis_formal/summary.json (heads.value_ratio)`. DINO has no high-norm patches in this split (value: NaN in `summary.json`).

### 4.8 Per-layer ClickMe quality of R0

**Question.** How does the ClickMe score of R0 change across layers? **Setup.** Single-layer R0 maps, ClickMe image-specific Spearman, 17,408 images; earlier run on an L4. Descriptive.

| Layer | DeiT-S | DeiT-B | DINO |
| --- | --- | --- | --- |
| 1 | −0.1080 [−0.1118, −0.1040] | −0.1108 [−0.1146, −0.1071] | −0.0891 [−0.0929, −0.0854] |
| 2 | 0.0074 [0.0048, 0.0100] | 0.0407 [0.0371, 0.0443] | 0.0636 [0.0596, 0.0674] |
| 3 | 0.0807 [0.0779, 0.0835] | 0.0040 [0.0012, 0.0068] | 0.0561 [0.0523, 0.0599] |
| 4 | 0.1316 [0.1284, 0.1350] | 0.1102 [0.1068, 0.1137] | 0.1399 [0.1358, 0.1439] |
| 5 | 0.1477 [0.1443, 0.1513] | 0.1299 [0.1264, 0.1333] | 0.1499 [0.1460, 0.1537] |
| 6 | 0.1361 [0.1327, 0.1396] | 0.1511 [0.1478, 0.1545] | 0.1672 [0.1632, 0.1713] |
| 7 | 0.1472 [0.1440, 0.1506] | 0.1314 [0.1283, 0.1347] | 0.1498 [0.1460, 0.1538] |
| 8 | 0.1492 [0.1457, 0.1528] | 0.1179 [0.1150, 0.1209] | 0.1703 [0.1663, 0.1744] |
| 9 | 0.1840 [0.1805, 0.1876] | 0.1122 [0.1091, 0.1152] | 0.1982 [0.1940, 0.2025] |
| 10 | 0.1712 [0.1681, 0.1745] | 0.1391 [0.1361, 0.1420] | 0.2067 [0.2023, 0.2113] |
| 11 | 0.1489 [0.1459, 0.1520] | 0.1201 [0.1173, 0.1231] | 0.2114 [0.2067, 0.2159] |
| 12 | 0.0983 [0.0953, 0.1011] | 0.0228 [0.0203, 0.0253] | 0.1795 [0.1751, 0.1841] |

Source: `experiments/2026-11-05_v5_analysis/fig3_diagnose.csv (panel quality, from experiments/2026-10-31_s_analysis/s2_layers.csv)`.

### 4.9 Table 1 (full): readings, stand-ins, references

**Question.** How do the readings, stand-ins and references score on the three metrics? **Setup.** Layers 9–12 summed unless stated. SRG: 2,000 ImageNet-S images; enrichment: ImageNet-S (n per cell below); ClickMe: 17,408 images. Own preprocessing; L4; code `5faa503`. **Registration.** Rows and metrics fixed before the runs; verdicts in 4.10.

| Reading | DeiT-S SRG | DeiT-S Enrichment | DeiT-S ClickMe | DeiT-B SRG | DeiT-B Enrichment | DeiT-B ClickMe | DINO SRG | DINO Enrichment | DINO ClickMe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reference: random | −0.0017 [−0.0089, 0.0055] | 0.0000 [−0.0008, 0.0007] | 0.0004 [−0.0014, 0.0023] | −0.0028 [−0.0094, 0.0037] | 0.0000 [−0.0008, 0.0007] | 0.0004 [−0.0014, 0.0023] | −0.0021 [−0.0089, 0.0051] | 0.0000 [−0.0007, 0.0007] | 0.0004 [−0.0014, 0.0023] |
| Reference: centre (centred Gaussian) | 0.1103 [0.0998, 0.1204] | 0.0001 [−0.0003, 0.0005] | 0.0000 [−0.0050, 0.0048] | 0.0990 [0.0894, 0.1089] | 0.0001 [−0.0003, 0.0005] | 0.0000 [−0.0050, 0.0048] | 0.1405 [0.1294, 0.1508] | 0.0000 [−0.0004, 0.0003] | 0.0000 [−0.0050, 0.0048] |
| Reference: mean map | 0.0583 [0.0484, 0.0676] | −0.0033 [−0.0041, −0.0026] | 0.0000 [−0.0041, 0.0041] | 0.0107 [0.0041, 0.0166] | −0.0022 [−0.0026, −0.0018] | 0.0000 [−0.0017, 0.0016] | 0.1391 [0.1277, 0.1501] | 0.0026 [0.0019, 0.0032] | 0.0000 [−0.0050, 0.0051] |
| Reference: GAE | 0.3450 [0.3336, 0.3547] | 0.2036 [0.2002, 0.2068] | 0.2340 [0.2301, 0.2380] | 0.3077 [0.2975, 0.3186] | 0.1957 [0.1926, 0.1990] | 0.2133 [0.2094, 0.2178] | 0.4175 [0.4064, 0.4275] | 0.2214 [0.2179, 0.2250] | 0.2343 [0.2299, 0.2388] |
| R0 | 0.2360 [0.2255, 0.2461] | 0.1401 [0.1381, 0.1423] | 0.1565 [0.1535, 0.1595] | 0.1694 [0.1595, 0.1785] | 0.0805 [0.0788, 0.0823] | 0.1071 [0.1045, 0.1099] | 0.2850 [0.2735, 0.2962] | 0.1994 [0.1962, 0.2025] | 0.2076 [0.2032, 0.2123] |
| C1 | 0.2076 [0.1974, 0.2169] | 0.1377 [0.1356, 0.1400] | 0.1655 [0.1625, 0.1686] | 0.1439 [0.1349, 0.1531] | 0.0773 [0.0756, 0.0791] | 0.0870 [0.0845, 0.0896] | 0.2335 [0.2213, 0.2452] | 0.2013 [0.1979, 0.2045] | 0.2518 [0.2475, 0.2564] |
| Stand-in for C1: centred Gaussian subtraction | 0.1673 [0.1560, 0.1775] | 0.1329 [0.1308, 0.1351] | 0.1684 [0.1648, 0.1721] | 0.0238 [0.0145, 0.0331] | 0.0555 [0.0542, 0.0569] | 0.0622 [0.0576, 0.0669] | 0.2322 [0.2205, 0.2441] | 0.2001 [0.1968, 0.2034] | 0.2525 [0.2482, 0.2573] |
| C2 | 0.0755 [0.0656, 0.0838] | 0.0414 [0.0399, 0.0429] | 0.0839 [0.0803, 0.0874] | 0.0510 [0.0437, 0.0592] | 0.0269 [0.0257, 0.0281] | 0.0501 [0.0471, 0.0532] | 0.1411 [0.1307, 0.1514] | 0.0845 [0.0826, 0.0862] | 0.1432 [0.1395, 0.1468] |
| Stand-in for C2: matched blur | 0.1809 [0.1703, 0.1915] | 0.0950 [0.0919, 0.0980] | 0.1520 [0.1473, 0.1568] | 0.0975 [0.0881, 0.1065] | −0.0081 [−0.0098, −0.0066] | 0.0680 [0.0636, 0.0721] | 0.2870 [0.2752, 0.2988] | 0.1893 [0.1856, 0.1928] | 0.2261 [0.2210, 0.2313] |
| C3 | 0.2533 [0.2424, 0.2633] | 0.1777 [0.1749, 0.1805] | 0.1899 [0.1864, 0.1932] | 0.1981 [0.1884, 0.2076] | 0.1459 [0.1431, 0.1486] | 0.1647 [0.1615, 0.1681] | 0.2853 [0.2740, 0.2964] | 0.1959 [0.1927, 0.1991] | 0.2069 [0.2024, 0.2115] |
| Stand-in for C3: zero high-norm tokens | 0.2409 [0.2304, 0.2509] | 0.1520 [0.1495, 0.1545] | 0.1678 [0.1646, 0.1710] | 0.1856 [0.1758, 0.1955] | 0.1206 [0.1181, 0.1232] | 0.1476 [0.1446, 0.1508] | 0.2850 [0.2735, 0.2962] | 0.1994 [0.1962, 0.2025] | 0.2076 [0.2032, 0.2123] |
| R_all (C3→C2→C1) | 0.0764 [0.0666, 0.0853] | 0.0589 [0.0572, 0.0606] | 0.1230 [0.1188, 0.1270] | 0.0850 [0.0769, 0.0934] | 0.0793 [0.0774, 0.0813] | 0.1370 [0.1336, 0.1404] | 0.0901 [0.0802, 0.1003] | 0.0880 [0.0861, 0.0900] | 0.1768 [0.1733, 0.1804] |
| +register (DeiT-S) | 0.2360 [0.2255, 0.2458] | 0.1521 [0.1496, 0.1545] | 0.1694 [0.1661, 0.1725] | — | — | — | — | — | — |
| Top-k training (MiTA-DeiT-S) | not computed (see 2.5) | 0.1001 [0.0982, 0.1019] | 0.1148 [0.1121, 0.1176] | — | — | — | — | — | — |

Images per cell: DeiT-S: SRG 2,000, enrichment 12,150, ClickMe 17,408; DeiT-B: SRG 2,000, enrichment 12,150, ClickMe 17,408; DINO: SRG 2,000, enrichment 12,144, ClickMe 17,408; +register 2,000 (SRG); +register 12,150 (Enrichment); +register 17,408 (ClickMe); top-k training 12,144 (Enrichment); top-k training 17,408 (ClickMe).

Source: `experiments/2026-11-05_v5_analysis/table1_long.csv`; `experiments/2026-11-05_v5_analysis/variants.csv (test = value)`.

Single-layer and early-layer rows of the same runs:

| Reading | DeiT-S SRG | DeiT-S Enrichment | DeiT-S ClickMe | DeiT-B SRG | DeiT-B Enrichment | DeiT-B ClickMe | DINO SRG | DINO Enrichment | DINO ClickMe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0, layer 8 only | 0.2011 [0.1910, 0.2114] | 0.0857 [0.0834, 0.0881] | 0.1492 [0.1457, 0.1528] | 0.1558 [0.1467, 0.1646] | 0.0558 [0.0543, 0.0574] | 0.1179 [0.1150, 0.1209] | 0.2022 [0.1911, 0.2125] | 0.1034 [0.1009, 0.1058] | 0.1701 [0.1662, 0.1742] |
| R0, layer 9 only | 0.2387 [0.2280, 0.2485] | 0.1397 [0.1373, 0.1422] | 0.1841 [0.1806, 0.1875] | 0.1620 [0.1529, 0.1707] | 0.0588 [0.0571, 0.0606] | 0.1122 [0.1091, 0.1152] | 0.2503 [0.2391, 0.2621] | 0.1508 [0.1480, 0.1536] | 0.1980 [0.1939, 0.2024] |
| R0, layer 10 only | 0.2435 [0.2327, 0.2533] | 0.1467 [0.1444, 0.1492] | 0.1714 [0.1681, 0.1746] | 0.1936 [0.1842, 0.2037] | 0.1113 [0.1092, 0.1135] | 0.1391 [0.1361, 0.1420] | 0.2777 [0.2666, 0.2891] | 0.2027 [0.1993, 0.2060] | 0.2065 [0.2021, 0.2109] |
| R0, layer 11 only | 0.2237 [0.2136, 0.2337] | 0.1521 [0.1497, 0.1544] | 0.1490 [0.1461, 0.1521] | 0.1686 [0.1587, 0.1785] | 0.1219 [0.1198, 0.1241] | 0.1201 [0.1173, 0.1231] | 0.2990 [0.2878, 0.3097] | 0.2240 [0.2208, 0.2275] | 0.2113 [0.2067, 0.2159] |
| R0, layer 12 only | 0.1942 [0.1841, 0.2039] | 0.1145 [0.1125, 0.1165] | 0.0984 [0.0955, 0.1013] | 0.1126 [0.1036, 0.1213] | 0.0344 [0.0327, 0.0360] | 0.0228 [0.0203, 0.0253] | 0.2553 [0.2438, 0.2661] | 0.1664 [0.1634, 0.1695] | 0.1794 [0.1750, 0.1840] |
| C1, layer 12 only | 0.1467 [0.1373, 0.1557] | 0.1008 [0.0990, 0.1027] | 0.0882 [0.0851, 0.0910] | 0.0565 [0.0481, 0.0641] | 0.0107 [0.0095, 0.0120] | −0.0190 [−0.0217, −0.0162] | 0.2137 [0.2019, 0.2251] | 0.1669 [0.1638, 0.1703] | 0.2147 [0.2103, 0.2191] |
| C2, layer 12 only | 0.0227 [0.0145, 0.0309] | 0.0109 [0.0095, 0.0122] | 0.0146 [0.0103, 0.0186] | −0.0436 [−0.0515, −0.0363] | −0.0284 [−0.0296, −0.0273] | −0.0688 [−0.0723, −0.0652] | 0.0680 [0.0591, 0.0770] | 0.0398 [0.0386, 0.0410] | 0.0760 [0.0728, 0.0790] |
| C3, layer 12 only | 0.1981 [0.1879, 0.2081] | 0.1377 [0.1355, 0.1397] | 0.1246 [0.1215, 0.1276] | 0.1361 [0.1270, 0.1454] | 0.0629 [0.0610, 0.0647] | 0.0749 [0.0722, 0.0777] | 0.2577 [0.2461, 0.2685] | 0.1634 [0.1605, 0.1664] | 0.1807 [0.1763, 0.1852] |
| R_all, layer 12 only | 0.0322 [0.0234, 0.0414] | 0.0277 [0.0261, 0.0293] | 0.0408 [0.0363, 0.0448] | −0.0090 [−0.0166, −0.0022] | −0.0003 [−0.0015, 0.0011] | −0.0183 [−0.0216, −0.0149] | 0.0345 [0.0258, 0.0432] | 0.0434 [0.0421, 0.0446] | 0.0934 [0.0904, 0.0963] |
| R0, layers 1–4 | — (no SRG by design) | 0.0278 [0.0264, 0.0293] | 0.0344 [0.0312, 0.0373] | — (no SRG by design) | 0.0160 [0.0143, 0.0176] | 0.0027 [−0.0008, 0.0062] | — (no SRG by design) | 0.0632 [0.0610, 0.0655] | 0.0775 [0.0737, 0.0816] |
| C2, layers 1–4 | — (no SRG by design) | 0.0268 [0.0254, 0.0282] | 0.0342 [0.0311, 0.0371] | — (no SRG by design) | 0.0160 [0.0144, 0.0176] | 0.0027 [−0.0008, 0.0062] | — (no SRG by design) | 0.0596 [0.0574, 0.0617] | 0.0759 [0.0720, 0.0800] |

Source: `experiments/2026-11-05_v5_analysis/table1_long.csv`.

### 4.10 Corrections vs R0 and vs their stand-ins

**Procedure.** Per-image paired differences, bootstrap, verdict rule of 2.9. **Registration.** Primary (P7): correction − stand-in; R_all − R0 (P3 for the four supervised enrichment and ClickMe cells). Descriptive: C1/C2/C3 − R0, stand-in − R0.

**DeiT-S**

| Comparison | SRG | Same-ring enrichment | ClickMe |
| --- | --- | --- | --- |
| C1 − R0 | −0.0285 [−0.0342, −0.0230] comparable | −0.0024 [−0.0029, −0.0017] comparable | 0.0090 [0.0076, 0.0103] comparable |
| C2 − R0 | −0.1605 [−0.1712, −0.1505] lower | −0.0987 [−0.1008, −0.0968] lower | −0.0726 [−0.0762, −0.0693] lower |
| C3 − R0 | 0.0173 [0.0130, 0.0213] comparable | 0.0376 [0.0363, 0.0389] higher | 0.0333 [0.0325, 0.0342] higher |
| R_all − R0 (primary; P3) | −0.1596 [−0.1706, −0.1493] lower | −0.0812 [−0.0834, −0.0793] lower | −0.0335 [−0.0376, −0.0296] lower |
| C1 − centred Gaussian subtraction (primary) | 0.0403 [0.0343, 0.0470] comparable | 0.0049 [0.0043, 0.0054] comparable | −0.0029 [−0.0049, −0.0009] comparable |
| C2 − matched blur (primary) | −0.1054 [−0.1163, −0.0949] lower | −0.0536 [−0.0564, −0.0509] lower | −0.0681 [−0.0730, −0.0635] lower |
| C3 − zero high-norm (primary) | 0.0124 [0.0079, 0.0163] comparable | 0.0256 [0.0245, 0.0267] higher | 0.0220 [0.0213, 0.0228] higher |
| Centred Gaussian subtraction − R0 | −0.0688 [−0.0763, −0.0616] lower | −0.0072 [−0.0078, −0.0067] comparable | 0.0119 [0.0088, 0.0148] comparable |
| Matched blur − R0 | −0.0551 [−0.0640, −0.0457] lower | −0.0451 [−0.0475, −0.0426] lower | −0.0045 [−0.0077, −0.0014] comparable |
| Zero high-norm − R0 | 0.0049 [0.0021, 0.0079] comparable | 0.0120 [0.0110, 0.0129] comparable | 0.0113 [0.0108, 0.0118] comparable |

**DeiT-B**

| Comparison | SRG | Same-ring enrichment | ClickMe |
| --- | --- | --- | --- |
| C1 − R0 | −0.0255 [−0.0310, −0.0194] comparable | −0.0032 [−0.0037, −0.0027] comparable | −0.0201 [−0.0210, −0.0192] lower |
| C2 − R0 | −0.1184 [−0.1267, −0.1096] lower | −0.0536 [−0.0552, −0.0522] lower | −0.0570 [−0.0594, −0.0547] lower |
| C3 − R0 | 0.0287 [0.0240, 0.0338] comparable | 0.0654 [0.0634, 0.0675] higher | 0.0576 [0.0563, 0.0588] higher |
| R_all − R0 (primary; P3) | −0.0843 [−0.0928, −0.0763] lower | −0.0012 [−0.0029, 0.0006] comparable | 0.0299 [0.0273, 0.0324] higher |
| C1 − centred Gaussian subtraction (primary) | 0.1201 [0.1094, 0.1305] higher | 0.0218 [0.0210, 0.0227] higher | 0.0248 [0.0197, 0.0295] higher |
| C2 − matched blur (primary) | −0.0465 [−0.0548, −0.0371] comparable | 0.0350 [0.0337, 0.0365] higher | −0.0178 [−0.0205, −0.0153] comparable |
| C3 − zero high-norm (primary) | 0.0125 [0.0081, 0.0178] comparable | 0.0252 [0.0242, 0.0264] higher | 0.0171 [0.0161, 0.0181] comparable |
| Centred Gaussian subtraction − R0 | −0.1456 [−0.1558, −0.1348] lower | −0.0250 [−0.0258, −0.0241] lower | −0.0449 [−0.0498, −0.0401] lower |
| Matched blur − R0 | −0.0719 [−0.0797, −0.0641] lower | −0.0886 [−0.0906, −0.0867] lower | −0.0392 [−0.0422, −0.0360] lower |
| Zero high-norm − R0 | 0.0162 [0.0122, 0.0201] comparable | 0.0401 [0.0383, 0.0421] higher | 0.0405 [0.0391, 0.0419] higher |

**DINO**

| Comparison | SRG | Same-ring enrichment | ClickMe |
| --- | --- | --- | --- |
| C1 − R0 | −0.0515 [−0.0593, −0.0439] lower | 0.0018 [0.0010, 0.0028] comparable | 0.0442 [0.0405, 0.0476] higher |
| C2 − R0 | −0.1439 [−0.1534, −0.1340] lower | −0.1150 [−0.1172, −0.1127] lower | −0.0644 [−0.0667, −0.0623] lower |
| C3 − R0 | 0.0003 [−0.0032, 0.0039] comparable | −0.0035 [−0.0040, −0.0030] comparable | −0.0007 [−0.0010, −0.0005] comparable |
| R_all − R0 (primary; P3) | −0.1949 [−0.2050, −0.1838] lower | −0.1114 [−0.1137, −0.1092] lower | −0.0309 [−0.0347, −0.0274] lower |
| C1 − centred Gaussian subtraction (primary) | 0.0013 [−0.0029, 0.0055] comparable | 0.0012 [0.0008, 0.0017] comparable | −0.0007 [−0.0013, −0.0002] comparable |
| C2 − matched blur (primary) | −0.1458 [−0.1553, −0.1366] lower | −0.1048 [−0.1072, −0.1022] lower | −0.0829 [−0.0855, −0.0803] lower |
| C3 − zero high-norm (primary) | 0.0003 [−0.0032, 0.0039] comparable | −0.0035 [−0.0040, −0.0030] comparable | −0.0007 [−0.0010, −0.0005] comparable |
| Centred Gaussian subtraction − R0 | −0.0528 [−0.0602, −0.0453] lower | 0.0006 [−0.0002, 0.0015] comparable | 0.0449 [0.0414, 0.0483] higher |
| Matched blur − R0 | 0.0020 [−0.0048, 0.0088] comparable | −0.0102 [−0.0117, −0.0087] comparable | 0.0184 [0.0175, 0.0194] comparable |
| Zero high-norm − R0 | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |

Source: `experiments/2026-11-05_v5_analysis/correct_tests.csv`.

P3 cells "higher": 1/4 → partly holds (pre-agreed reading 2). Cells: DeiT-S Enrichment −0.0812 lower; DeiT-S ClickMe −0.0335 lower; DeiT-B Enrichment −0.0012 comparable; DeiT-B ClickMe 0.0299 higher.

### 4.11 Change the model

**Setup.** +register: DeiT-S with the test-time register; top-k training: MiTA-DeiT-S (own crop 0.875). R0 of each, layers 9–12, paired with DeiT-S R0 by image index. **Registration.** P6 part (2): variant − DeiT-S R0 not "higher" on enrichment and ClickMe.

| Row | Metric | Images | Variant − DeiT-S R0 | Verdict | P6 (2) |
| --- | --- | --- | --- | --- | --- |
| +register | SRG | 2000 | 0.0000 [−0.0052, 0.0045] | comparable |  |
| +register | Enrichment | 12150 | 0.0120 [0.0109, 0.0132] | comparable | not higher |
| +register | ClickMe | 17408 | 0.0128 [0.0124, 0.0133] | comparable | not higher |
| Top-k training | SRG | — | not computed (2.5) |  |  |
| Top-k training | Enrichment | 12118 | −0.0400 [−0.0415, −0.0386] | lower | not higher |
| Top-k training | ClickMe | 17408 | −0.0417 [−0.0440, −0.0395] | lower | not higher |
| For comparison: C3 − R0, DeiT-S | SRG | 2000 | 0.0173 [0.0130, 0.0213] | comparable |  |
| For comparison: C3 − R0, DeiT-S | Enrichment | 12150 | 0.0376 [0.0363, 0.0389] | higher |  |
| For comparison: C3 − R0, DeiT-S | ClickMe | 17408 | 0.0333 [0.0325, 0.0342] | higher |  |

Source: `experiments/2026-11-05_v5_analysis/variants.csv`; `experiments/2026-11-05_v5_analysis/correct_tests.csv`. P6: 4/4 cells not higher → holds.

Register checks from the attention-budget study: on the 500 pilot images, shifting the 8 units reduced high-norm patch-layers by 0.9275 (agreement 0.990); 8 random units: reductions −0.0474 to −0.0375 (`experiments/2026-11-03_ab_t2_deit/gate.csv`). On the formal split: reduction 0.9150, ImageNet accuracy 79.828% before and 79.828% after (`experiments/2026-11-03_ab_analysis_formal/summary.json`, H4). DeiT-B has no register row: its unit search did not pass the registered gate (Section 6).

### 4.12 Combination orders

**Setup.** Separate runs (`correct_order`) computing R0, R_all (C3→C2→C1) and the other order (C3→C1→C2) on the same images as Table 1. **Registration.** R_all is the registered combination; the other order is a descriptive check ("robust" if both orders have the same verdict against R0 in every cell). The other order is not used for any verdict.

Absolute values in the order runs:

| Model | Metric | R0 | R_all (C3→C2→C1) | Other order (C3→C1→C2) |
| --- | --- | --- | --- | --- |
| DeiT-S | SRG | 0.2360 [0.2255, 0.2461] | 0.0764 [0.0666, 0.0853] | 0.1677 [0.1569, 0.1778] |
| DeiT-S | Enrichment | 0.1401 [0.1381, 0.1423] | 0.0589 [0.0572, 0.0606] | 0.1077 [0.1057, 0.1097] |
| DeiT-S | ClickMe | 0.1565 [0.1535, 0.1595] | 0.1230 [0.1188, 0.1270] | 0.2211 [0.2169, 0.2252] |
| DeiT-B | SRG | 0.1694 [0.1595, 0.1785] | 0.0850 [0.0769, 0.0934] | 0.1593 [0.1500, 0.1697] |
| DeiT-B | Enrichment | 0.0805 [0.0788, 0.0823] | 0.0793 [0.0774, 0.0813] | 0.1226 [0.1202, 0.1250] |
| DeiT-B | ClickMe | 0.1071 [0.1045, 0.1099] | 0.1370 [0.1336, 0.1404] | 0.1995 [0.1955, 0.2032] |
| DINO | SRG | 0.2850 [0.2735, 0.2962] | 0.0901 [0.0802, 0.1003] | 0.2227 [0.2112, 0.2341] |
| DINO | Enrichment | 0.1994 [0.1962, 0.2025] | 0.0880 [0.0861, 0.0900] | 0.1555 [0.1529, 0.1583] |
| DINO | ClickMe | 0.2076 [0.2032, 0.2123] | 0.1768 [0.1733, 0.1804] | 0.2627 [0.2578, 0.2674] |

Computed for this dossier with `per_image` and `summary` of `scripts/analyze_v5.py` on `experiments/2026-11-05_correct_order_{imagenet_s,clickme}_{deit,deit_b,dino}` (helper output `order_abs.csv`, not part of the repository).

Paired differences:

| Model | Metric | R_all − R0 | Other order − R0 | Other order − R_all | Same verdict vs R0 |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | SRG | −0.1596 [−0.1706, −0.1493] lower | −0.0683 [−0.0767, −0.0604] lower | 0.0913 [0.0816, 0.1005] higher | yes |
| DeiT-S | Enrichment | −0.0812 [−0.0834, −0.0793] lower | −0.0323 [−0.0341, −0.0305] lower | 0.0488 [0.0475, 0.0502] higher | yes |
| DeiT-S | ClickMe | −0.0335 [−0.0376, −0.0296] lower | 0.0646 [0.0616, 0.0676] higher | 0.0981 [0.0945, 0.1023] higher | no |
| DeiT-B | SRG | −0.0843 [−0.0928, −0.0763] lower | −0.0100 [−0.0162, −0.0027] comparable | 0.0743 [0.0665, 0.0827] higher | no |
| DeiT-B | Enrichment | −0.0012 [−0.0029, 0.0006] comparable | 0.0421 [0.0403, 0.0441] higher | 0.0433 [0.0422, 0.0445] higher | no |
| DeiT-B | ClickMe | 0.0299 [0.0273, 0.0324] higher | 0.0923 [0.0899, 0.0948] higher | 0.0625 [0.0600, 0.0652] higher | yes |
| DINO | SRG | −0.1949 [−0.2050, −0.1838] lower | −0.0623 [−0.0707, −0.0535] lower | 0.1325 [0.1231, 0.1420] higher | yes |
| DINO | Enrichment | −0.1114 [−0.1137, −0.1092] lower | −0.0439 [−0.0457, −0.0422] lower | 0.0675 [0.0661, 0.0689] higher | yes |
| DINO | ClickMe | −0.0309 [−0.0347, −0.0274] lower | 0.0550 [0.0515, 0.0586] higher | 0.0859 [0.0834, 0.0885] higher | no |

Source: `experiments/2026-11-05_v5_analysis/a7_order.csv`. Same verdict in 5/9 cells → not "robust" by the registered description.

### 4.13 Last-layer recovery

**Definition.** Recovered share = (corrected layer-12 map − R0 layer 12) ÷ (R0 at the best of layers 8–11 by mean − R0 layer 12), per-image paired bootstrap of the ratio of means. **Registration.** P5, descriptive. Supervised models only.

| Model | Metric | Best R0 layer (8–11) | Drop to layer 12 | C3 | C1 | C2 | R_all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeiT-S | SRG | layer 10 | 0.0493 | 0.0782 [0.0042, 0.1509] | −0.9643 [−1.2460, −0.7496] | −3.4790 [−4.3749, −2.8670] | −3.2863 [−4.1498, −2.7117] |
| DeiT-S | Enrichment | layer 11 | 0.0376 | 0.6162 [0.5875, 0.6450] | −0.3648 [−0.3944, −0.3369] | −2.7591 [−2.9426, −2.5981] | −2.3121 [−2.4782, −2.1663] |
| DeiT-S | ClickMe | layer 9 | 0.0856 | 0.3057 [0.2942, 0.3175] | −0.1196 [−0.1326, −0.1069] | −0.9793 [−1.0430, −0.9188] | −0.6731 [−0.7352, −0.6171] |
| DeiT-B | SRG | layer 10 | 0.0810 | 0.2904 [0.2337, 0.3447] | −0.6923 [−0.8482, −0.5469] | −1.9276 [−2.2297, −1.6472] | −1.5012 [−1.7704, −1.2659] |
| DeiT-B | Enrichment | layer 11 | 0.0875 | 0.3257 [0.3114, 0.3390] | −0.2700 [−0.2841, −0.2563] | −0.7177 [−0.7502, −0.6834] | −0.3958 [−0.4254, −0.3656] |
| DeiT-B | ClickMe | layer 10 | 0.1163 | 0.4481 [0.4359, 0.4607] | −0.3589 [−0.3802, −0.3387] | −0.7871 [−0.8196, −0.7548] | −0.3527 [−0.3808, −0.3252] |

Source: `experiments/2026-11-05_v5_analysis/a6_last_layer.csv`.

### 4.14 Sanity check (model randomisation)

**Procedure** (Adebayo et al. 2018): 1,000 ImageNet-S images (seed 0); from the classifier down to the patch embedding, one more stage is re-initialised at each level (cumulative; init seed 0); per image, Spearman between the map of the randomised model and the map of the trained model. Mean maps and probes are not re-estimated. **Registration.** Passes if the median Spearman after full randomisation (level "embed") < 0.5; predicted: R0, C1, C2, C3, R_all pass; centre as the failing example.

**DeiT-S** — median Spearman per level:

| Level | R0 | C1 | C2 | C3 | R_all (C3→C2→C1) | Centre |
| --- | --- | --- | --- | --- | --- | --- |
| head | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| block12 | 0.9818 | 0.9732 | 0.8473 | 0.9712 | 0.9064 | 1.0000 |
| block11 | 0.9298 | 0.8886 | 0.6687 | 0.9054 | 0.7911 | 1.0000 |
| block10 | 0.8292 | 0.6869 | 0.5213 | 0.7816 | 0.6107 | 1.0000 |
| block9 | 0.0352 | 0.1907 | 0.2954 | −0.0478 | 0.2513 | 1.0000 |
| block8 | 0.1406 | 0.2015 | 0.1541 | 0.0440 | 0.1820 | 1.0000 |
| block7 | 0.1272 | 0.2011 | 0.0534 | 0.0564 | 0.1375 | 1.0000 |
| block6 | 0.0844 | 0.1989 | −0.0030 | 0.0721 | 0.1051 | 1.0000 |
| block5 | 0.0639 | 0.1945 | −0.0150 | 0.0528 | 0.0949 | 1.0000 |
| block4 | 0.0388 | 0.1933 | −0.0094 | 0.0479 | 0.0852 | 1.0000 |
| block3 | −0.0035 | 0.1897 | −0.0474 | 0.0317 | 0.0812 | 1.0000 |
| block2 | 0.0469 | 0.1976 | −0.0610 | 0.0894 | 0.0778 | 1.0000 |
| block1 | 0.0745 | 0.1930 | −0.0745 | 0.0772 | 0.0764 | 1.0000 |
| embed | −0.0328 | 0.1867 | 0.0631 | −0.0641 | 0.0690 | 1.0000 |

**DeiT-B** — median Spearman per level:

| Level | R0 | C1 | C2 | C3 | R_all (C3→C2→C1) | Centre |
| --- | --- | --- | --- | --- | --- | --- |
| head | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| block12 | 0.8847 | 0.9022 | 0.8011 | 0.8611 | 0.8754 | 1.0000 |
| block11 | 0.7810 | 0.8205 | 0.6357 | 0.7584 | 0.6960 | 1.0000 |
| block10 | 0.6702 | 0.6801 | 0.4057 | 0.6630 | 0.4417 | 1.0000 |
| block9 | 0.1569 | 0.3782 | 0.0821 | 0.0624 | 0.1009 | 1.0000 |
| block8 | 0.1348 | 0.3743 | 0.0692 | 0.1318 | 0.0952 | 1.0000 |
| block7 | 0.0801 | 0.3640 | 0.0589 | 0.1204 | 0.0878 | 1.0000 |
| block6 | 0.0060 | 0.3560 | 0.0721 | 0.0661 | 0.0894 | 1.0000 |
| block5 | 0.0668 | 0.3626 | 0.0629 | 0.1075 | 0.0846 | 1.0000 |
| block4 | 0.1124 | 0.3700 | 0.0585 | 0.1224 | 0.0818 | 1.0000 |
| block3 | 0.0902 | 0.3703 | 0.0446 | 0.1192 | 0.0723 | 1.0000 |
| block2 | 0.0840 | 0.3688 | 0.0445 | 0.1194 | 0.0732 | 1.0000 |
| block1 | 0.0843 | 0.3635 | 0.0008 | 0.0789 | 0.0639 | 1.0000 |
| embed | 0.0552 | 0.3656 | 0.0036 | 0.0703 | 0.0780 | 1.0000 |

**DINO** — median Spearman per level:

| Level | R0 | C1 | C2 | C3 | R_all (C3→C2→C1) | Centre |
| --- | --- | --- | --- | --- | --- | --- |
| head | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| block12 | 0.9950 | 0.9834 | 0.8872 | 0.9942 | 0.8511 | 1.0000 |
| block11 | 0.9754 | 0.8621 | 0.7784 | 0.9721 | 0.6822 | 1.0000 |
| block10 | 0.8943 | 0.5430 | 0.4050 | 0.8681 | 0.3234 | 1.0000 |
| block9 | 0.0872 | 0.0991 | 0.0852 | 0.0908 | 0.0923 | 1.0000 |
| block8 | 0.0484 | 0.0940 | 0.0608 | 0.0569 | 0.0393 | 1.0000 |
| block7 | 0.1574 | 0.1025 | −0.0221 | 0.0378 | 0.0335 | 1.0000 |
| block6 | −0.0122 | 0.0967 | 0.0017 | −0.0076 | 0.0188 | 1.0000 |
| block5 | −0.2142 | 0.0815 | −0.1489 | −0.0538 | 0.0422 | 1.0000 |
| block4 | 0.1493 | 0.0964 | −0.1922 | 0.0918 | 0.0436 | 1.0000 |
| block3 | 0.2044 | 0.1032 | −0.0324 | 0.1132 | 0.0439 | 1.0000 |
| block2 | 0.1133 | 0.0994 | −0.1313 | 0.1482 | 0.0384 | 1.0000 |
| block1 | 0.0401 | 0.0943 | 0.1443 | 0.0865 | 0.0140 | 1.0000 |
| embed | 0.0706 | 0.0996 | −0.0016 | 0.1141 | 0.0252 | 1.0000 |

Source: `experiments/2026-11-05_v5_analysis/sanity.csv`. At level "embed": 15/15 reading cells pass (largest median 0.3656, DeiT-B C1); centre median 1.0000 in every model (fails, as designed). At level "head" every reading has median 1.0000; these readings do not depend on the class.

### 4.15 Negative controls

**Registration.** P4 (must hold; the queue stopped automatically if not, via `scripts/check_p4.py` after each model). P4a: DINO, C3 − R0, layers 9–12, three metrics. P4b: C2 − R0 at layers 1–4, enrichment and ClickMe.

| Control | Model | Metric | Images | Difference | Verdict |
| --- | --- | --- | --- | --- | --- |
| P4a | DINO | Enrichment | 12144 | −0.0035 [−0.0040, −0.0030] | comparable |
| P4b | DINO | Enrichment | 12144 | −0.0037 [−0.0043, −0.0030] | comparable |
| P4a | DINO | SRG | 2000 | 0.0003 [−0.0032, 0.0039] | comparable |
| P4a | DINO | ClickMe | 17408 | −0.0007 [−0.0010, −0.0005] | comparable |
| P4b | DINO | ClickMe | 17408 | −0.0017 [−0.0018, −0.0015] | comparable |
| P4b | DeiT-S | Enrichment | 12150 | −0.0010 [−0.0015, −0.0005] | comparable |
| P4b | DeiT-S | ClickMe | 17408 | −0.0002 [−0.0004, 0.0000] | comparable |
| P4b | DeiT-B | Enrichment | 12150 | 0.0000 [−0.0001, 0.0002] | comparable |
| P4b | DeiT-B | ClickMe | 17408 | 0.0000 [−0.0001, 0.0000] | comparable |

Source: `experiments/2026-11-05_p4_{dino,deit,deit_b}.csv`. The same rows are in `correct_tests.csv` (test = P4a / P4b). Largest |mean| = 0.0037. Several intervals exclude 0; those cells are "comparable" because |mean| < 0.02 (rule in 2.9). P4a 3/3 and P4b 6/6 comparable → P4 holds; the queue did not stop.

### 4.16 Reproduction check

**Procedure.** New R0 (v5 runs, L4) vs the earlier full-attention map (earlier runs: enrichment and SRG on an L4, ClickMe on an L4) on the same images; difference of means. **Registration.** |difference| < 0.01 on every metric; reported, no stop.

| Model | Metric | Images | New R0 | Earlier | Difference | < 0.01 |
| --- | --- | --- | --- | --- | --- | --- |
| DeiT-S | Enrichment | 12150 | 0.140094 | 0.140038 | 0.000056 | yes |
| DeiT-S | SRG | 2000 | 0.236022 | 0.235695 | 0.000326 | yes |
| DeiT-S | ClickMe | 17408 | 0.156526 | 0.156368 | 0.000158 | yes |
| DeiT-B | Enrichment | 12150 | 0.080476 | 0.080476 | 0.000000 | yes |
| DeiT-B | SRG | 2000 | 0.169366 | 0.169365 | 0.000001 | yes |
| DeiT-B | ClickMe | 17408 | 0.107121 | 0.107121 | 0.000000 | yes |
| DINO | Enrichment | 12144 | 0.199436 | 0.199449 | −0.000012 | yes |
| DINO | SRG | 2000 | 0.285009 | 0.285197 | −0.000188 | yes |
| DINO | ClickMe | 17408 | 0.207632 | 0.207775 | −0.000143 | yes |

Source: `experiments/2026-11-05_v5_analysis/reproduction.csv`. Largest |difference| = 0.000326.

### 4.17 Qualitative gallery

Figure 2(b) shows two ClickMe images (DeiT-S): image, human map, R0, C3, C2. Images chosen by a rule fixed in the earlier study: a seed-0 random draw of 24 images per dataset without picking; the two smallest indices are shown (ClickMe indices 47 and 287; the `index` arrays of the 24 gallery entries). The main figure uses layers 9–12 (as Table 1); a layer-12 version is an appendix figure. Each heat map is min–max normalised separately. Source: `experiments/2026-11-05_correct_ablation_clickme_deit/gallery_*.npz`; script `scripts/plot_paper_figures.py`.

### 4.18 Exploratory analyses (post hoc)

All analyses in this section were proposed after the formal results had been seen. For each, the definitions, descriptive criteria and branch rules were committed before the runs (Section 3.1). They change no registered verdict. Statistics and verdict rule as in 2.9. Mean maps, probes and blur widths are those of the formal runs. Runs: one L4, code `ebfd25f` (first three subsections) and `534e333` (the rest). In the re-evaluation runs, the recomputed R0 and C3 have the same means as the formal runs to 6 decimals in all 18 model × metric × reading cells (largest per-image absolute difference 0.0068; helper output `supp_check.csv`, computed with `scripts/analyze_v5.py` `per_image` on `experiments/2026-11-07_supp_eval_*` and `experiments/2026-11-05_correct_ablation_*`).

#### 4.18.1 Geometry of the probe's errors

**Question.** Are the position probe's errors mostly far from the true patch? **Setup.** Layers 6–12, every patch token of the 2,000 held-out test images; P = probe distribution over 196 positions, j = true position, Euclidean grid distance in patches. Per token: exact (argmax = j), top-5, expected distance Σ<sub>p</sub> P(p)‖p − j‖, probability within the 3×3 neighbourhood of j, probability at distance > 3, far share of errors = P(distance > 3) ÷ (1 − P(j)) (computed in float64 after the fix of Section 5.2; tokens with no error mass excluded). Averaged over the 196 tokens per image, bootstrap over images. The matched blur (σ of that layer) is evaluated with the same definitions; its exact and top-5 values are 1 by construction.

**Descriptive criterion (written before running).** Errors mainly far: at layers 9–12 the far share of errors > 0.5 (lower bound > 0.5) in all three models = supported; < 0.5 (upper bound < 0.5) in all three = not supported; otherwise partly supported.

**DeiT-S** (probe [interval] / matched blur):

| Layer | Exact | Top-5 | Expected distance | P(3×3) | P(> 3) | Far share of errors |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.8534 [0.8515, 0.8553] | 0.9988 [0.9986, 0.9989] | 0.2974 [0.2945, 0.3005] / 0.6128 | 0.9867 [0.9863, 0.9870] / 0.9883 | 0.0053 [0.0052, 0.0055] / 0.0000 | 0.0306 [0.0300, 0.0312] / 0.0000 |
| 7 | 0.6504 [0.6477, 0.6529] | 0.9831 [0.9823, 0.9838] | 0.6295 [0.6250, 0.6343] / 1.0328 | 0.9397 [0.9386, 0.9408] / 0.8622 | 0.0124 [0.0121, 0.0128] / 0.0033 | 0.0289 [0.0283, 0.0296] / 0.0041 |
| 8 | 0.4390 [0.4364, 0.4414] | 0.8994 [0.8972, 0.9017] | 1.0342 [1.0278, 1.0410] / 1.4023 | 0.8330 [0.8309, 0.8350] / 0.6752 | 0.0326 [0.0318, 0.0335] / 0.0354 | 0.0436 [0.0426, 0.0446] / 0.0406 |
| 9 | 0.3102 [0.3079, 0.3124] | 0.7616 [0.7584, 0.7646] | 1.5169 [1.5061, 1.5277] / 1.9502 | 0.6866 [0.6839, 0.6893] / 0.4521 | 0.0926 [0.0908, 0.0945] / 0.1630 | 0.1062 [0.1042, 0.1082] / 0.1754 |
| 10 | 0.2315 [0.2294, 0.2335] | 0.6311 [0.6276, 0.6344] | 2.0751 [2.0578, 2.0910] / 2.6024 | 0.5533 [0.5504, 0.5563] / 0.2948 | 0.1831 [0.1801, 0.1858] / 0.3478 | 0.2008 [0.1977, 0.2036] / 0.3630 |
| 11 | 0.1937 [0.1918, 0.1956] | 0.5558 [0.5522, 0.5593] | 2.4693 [2.4512, 2.4875] / 3.0365 | 0.4780 [0.4754, 0.4809] / 0.2305 | 0.2485 [0.2454, 0.2513] / 0.4531 | 0.2684 [0.2653, 0.2713] / 0.4680 |
| 12 | 0.1812 [0.1794, 0.1830] | 0.5311 [0.5276, 0.5347] | 2.6287 [2.6094, 2.6481] / 3.2095 | 0.4520 [0.4494, 0.4548] / 0.2106 | 0.2737 [0.2706, 0.2767] / 0.4896 | 0.2944 [0.2912, 0.2974] / 0.5042 |

**DeiT-B** (probe [interval] / matched blur):

| Layer | Exact | Top-5 | Expected distance | P(3×3) | P(> 3) | Far share of errors |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.9947 [0.9944, 0.9950] | 1.0000 [1.0000, 1.0000] | 0.0137 [0.0133, 0.0140] / 0.0281 | 0.9996 [0.9995, 0.9996] / 1.0000 | 0.0003 [0.0003, 0.0004] / 0.0000 | 0.1254 [0.1241, 0.1268] / 0.0000 |
| 7 | 0.9683 [0.9677, 0.9690] | 0.9999 [0.9998, 0.9999] | 0.0568 [0.0558, 0.0578] / 0.1250 | 0.9974 [0.9973, 0.9975] / 1.0000 | 0.0015 [0.0014, 0.0015] / 0.0000 | 0.0570 [0.0560, 0.0580] / 0.0000 |
| 8 | 0.9184 [0.9171, 0.9197] | 0.9992 [0.9991, 0.9993] | 0.1490 [0.1468, 0.1513] / 0.4049 | 0.9911 [0.9908, 0.9914] / 0.9987 | 0.0052 [0.0050, 0.0054] / 0.0000 | 0.0570 [0.0561, 0.0579] / 0.0000 |
| 9 | 0.8365 [0.8345, 0.8383] | 0.9963 [0.9961, 0.9966] | 0.2937 [0.2903, 0.2976] / 0.7256 | 0.9807 [0.9801, 0.9811] / 0.9713 | 0.0100 [0.0097, 0.0103] / 0.0001 | 0.0498 [0.0489, 0.0506] / 0.0001 |
| 10 | 0.6684 [0.6659, 0.6708] | 0.9732 [0.9722, 0.9741] | 0.5725 [0.5672, 0.5781] / 1.0605 | 0.9430 [0.9418, 0.9441] / 0.8489 | 0.0171 [0.0166, 0.0176] / 0.0042 | 0.0426 [0.0418, 0.0434] / 0.0053 |
| 11 | 0.4916 [0.4886, 0.4941] | 0.8915 [0.8890, 0.8936] | 0.9806 [0.9727, 0.9898] / 1.4939 | 0.8454 [0.8431, 0.8475] / 0.6313 | 0.0424 [0.0413, 0.0435] / 0.0510 | 0.0633 [0.0620, 0.0646] / 0.0576 |
| 12 | 0.3177 [0.3148, 0.3203] | 0.7165 [0.7124, 0.7201] | 1.6758 [1.6588, 1.6938] / 2.2243 | 0.6573 [0.6535, 0.6606] / 0.3745 | 0.1304 [0.1277, 0.1333] / 0.2422 | 0.1524 [0.1496, 0.1554] / 0.2566 |

**DINO** (probe [interval] / matched blur):

| Layer | Exact | Top-5 | Expected distance | P(3×3) | P(> 3) | Far share of errors |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.9890 [0.9885, 0.9896] | 1.0000 [0.9999, 1.0000] | 0.0573 [0.0558, 0.0588] / 0.1496 | 0.9984 [0.9983, 0.9986] / 1.0000 | 0.0015 [0.0014, 0.0016] / 0.0000 | 0.0354 [0.0344, 0.0363] / 0.0000 |
| 7 | 0.9587 [0.9573, 0.9599] | 1.0000 [0.9999, 1.0000] | 0.1455 [0.1428, 0.1485] / 0.3666 | 0.9958 [0.9956, 0.9961] / 0.9992 | 0.0037 [0.0034, 0.0039] / 0.0000 | 0.0452 [0.0444, 0.0460] / 0.0000 |
| 8 | 0.8562 [0.8536, 0.8586] | 0.9990 [0.9988, 0.9992] | 0.3157 [0.3114, 0.3203] / 0.6427 | 0.9901 [0.9896, 0.9906] / 0.9849 | 0.0062 [0.0058, 0.0066] / 0.0000 | 0.0374 [0.0365, 0.0382] / 0.0000 |
| 9 | 0.6738 [0.6707, 0.6768] | 0.9892 [0.9884, 0.9898] | 0.5605 [0.5552, 0.5665] / 0.8749 | 0.9674 [0.9664, 0.9683] / 0.9291 | 0.0075 [0.0071, 0.0080] / 0.0005 | 0.0211 [0.0204, 0.0218] / 0.0008 |
| 10 | 0.5048 [0.5019, 0.5077] | 0.9500 [0.9485, 0.9515] | 0.8104 [0.8050, 0.8159] / 1.0568 | 0.9124 [0.9110, 0.9139] / 0.8507 | 0.0105 [0.0101, 0.0110] / 0.0041 | 0.0174 [0.0169, 0.0180] / 0.0051 |
| 11 | 0.3923 [0.3896, 0.3950] | 0.8810 [0.8787, 0.8833] | 1.0390 [1.0332, 1.0446] / 1.2652 | 0.8372 [0.8353, 0.8392] / 0.7445 | 0.0233 [0.0227, 0.0240] / 0.0179 | 0.0290 [0.0283, 0.0298] / 0.0212 |
| 12 | 0.3575 [0.3549, 0.3600] | 0.8510 [0.8486, 0.8534] | 1.1281 [1.1223, 1.1339] / 1.3554 | 0.8054 [0.8035, 0.8074] / 0.6985 | 0.0310 [0.0302, 0.0317] / 0.0287 | 0.0371 [0.0363, 0.0379] / 0.0332 |

Source: `experiments/2026-11-07_explore_c2_analysis/errors.csv`. The far-share column comes from the rerun `experiments/2026-11-07_explore_c2_probefix_*`; the other columns from `experiments/2026-11-07_explore_c2_*`. Far share of errors at layers 9–12: 0.0174 to 0.2944 (largest upper bound 0.2974). Outcome recorded in `hypotheses.csv`: "not supported". Probe − blur differences with intervals are in the same file (source = "probe - blur").

#### 4.18.2 The probe's mean prediction

**Question.** Does the probe carry an image-independent position preference? **Setup.** Per layer, the probe distributions of all held-out test tokens are averaged into one distribution over 196 positions (uniform = 1/196 each). Reported: maximum ÷ (1/196); KL divergence from uniform; probability on the central 6×6 block (area share 36/196 = 0.1837); Pearson correlation with the model's R0 mean map of that layer; Pearson with closeness to the centre (negative distance). Intervals: images resampled, distribution recomputed.

**Descriptive criterion.** At layers 9–12, all three models satisfy both: maximum ≥ 2 × 1/196, and Pearson with the mean map ≥ 0.5 with lower bound > 0 = supported; none = not supported; otherwise partly supported. Side evidence: correlation of (C2 − R0) with the mean map (4.18.3).

| Model | Layer | Max ÷ (1/196) | KL from uniform | P(central 6×6) | Pearson with mean map | Pearson with closeness to centre |
| --- | --- | --- | --- | --- | --- | --- |
| DeiT-S | 6 | 1.1250 [1.1165, 1.1450] | 0.0011 | 0.1838 | −0.0056 [−0.0277, 0.0138] | −0.0095 [−0.0250, 0.0049] |
| DeiT-S | 7 | 1.1651 [1.1492, 1.1950] | 0.0019 | 0.1828 | −0.0573 [−0.0855, −0.0263] | −0.0382 [−0.0620, −0.0136] |
| DeiT-S | 8 | 1.2068 [1.1734, 1.2429] | 0.0026 | 0.1824 | −0.0579 [−0.0943, −0.0186] | −0.0642 [−0.0977, −0.0238] |
| DeiT-S | 9 | 1.2223 [1.1935, 1.2583] | 0.0026 | 0.1824 | −0.0467 [−0.0894, −0.0019] | −0.0851 [−0.1338, −0.0270] |
| DeiT-S | 10 | 1.1964 [1.1873, 1.2383] | 0.0026 | 0.1827 | −0.0807 [−0.1281, −0.0309] | −0.1236 [−0.1877, −0.0502] |
| DeiT-S | 11 | 1.2081 [1.1782, 1.2480] | 0.0024 | 0.1831 | −0.1669 [−0.2145, −0.1102] | −0.0749 [−0.1523, 0.0032] |
| DeiT-S | 12 | 1.2179 [1.1730, 1.2643] | 0.0024 | 0.1830 | −0.1795 [−0.2183, −0.1291] | −0.0776 [−0.1603, 0.0031] |
| DeiT-B | 6 | 1.0159 [1.0127, 1.0224] | 0.0000 | 0.1839 | 0.1029 [0.0697, 0.1108] | 0.2193 [0.1645, 0.2181] |
| DeiT-B | 7 | 1.0439 [1.0383, 1.0574] | 0.0001 | 0.1842 | 0.0198 [−0.0083, 0.0432] | 0.2279 [0.1884, 0.2321] |
| DeiT-B | 8 | 1.0558 [1.0541, 1.0756] | 0.0003 | 0.1833 | 0.0622 [0.0276, 0.0888] | −0.0064 [−0.0233, 0.0097] |
| DeiT-B | 9 | 1.1362 [1.1141, 1.1578] | 0.0006 | 0.1827 | 0.0865 [0.0450, 0.1204] | −0.1060 [−0.1211, −0.0836] |
| DeiT-B | 10 | 1.1541 [1.1317, 1.1923] | 0.0014 | 0.1827 | 0.0518 [0.0068, 0.0929] | −0.1345 [−0.1551, −0.1052] |
| DeiT-B | 11 | 1.1714 [1.1564, 1.2175] | 0.0024 | 0.1821 | 0.0044 [−0.0408, 0.0480] | −0.1730 [−0.2075, −0.1286] |
| DeiT-B | 12 | 1.1902 [1.1869, 1.2687] | 0.0038 | 0.1788 | 0.0519 [−0.0060, 0.1058] | −0.2959 [−0.3438, −0.2229] |
| DINO | 6 | 1.0557 [1.0500, 1.0629] | 0.0001 | 0.1840 | 0.0678 [0.0466, 0.0879] | 0.0493 [0.0324, 0.0648] |
| DINO | 7 | 1.1068 [1.0963, 1.1198] | 0.0005 | 0.1839 | 0.0054 [−0.0133, 0.0242] | 0.0119 [−0.0034, 0.0278] |
| DINO | 8 | 1.1581 [1.1443, 1.1787] | 0.0016 | 0.1837 | −0.0011 [−0.0233, 0.0200] | −0.0146 [−0.0326, 0.0052] |
| DINO | 9 | 1.2452 [1.2215, 1.2713] | 0.0032 | 0.1844 | −0.0123 [−0.0376, 0.0111] | −0.0247 [−0.0464, −0.0015] |
| DINO | 10 | 1.3086 [1.2801, 1.3391] | 0.0034 | 0.1837 | −0.0266 [−0.0593, 0.0056] | −0.0514 [−0.0805, −0.0210] |
| DINO | 11 | 1.3343 [1.3010, 1.3673] | 0.0039 | 0.1839 | −0.0369 [−0.0735, 0.0002] | −0.0660 [−0.0995, −0.0283] |
| DINO | 12 | 1.3174 [1.2828, 1.3531] | 0.0038 | 0.1832 | −0.0347 [−0.0731, 0.0029] | −0.0771 [−0.1150, −0.0387] |

Source: `experiments/2026-11-07_explore_c2_analysis/mean_prediction.csv`. Spearman versions are in the same file. These statistics are not means over images, so a point estimate can lie outside its percentile interval (e.g. DeiT-B, layer 6). Outcome in `hypotheses.csv`: "not supported" (0/12 model-layer cells meet both conditions).

#### 4.18.3 Where C2 moves the score on ImageNet-S

**Setup.** Layers 9–12 summed; all 12,419 images (C2, R0 and matched blur as in Table 1). Object share = score on valid object patches ÷ score on all valid patches. "Moved to background" = object share of R0 − object share after the change. Correlations: per image, (map − R0) against the summed layer 9–12 mean map and against closeness to the centre, averaged over images.

| Quantity | DeiT-S C2 | DeiT-S blur | DeiT-B C2 | DeiT-B blur | DINO C2 | DINO blur |
| --- | --- | --- | --- | --- | --- | --- |
| Object share, R0 | 0.4534 [0.4491, 0.4576] | 0.4534 [0.4491, 0.4576] | 0.3600 [0.3558, 0.3642] | 0.3600 [0.3558, 0.3642] | 0.5114 [0.5066, 0.5162] | 0.5114 [0.5066, 0.5162] |
| Object share, after | 0.4129 [0.4081, 0.4175] | 0.4149 [0.4101, 0.4195] | 0.3604 [0.3560, 0.3648] | 0.3625 [0.3581, 0.3669] | 0.4710 [0.4661, 0.4760] | 0.4819 [0.4770, 0.4869] |
| Moved to background | 0.0406 [0.0394, 0.0418] | 0.0386 [0.0375, 0.0396] | −0.0004 [−0.0010, 0.0002] | −0.0025 [−0.0030, −0.0019] | 0.0403 [0.0394, 0.0412] | 0.0295 [0.0289, 0.0300] |
| Pearson(change, mean map) | −0.0380 [−0.0408, −0.0353] | −0.1770 [−0.1804, −0.1737] | −0.0367 [−0.0391, −0.0345] | −0.2467 [−0.2498, −0.2436] | −0.0043 [−0.0062, −0.0025] | −0.0800 [−0.0808, −0.0791] |
| Spearman(change, mean map) | 0.0592 [0.0574, 0.0612] | 0.0128 [0.0108, 0.0147] | 0.0980 [0.0964, 0.0997] | −0.0489 [−0.0507, −0.0471] | −0.0122 [−0.0140, −0.0104] | −0.0340 [−0.0350, −0.0328] |
| Pearson(change, closeness) | −0.0025 [−0.0042, −0.0008] | 0.0576 [0.0565, 0.0587] | −0.0029 [−0.0037, −0.0021] | 0.0304 [0.0301, 0.0308] | −0.0068 [−0.0086, −0.0051] | −0.0126 [−0.0136, −0.0116] |
| Spearman(change, closeness) | −0.0308 [−0.0329, −0.0286] | 0.1069 [0.1050, 0.1090] | −0.0739 [−0.0753, −0.0723] | −0.0439 [−0.0459, −0.0416] | −0.0154 [−0.0171, −0.0135] | 0.0653 [0.0642, 0.0665] |

Source: `experiments/2026-11-07_explore_c2_analysis/shift.csv`.

Change in the share of the total score per distance bin (⌊distance from centre⌋ in patches), C2 − R0 and blur − R0:

| Bin | DeiT-S R0 share | DeiT-S C2 Δ | DeiT-S blur Δ | DeiT-B R0 share | DeiT-B C2 Δ | DeiT-B blur Δ | DINO R0 share | DINO C2 Δ | DINO blur Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0270 | −0.0023 [−0.0026, −0.0019] | −0.0022 [−0.0024, −0.0019] | 0.0209 | −0.0014 [−0.0016, −0.0012] | −0.0009 [−0.0011, −0.0008] | 0.0334 | −0.0008 [−0.0010, −0.0006] | −0.0019 [−0.0020, −0.0018] |
| 1 | 0.0495 | 0.0010 [0.0006, 0.0013] | −0.0003 [−0.0006, 0.0000] | 0.0389 | 0.0011 [0.0008, 0.0013] | 0.0001 [−0.0001, 0.0003] | 0.0626 | −0.0001 [−0.0003, 0.0002] | −0.0026 [−0.0027, −0.0025] |
| 2 | 0.1139 | −0.0011 [−0.0017, −0.0006] | 0.0070 [0.0066, 0.0074] | 0.0901 | −0.0022 [−0.0026, −0.0018] | 0.0048 [0.0045, 0.0051] | 0.1400 | −0.0008 [−0.0012, −0.0004] | −0.0039 [−0.0041, −0.0037] |
| 3 | 0.1052 | −0.0002 [−0.0007, 0.0003] | 0.0124 [0.0120, 0.0128] | 0.0879 | −0.0004 [−0.0008, 0.0001] | 0.0076 [0.0072, 0.0079] | 0.1222 | 0.0017 [0.0014, 0.0020] | −0.0015 [−0.0017, −0.0014] |
| 4 | 0.1375 | −0.0025 [−0.0031, −0.0020] | 0.0208 [0.0203, 0.0213] | 0.1199 | −0.0022 [−0.0028, −0.0016] | 0.0202 [0.0197, 0.0207] | 0.1482 | −0.0023 [−0.0027, −0.0020] | 0.0031 [0.0029, 0.0032] |
| 5 | 0.1701 | 0.0009 [0.0001, 0.0016] | −0.0022 [−0.0029, −0.0015] | 0.1774 | −0.0021 [−0.0029, −0.0013] | −0.0072 [−0.0080, −0.0062] | 0.1477 | 0.0007 [0.0004, 0.0011] | 0.0113 [0.0111, 0.0114] |
| 6 | 0.1936 | 0.0089 [0.0082, 0.0096] | 0.0049 [0.0042, 0.0055] | 0.2088 | 0.0073 [0.0065, 0.0080] | 0.0108 [0.0100, 0.0116] | 0.1901 | −0.0014 [−0.0017, −0.0011] | −0.0021 [−0.0022, −0.0019] |
| 7 | 0.1557 | −0.0064 [−0.0071, −0.0056] | −0.0343 [−0.0350, −0.0337] | 0.2065 | −0.0014 [−0.0020, −0.0008] | −0.0438 [−0.0447, −0.0430] | 0.1078 | 0.0017 [0.0014, 0.0020] | 0.0030 [0.0029, 0.0031] |
| 8 | 0.0339 | 0.0012 [0.0009, 0.0016] | −0.0046 [−0.0049, −0.0042] | 0.0360 | 0.0002 [−0.0001, 0.0005] | 0.0057 [0.0053, 0.0061] | 0.0306 | 0.0006 [0.0005, 0.0008] | −0.0008 [−0.0008, −0.0007] |
| 9 | 0.0136 | 0.0006 [0.0004, 0.0007] | −0.0015 [−0.0016, −0.0014] | 0.0136 | 0.0011 [0.0011, 0.0012] | 0.0027 [0.0026, 0.0028] | 0.0174 | 0.0005 [0.0005, 0.0006] | −0.0046 [−0.0046, −0.0045] |

Source: `experiments/2026-11-07_explore_c2_analysis/shift_bins.csv`.

#### 4.18.4 ClickMe and smoothing (compilation of existing numbers)

**Question.** Does the ClickMe metric change differently from enrichment under smoothing-like changes? No new runs; differences vs R0 from 4.10 and 4.12.

| Change | Model | ClickMe | Same-ring enrichment |
| --- | --- | --- | --- |
| Matched blur − R0 | DeiT-S | −0.0045 [−0.0077, −0.0014] comparable | −0.0451 [−0.0475, −0.0426] lower |
| C2 − R0 | DeiT-S | −0.0726 [−0.0762, −0.0693] lower | −0.0987 [−0.1008, −0.0968] lower |
| Other order − R0 | DeiT-S | 0.0646 [0.0616, 0.0676] higher | −0.0323 [−0.0341, −0.0305] lower |
| Matched blur − R0 | DeiT-B | −0.0392 [−0.0422, −0.0360] lower | −0.0886 [−0.0906, −0.0867] lower |
| C2 − R0 | DeiT-B | −0.0570 [−0.0594, −0.0547] lower | −0.0536 [−0.0552, −0.0522] lower |
| Other order − R0 | DeiT-B | 0.0923 [0.0899, 0.0948] higher | 0.0421 [0.0403, 0.0441] higher |
| Matched blur − R0 | DINO | 0.0184 [0.0175, 0.0194] comparable | −0.0102 [−0.0117, −0.0087] comparable |
| C2 − R0 | DINO | −0.0644 [−0.0667, −0.0623] lower | −0.1150 [−0.1172, −0.1127] lower |
| Other order − R0 | DINO | 0.0550 [0.0515, 0.0586] higher | −0.0439 [−0.0457, −0.0422] lower |

Source: `experiments/2026-11-05_v5_analysis/correct_tests.csv`; `experiments/2026-11-05_v5_analysis/a7_order.csv`. Human maps are blurred (Section 1.3: σ = 10 pixels at 64×64, upsampled to 224×224).

#### 4.18.5 Pairwise combinations

**Definitions.** Implemented like R_all, in the fixed order C3 → C2 → C1, skipping the absent correction: C1+C3 = Σ<sub>ℓ</sub>(c<sub>ℓ</sub> − c̄<sub>ℓ</sub>) with c̄ the held-out mean map of C3; C2+C3 = Σ<sub>ℓ</sub> relocate(c<sub>ℓ</sub>, P<sub>ℓ</sub>); C1+C2 = Σ<sub>ℓ</sub>(relocate(s<sub>ℓ</sub>, P<sub>ℓ</sub>) − held-out mean of relocated R0). SRG only for C1+C3 (with R0 and C3 recomputed in the same run).

**Branch rule (written before running).** Branch A: C1+C3 − C3 "higher" in at least 2 of the 4 supervised cells (DeiT-S, DeiT-B × enrichment, ClickMe) and "lower" in none → report a stand-in "C3 with centred Gaussian subtraction" and (C1+C3) − that stand-in. Branch B otherwise: no further analysis; the paper keeps recommending C3 alone. The stand-in was computed in the same run but is reported only under branch A.

| Comparison | Model | SRG | Same-ring enrichment | ClickMe |
| --- | --- | --- | --- | --- |
| C1+C3 − R0 | DeiT-S | −0.0192 [−0.0257, −0.0127] comparable | 0.0347 [0.0333, 0.0360] higher | 0.0522 [0.0503, 0.0541] higher |
| C1+C3 − R0 | DeiT-B | 0.0008 [−0.0046, 0.0063] comparable | 0.0653 [0.0632, 0.0674] higher | 0.0744 [0.0724, 0.0764] higher |
| C1+C3 − R0 | DINO | −0.0494 [−0.0576, −0.0419] comparable | −0.0018 [−0.0027, −0.0008] comparable | 0.0434 [0.0397, 0.0468] higher |
| C2+C3 − R0 | DeiT-S | — | −0.0823 [−0.0844, −0.0804] lower | −0.0446 [−0.0482, −0.0414] lower |
| C2+C3 − R0 | DeiT-B | — | −0.0021 [−0.0039, −0.0004] comparable | 0.0214 [0.0193, 0.0236] higher |
| C2+C3 − R0 | DINO | — | −0.1132 [−0.1155, −0.1110] lower | −0.0606 [−0.0628, −0.0585] lower |
| C1+C2 − R0 | DeiT-S | — | −0.0964 [−0.0986, −0.0945] lower | −0.0663 [−0.0702, −0.0625] lower |
| C1+C2 − R0 | DeiT-B | — | −0.0512 [−0.0528, −0.0497] lower | −0.0660 [−0.0683, −0.0635] lower |
| C1+C2 − R0 | DINO | — | −0.1140 [−0.1163, −0.1116] lower | −0.0372 [−0.0410, −0.0337] lower |
| C1+C3 − C3 | DeiT-S | −0.0366 [−0.0430, −0.0304] comparable | −0.0029 [−0.0034, −0.0023] comparable | 0.0189 [0.0169, 0.0209] comparable |
| C1+C3 − C3 | DeiT-B | −0.0279 [−0.0335, −0.0225] comparable | −0.0001 [−0.0007, 0.0004] comparable | 0.0168 [0.0149, 0.0186] comparable |
| C1+C3 − C3 | DINO | −0.0497 [−0.0576, −0.0419] comparable | 0.0017 [0.0009, 0.0026] comparable | 0.0441 [0.0404, 0.0475] higher |
| C2+C3 − C3 | DeiT-S | — | −0.1199 [−0.1223, −0.1175] lower | −0.0779 [−0.0816, −0.0747] lower |
| C2+C3 − C3 | DeiT-B | — | −0.0675 [−0.0695, −0.0656] lower | −0.0361 [−0.0383, −0.0341] lower |
| C2+C3 − C3 | DINO | — | −0.1097 [−0.1120, −0.1076] lower | −0.0599 [−0.0620, −0.0578] lower |

Source: `experiments/2026-11-07_v5_supp_analysis/x1.csv`. C1+C3 − C3 in the supervised cells: 0 higher, 0 lower → branch B (`branches.csv`). The rows "C1C3 − (C3 − centre)" in `x1.csv` are the stand-in computed for branch A and are not reported here.

#### 4.18.6 C3 mechanism checks

**Definitions.** C3 with ‖v<sub>j</sub>‖: weight ‖v<sub>j</sub>‖ instead of ‖W<sub>O</sub><sup>h</sup>v<sub>j</sub>‖. C3 with shuffled weights: within each image, layer and head, the 196 patch weights ‖W<sub>O</sub>v<sub>j</sub>‖ are randomly permuted (CLS excluded), seeds 0–4; per image the metric is averaged over the 5 seeds before the bootstrap. C3 with head-mean weights: every token gets the mean of the 196 patch weights of its image, layer and head. Enrichment and ClickMe; three models.

**Reading rule (written before running).** "Expected": C3 − shuffled and C3 − head-mean each "higher" on at least 1 metric in both supervised models and "lower" nowhere. "Head weighting": C3 − head-mean "comparable" in all 4 supervised cells (then add a layer × head table). "Other": neither. C3 − C3 with ‖v‖: "higher" = W<sub>O</sub> matters; "comparable" = value length suffices.

| Comparison | Model | Same-ring enrichment | ClickMe |
| --- | --- | --- | --- |
| C3 − C3 with shuffled weights | DeiT-S | 0.0385 [0.0372, 0.0398] higher | 0.0355 [0.0346, 0.0363] higher |
| C3 − C3 with shuffled weights | DeiT-B | 0.0734 [0.0712, 0.0756] higher | 0.0666 [0.0651, 0.0680] higher |
| C3 − C3 with shuffled weights | DINO | −0.0039 [−0.0044, −0.0034] comparable | 0.0001 [−0.0001, 0.0003] comparable |
| C3 − C3 with head-mean weights | DeiT-S | 0.0383 [0.0369, 0.0397] higher | 0.0352 [0.0344, 0.0361] higher |
| C3 − C3 with head-mean weights | DeiT-B | 0.0733 [0.0711, 0.0755] higher | 0.0662 [0.0648, 0.0677] higher |
| C3 − C3 with head-mean weights | DINO | −0.0042 [−0.0047, −0.0036] comparable | −0.0001 [−0.0003, 0.0001] comparable |
| C3 − C3 with ‖v‖ | DeiT-S | −0.0100 [−0.0108, −0.0091] comparable | −0.0104 [−0.0106, −0.0101] comparable |
| C3 − C3 with ‖v‖ | DeiT-B | −0.0067 [−0.0071, −0.0062] comparable | −0.0061 [−0.0063, −0.0059] comparable |
| C3 − C3 with ‖v‖ | DINO | −0.0004 [−0.0006, −0.0001] comparable | −0.0006 [−0.0006, −0.0005] comparable |

Source: `experiments/2026-11-07_v5_supp_analysis/x2.csv`; `experiments/2026-11-07_v5_supp_analysis/x2_seeds.csv`. Range of the 5 per-seed means of the shuffled reading, largest over cells: 0.0003. Outcome: "expected"; C3 − C3 with ‖v‖ "comparable" in 4/4 supervised cells (`branches.csv`). Mean weight per layer and head is stored in `x2_head_weights.csv`; under the rule the layer × head table is not reported.

#### 4.18.7 C3 dose–response and per-layer effects

**Definitions.** Dose of an image = share of its R0 (layers 9–12) on high-norm tokens, judged per layer. Across images, Spearman correlation between dose and the image's improvement (C3 − R0, and C3 − zero high-norm); interval by bootstrap over images. Per layer (6–12 and 9–12 summed): C3 − R0 and zero high-norm − R0, descriptive.

**Branch rule.** Branch A: Spearman(dose, C3 − R0) on ClickMe positive with interval excluding 0 in both supervised models → dose–response holds. Branch B otherwise: report, no further analysis.

| Model | Metric | Improvement | Images | Mean dose | Spearman(dose, improvement) |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | ClickMe | C3 − R0 | 17408 | 0.0823 | 0.0453 [0.0316, 0.0603] |
| DeiT-S | ClickMe | C3 − zero high-norm | 17408 | 0.0823 | −0.0651 [−0.0790, −0.0513] |
| DeiT-S | Enrichment | C3 − R0 | 12150 | 0.0784 | 0.0748 [0.0573, 0.0921] |
| DeiT-S | Enrichment | C3 − zero high-norm | 12150 | 0.0784 | −0.0349 [−0.0518, −0.0177] |
| DeiT-B | ClickMe | C3 − R0 | 17408 | 0.2929 | 0.1578 [0.1430, 0.1726] |
| DeiT-B | ClickMe | C3 − zero high-norm | 17408 | 0.2929 | −0.0282 [−0.0440, −0.0137] |
| DeiT-B | Enrichment | C3 − R0 | 12150 | 0.2891 | 0.1316 [0.1127, 0.1497] |
| DeiT-B | Enrichment | C3 − zero high-norm | 12150 | 0.2891 | −0.0310 [−0.0487, −0.0133] |

Source: `experiments/2026-11-07_v5_supp_analysis/x3_dose.csv`. Outcome: branch A.

| Model | Layer | C3 − R0, enrichment | C3 − R0, ClickMe | Zero high-norm − R0, enrichment | Zero high-norm − R0, ClickMe |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | 6 | 0.0000 [−0.0007, 0.0007] comparable | 0.0052 [0.0047, 0.0058] comparable | 0.0089 [0.0082, 0.0096] comparable | 0.0117 [0.0110, 0.0123] comparable |
| DeiT-S | 7 | 0.0091 [0.0081, 0.0100] comparable | 0.0159 [0.0152, 0.0165] comparable | 0.0188 [0.0178, 0.0198] comparable | 0.0270 [0.0260, 0.0281] higher |
| DeiT-S | 8 | 0.0202 [0.0190, 0.0213] higher | 0.0259 [0.0251, 0.0268] higher | 0.0221 [0.0211, 0.0232] higher | 0.0271 [0.0260, 0.0282] higher |
| DeiT-S | 9 | 0.0306 [0.0293, 0.0319] higher | 0.0331 [0.0320, 0.0342] higher | 0.0217 [0.0207, 0.0227] higher | 0.0238 [0.0228, 0.0248] higher |
| DeiT-S | 10 | 0.0378 [0.0363, 0.0393] higher | 0.0347 [0.0338, 0.0356] higher | 0.0164 [0.0156, 0.0174] comparable | 0.0184 [0.0176, 0.0192] comparable |
| DeiT-S | 11 | 0.0459 [0.0442, 0.0476] higher | 0.0437 [0.0426, 0.0447] higher | 0.0105 [0.0098, 0.0112] comparable | 0.0093 [0.0087, 0.0098] comparable |
| DeiT-S | 12 | 0.0231 [0.0220, 0.0242] higher | 0.0262 [0.0255, 0.0268] higher | 0.0061 [0.0057, 0.0065] comparable | 0.0067 [0.0062, 0.0070] comparable |
| DeiT-S | 9–12 summed | 0.0376 [0.0363, 0.0389] higher | 0.0333 [0.0325, 0.0342] higher | 0.0120 [0.0110, 0.0129] comparable | 0.0113 [0.0108, 0.0118] comparable |
| DeiT-B | 6 | −0.0033 [−0.0040, −0.0026] comparable | −0.0001 [−0.0007, 0.0005] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |
| DeiT-B | 7 | 0.0112 [0.0101, 0.0123] comparable | 0.0173 [0.0166, 0.0181] comparable | 0.0113 [0.0105, 0.0121] comparable | 0.0164 [0.0156, 0.0172] comparable |
| DeiT-B | 8 | 0.0082 [0.0073, 0.0091] comparable | 0.0237 [0.0229, 0.0245] higher | 0.0303 [0.0289, 0.0318] higher | 0.0442 [0.0426, 0.0458] higher |
| DeiT-B | 9 | 0.0343 [0.0327, 0.0359] higher | 0.0466 [0.0452, 0.0479] higher | 0.0415 [0.0397, 0.0431] higher | 0.0570 [0.0551, 0.0589] higher |
| DeiT-B | 10 | 0.0494 [0.0474, 0.0515] higher | 0.0499 [0.0485, 0.0513] higher | 0.0554 [0.0533, 0.0574] higher | 0.0578 [0.0559, 0.0596] higher |
| DeiT-B | 11 | 0.0503 [0.0483, 0.0523] higher | 0.0477 [0.0466, 0.0488] higher | 0.0552 [0.0531, 0.0574] higher | 0.0563 [0.0545, 0.0580] higher |
| DeiT-B | 12 | 0.0285 [0.0273, 0.0297] higher | 0.0521 [0.0510, 0.0533] higher | 0.0142 [0.0136, 0.0147] comparable | 0.0207 [0.0197, 0.0216] higher |
| DeiT-B | 9–12 summed | 0.0654 [0.0634, 0.0675] higher | 0.0576 [0.0563, 0.0588] higher | 0.0401 [0.0383, 0.0421] higher | 0.0405 [0.0391, 0.0419] higher |
| DINO | 6 | −0.0083 [−0.0091, −0.0074] comparable | 0.0091 [0.0081, 0.0099] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |
| DINO | 7 | 0.0052 [0.0042, 0.0062] comparable | 0.0130 [0.0122, 0.0137] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |
| DINO | 8 | 0.0058 [0.0050, 0.0065] comparable | 0.0078 [0.0073, 0.0084] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |
| DINO | 9 | −0.0044 [−0.0051, −0.0037] comparable | −0.0016 [−0.0021, −0.0011] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |
| DINO | 10 | −0.0026 [−0.0033, −0.0020] comparable | −0.0011 [−0.0014, −0.0008] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |
| DINO | 11 | −0.0044 [−0.0049, −0.0038] comparable | −0.0011 [−0.0014, −0.0009] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |
| DINO | 12 | −0.0030 [−0.0036, −0.0023] comparable | 0.0013 [0.0010, 0.0016] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |
| DINO | 9–12 summed | −0.0035 [−0.0040, −0.0030] comparable | −0.0007 [−0.0010, −0.0005] comparable | 0.0000 [0.0000, 0.0000] comparable | 0.0000 [0.0000, 0.0000] comparable |

Source: `experiments/2026-11-07_v5_supp_analysis/x3_layers.csv`.

#### 4.18.8 Within-layer globalisation

**Definitions.** Layers 6–12, 2,000 held-out test images, tokens after norm1. High-norm tokens of the layer are excluded. Globality g<sub>j</sub> = cosine similarity of x<sub>j</sub> with the mean of the other (non-high-norm) patch tokens of the image and layer. Within each image, tokens are split into 5 rank groups by g; per group: probe exact accuracy, entropy of the probe distribution (natural log), expected C2 move Σ<sub>p</sub> P(p | x<sub>j</sub>) dist(p, j). Per image, Spearman between group number and each quantity, averaged over images (bootstrap). Ring control: positions split into 4 rings of 49 by distance from the centre, the same within each ring, averaged over rings.

**Rule.** Holds if at each of layers 10, 11, 12 at least 2 models show: higher g → lower accuracy (Spearman < 0, interval excluding 0) and higher entropy (Spearman > 0, interval excluding 0), with the ring-control Spearmans of the same sign (point estimates).

| Model | Layer | Mean g | ρ(group, accuracy) | ρ(group, entropy) | ρ(group, C2 move) | Ring: ρ accuracy | Ring: ρ entropy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeiT-S | 10 | 0.4986 | 0.0080 [0.0028, 0.0130] | −0.0780 [−0.0850, −0.0705] | −0.0533 [−0.0601, −0.0462] | −0.0091 | −0.0459 |
| DeiT-S | 11 | 0.5242 | 0.0165 [0.0116, 0.0211] | −0.0989 [−0.1058, −0.0919] | −0.0867 [−0.0933, −0.0799] | 0.0010 | −0.0630 |
| DeiT-S | 12 | 0.5124 | 0.0208 [0.0163, 0.0253] | −0.1108 [−0.1174, −0.1043] | −0.1060 [−0.1127, −0.0993] | 0.0069 | −0.0791 |
| DeiT-B | 10 | 0.4210 | −0.0336 [−0.0388, −0.0287] | −0.0681 [−0.0757, −0.0611] | −0.0108 [−0.0179, −0.0035] | −0.0382 | −0.0669 |
| DeiT-B | 11 | 0.4623 | −0.0820 [−0.0878, −0.0764] | −0.1146 [−0.1218, −0.1074] | 0.0189 [0.0118, 0.0261] | −0.0749 | −0.1454 |
| DeiT-B | 12 | 0.5610 | −0.1596 [−0.1649, −0.1548] | −0.1027 [−0.1102, −0.0951] | 0.0962 [0.0893, 0.1034] | −0.1288 | −0.1572 |
| DINO | 10 | 0.3915 | 0.0529 [0.0475, 0.0576] | −0.0740 [−0.0818, −0.0657] | −0.0653 [−0.0715, −0.0585] | 0.0246 | 0.0330 |
| DINO | 11 | 0.4008 | 0.0675 [0.0624, 0.0725] | −0.0702 [−0.0777, −0.0627] | −0.0723 [−0.0789, −0.0651] | 0.0361 | 0.0592 |
| DINO | 12 | 0.3232 | 0.0820 [0.0770, 0.0862] | −0.0942 [−0.1014, −0.0869] | −0.0939 [−0.1002, −0.0874] | 0.0428 | 0.0462 |

Layer 12, per g group (1 = least global, 5 = most global; no intervals stored):

| Model | Quantity | Group 1 | Group 2 | Group 3 | Group 4 | Group 5 |
| --- | --- | --- | --- | --- | --- | --- |
| DeiT-S | Exact accuracy | 0.1725 | 0.1640 | 0.1822 | 0.1936 | 0.1892 |
| DeiT-S | Entropy | 2.9557 | 2.5597 | 2.5222 | 2.5392 | 2.6328 |
| DeiT-S | Expected C2 move | 3.0275 | 2.5168 | 2.4876 | 2.5006 | 2.5144 |
| DeiT-B | Exact accuracy | 0.4551 | 0.3325 | 0.2852 | 0.2550 | 0.2371 |
| DeiT-B | Entropy | 2.0353 | 1.3689 | 1.4541 | 1.5399 | 1.5845 |
| DeiT-B | Expected C2 move | 1.4970 | 1.4534 | 1.6193 | 1.7743 | 1.8860 |
| DINO | Exact accuracy | 0.3018 | 0.3326 | 0.3560 | 0.3829 | 0.4154 |
| DINO | Entropy | 1.7788 | 1.6966 | 1.6509 | 1.6394 | 1.6578 |
| DINO | Expected C2 move | 1.2038 | 1.1520 | 1.1233 | 1.0961 | 1.0634 |

Mean g per layer:

| Model | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeiT-S | 0.3387 | 0.3702 | 0.4124 | 0.4390 | 0.4986 | 0.5242 | 0.5124 |
| DeiT-B | 0.2343 | 0.2614 | 0.3068 | 0.3606 | 0.4210 | 0.4623 | 0.5610 |
| DINO | 0.3337 | 0.3603 | 0.3726 | 0.3941 | 0.3915 | 0.4008 | 0.3232 |

Source: `experiments/2026-11-07_v5_supp_analysis/x4_layers.csv`. Outcome: the rule is not met (models meeting it at layers 10–12: none; `branches.csv`).

#### 4.18.9 Two-layer MLP probe and C2 variants

**Definitions.** MLP probe: Linear(D → 1,024) → GELU → Linear(1,024 → 196); standardised input, Adam, learning rate 0.01, batch 4,096, 30 epochs, seed 0; trained on 1,800 of the 2,000 probe-train images, the last 200 used for early stopping (epoch with the best validation exact accuracy). Tested on the same 2,000 held-out test images as the linear probe. C2 with the MLP probe: C2 with P<sub>ℓ</sub> from the MLP probe. C2 moving only confident tokens: linear probe; only tokens whose largest probability ≥ 0.5 are relocated, the others keep their score in place.

**Branch rule.** Weak: at layer 12 the MLP exact accuracy exceeds the linear probe by ≤ 10 percentage points in all three models → "probe capacity is not the cause". Strong, C2 still fails: > 10 points in at least one model and C2 with the MLP probe "lower" in more than 2 of 6 cells. Strong, C2 rescued: > 10 points in at least one model and C2 with the MLP probe not "lower" in at least 4 of 6 cells → also report its SRG, a blur matched to its displacement, and C3 + C2 with the MLP probe (computed in the same run, reported only in this branch). Confident-only: if not "lower" in at least 4 of 6 cells → "C2's harm comes mainly from tokens it cannot locate" (report only).

Exact accuracy, linear vs MLP probe:

| Layer | DeiT-S linear | DeiT-S MLP | DeiT-B linear | DeiT-B MLP | DINO linear | DINO MLP |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9993, 0.9995] | 0.9986 [0.9984, 0.9987] |
| 2 | 1.0000 [1.0000, 1.0000] | 0.9997 [0.9996, 0.9998] | 1.0000 [1.0000, 1.0000] | 0.9999 [0.9999, 1.0000] | 0.9996 [0.9995, 0.9997] | 0.9982 [0.9980, 0.9984] |
| 3 | 0.9995 [0.9994, 0.9996] | 0.9997 [0.9996, 0.9997] | 1.0000 [1.0000, 1.0000] | 0.9999 [0.9998, 0.9999] | 0.9988 [0.9985, 0.9990] | 0.9986 [0.9984, 0.9988] |
| 4 | 0.9904 [0.9901, 0.9908] | 0.9964 [0.9962, 0.9966] | 0.9999 [0.9999, 1.0000] | 0.9999 [0.9999, 0.9999] | 0.9982 [0.9979, 0.9984] | 0.9981 [0.9980, 0.9983] |
| 5 | 0.9746 [0.9739, 0.9752] | 0.9904 [0.9900, 0.9907] | 0.9994 [0.9993, 0.9994] | 0.9995 [0.9994, 0.9996] | 0.9956 [0.9952, 0.9959] | 0.9969 [0.9966, 0.9972] |
| 6 | 0.8534 [0.8514, 0.8553] | 0.9052 [0.9036, 0.9068] | 0.9947 [0.9944, 0.9950] | 0.9991 [0.9991, 0.9992] | 0.9890 [0.9885, 0.9896] | 0.9977 [0.9975, 0.9980] |
| 7 | 0.6505 [0.6477, 0.6529] | 0.7021 [0.6995, 0.7048] | 0.9683 [0.9677, 0.9690] | 0.9913 [0.9910, 0.9916] | 0.9587 [0.9573, 0.9599] | 0.9839 [0.9832, 0.9847] |
| 8 | 0.4390 [0.4364, 0.4414] | 0.4746 [0.4716, 0.4772] | 0.9184 [0.9171, 0.9197] | 0.9660 [0.9652, 0.9669] | 0.8562 [0.8536, 0.8586] | 0.9306 [0.9288, 0.9323] |
| 9 | 0.3101 [0.3079, 0.3124] | 0.3337 [0.3309, 0.3359] | 0.8365 [0.8345, 0.8384] | 0.8913 [0.8897, 0.8931] | 0.6738 [0.6707, 0.6768] | 0.7972 [0.7943, 0.8001] |
| 10 | 0.2315 [0.2294, 0.2335] | 0.2562 [0.2541, 0.2582] | 0.6683 [0.6658, 0.6708] | 0.7086 [0.7059, 0.7113] | 0.5048 [0.5019, 0.5078] | 0.6202 [0.6170, 0.6235] |
| 11 | 0.1937 [0.1918, 0.1956] | 0.2154 [0.2133, 0.2173] | 0.4915 [0.4886, 0.4940] | 0.5184 [0.5153, 0.5213] | 0.3923 [0.3896, 0.3950] | 0.4770 [0.4739, 0.4798] |
| 12 | 0.1812 [0.1794, 0.1830] | 0.1990 [0.1971, 0.2009] | 0.3177 [0.3148, 0.3203] | 0.3559 [0.3529, 0.3587] | 0.3575 [0.3549, 0.3600] | 0.4245 [0.4218, 0.4273] |

Layer 12:

| Model | MLP − linear exact (difference of means) | Top-5 linear | Top-5 MLP | Expected distance linear | Expected distance MLP |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | 0.0178 | 0.5310 [0.5276, 0.5346] | 0.5842 [0.5808, 0.5877] | 2.6287 [2.6094, 2.6481] | 2.0454 [2.0282, 2.0630] |
| DeiT-B | 0.0382 | 0.7165 [0.7124, 0.7201] | 0.7602 [0.7558, 0.7642] | 1.6758 [1.6588, 1.6938] | 1.2572 [1.2402, 1.2749] |
| DINO | 0.0671 | 0.8510 [0.8486, 0.8534] | 0.9123 [0.9104, 0.9143] | 1.1281 [1.1223, 1.1339] | 0.7586 [0.7535, 0.7636] |

Source: `experiments/2026-11-07_v5_supp_analysis/x5_probe.csv`.

| Comparison | Model | Same-ring enrichment | ClickMe |
| --- | --- | --- | --- |
| C2 with the MLP probe − R0 | DeiT-S | −0.1054 [−0.1073, −0.1035] lower | −0.0827 [−0.0858, −0.0797] lower |
| C2 with the MLP probe − R0 | DeiT-B | −0.0458 [−0.0470, −0.0446] lower | −0.0482 [−0.0498, −0.0467] lower |
| C2 with the MLP probe − R0 | DINO | −0.0966 [−0.0987, −0.0944] lower | −0.0486 [−0.0504, −0.0469] lower |
| C2 moving only confident tokens − R0 | DeiT-S | −0.0082 [−0.0087, −0.0078] comparable | −0.0043 [−0.0047, −0.0040] comparable |
| C2 moving only confident tokens − R0 | DeiT-B | −0.0338 [−0.0348, −0.0329] lower | −0.0254 [−0.0265, −0.0243] lower |
| C2 moving only confident tokens − R0 | DINO | −0.0397 [−0.0409, −0.0384] lower | −0.0179 [−0.0188, −0.0171] comparable |

Source: `experiments/2026-11-07_v5_supp_analysis/x5.csv`. C2 with the MLP probe: 6/6 cells lower. Confident-only: 3/6 cells not lower. Outcome: branch "weak" (`branches.csv`); confident-only rule not met. The rows for SRG, the matched blur and C3 + C2 with the MLP probe in `x5.csv` belong to the "rescued" branch and are not reported here.

#### 4.18.10 Share of the GAE gap closed by C3

**Definition.** (C3 − R0) ÷ (GAE − R0) from the formal per-image values; per bootstrap resample the means are taken first, then the ratio. Descriptive.

| Model | Metric | Images | C3 − R0 | GAE − R0 | Share closed |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | Enrichment | 12150 | 0.0376 | 0.0635 | 0.5917 [0.5736, 0.6107] |
| DeiT-S | SRG | 2000 | 0.0173 | 0.1090 | 0.1589 [0.1227, 0.1954] |
| DeiT-S | ClickMe | 17408 | 0.0333 | 0.0774 | 0.4304 [0.4196, 0.4418] |
| DeiT-B | Enrichment | 12150 | 0.0654 | 0.1152 | 0.5674 [0.5541, 0.5819] |
| DeiT-B | SRG | 2000 | 0.0287 | 0.1383 | 0.2077 [0.1729, 0.2419] |
| DeiT-B | ClickMe | 17408 | 0.0576 | 0.1062 | 0.5421 [0.5323, 0.5540] |

Source: `experiments/2026-11-07_v5_supp_analysis/x6.csv`.

#### 4.18.11 Outcome summary of the exploratory rules

| Analysis | Outcome under the rule written before running |
| --- | --- |
| Pairwise combinations (4.18.5) | branch B (no further analysis) |
| C3 mechanism (4.18.6) | "expected"; C3 − C3 with ‖v‖ comparable in 4/4 supervised cells |
| Dose–response (4.18.7) | branch A (dose–response holds) |
| MLP probe (4.18.9) | weak (probe capacity not the cause); confident-only rule not met |
| Within-layer globalisation (4.18.8) | rule not met in any model |
| Errors mainly far (4.18.1) | not supported |
| Image-independent probe preference (4.18.2) | not supported |

Source: `experiments/2026-11-07_v5_supp_analysis/branches.csv`; `experiments/2026-11-07_explore_c2_analysis/hypotheses.csv`. The author chose "version B" of paper Section 4.5 (mechanism of the C2 failure not determined) on 2026-11-08 (`81bad1c`).

#### 4.18.12 Discount without relocation (D2)

**Question.** If each unit's R0 score is kept at its own patch but multiplied by the probe's probability that the unit is still at that patch, does the map change?

**Setup.** D2: per layer, R0<sub>j</sub> × P<sub>ℓ</sub>(j | x<sub>j</sub>) (the C2 probe, applied to the same token), no relocation, summed over layers 9–12. Stand-in: the same probabilities permuted among the 196 patch tokens of the image and layer (seeded per seed, batch start and layer; seeds 0–4; per image, the metric is averaged over the five seeds before the paired comparison). Comparisons: D2 − R0, D2 − stand-in, D2 − C2, on same-ring enrichment (ImageNet-S) and ClickMe image-specific Spearman; no SRG. A combined reading (C3 then the same discount) was computed in the same runs and was to be reported only under branch A.

**Registration.** Exploratory, post hoc; definition, stand-in and branch rule committed on 2026-11-08 09:19 UTC (`d9bae2d`) before the runs (code `7f75fa1`, runs from 10:00 UTC, NVIDIA L4, 0.67 GPU hours); analysis script committed before the results (`5fdc681`). Rule: branch A if D2 − R0 is "higher" in ≥ 2 of the 6 cells, "lower" in none, and D2 − stand-in is "higher" in each of those cells; otherwise branch B (report only).

Absolute values (layers 9–12):

| Model | Metric | R0 | D2 | Shuffled stand-in | C2 |
| --- | --- | --- | --- | --- | --- |
| DeiT-S | Same-ring enrichment | 0.1401 [0.1381, 0.1423] | 0.0686 [0.0670, 0.0702] | 0.1090 [0.1074, 0.1108] | 0.0414 [0.0399, 0.0429] |
| DeiT-S | ClickMe image-specific Spearman | 0.1565 [0.1535, 0.1595] | 0.1136 [0.1104, 0.1166] | 0.1366 [0.1341, 0.1392] | 0.0839 [0.0803, 0.0874] |
| DeiT-B | Same-ring enrichment | 0.0805 [0.0788, 0.0823] | 0.0705 [0.0689, 0.0721] | 0.0839 [0.0823, 0.0856] | 0.0269 [0.0257, 0.0281] |
| DeiT-B | ClickMe image-specific Spearman | 0.1071 [0.1045, 0.1099] | 0.1088 [0.1058, 0.1119] | 0.1106 [0.1081, 0.1132] | 0.0501 [0.0471, 0.0532] |
| DINO | Same-ring enrichment | 0.1994 [0.1962, 0.2025] | 0.1150 [0.1128, 0.1172] | 0.1582 [0.1555, 0.1608] | 0.0845 [0.0826, 0.0862] |
| DINO | ClickMe image-specific Spearman | 0.2076 [0.2032, 0.2123] | 0.1680 [0.1645, 0.1714] | 0.1891 [0.1851, 0.1932] | 0.1432 [0.1395, 0.1468] |

Source: `experiments/2026-11-08_explore_d2_analysis/tests.csv` (R0 and C2 recomputed in the same runs).

| Comparison | Model | Same-ring enrichment | ClickMe image-specific Spearman |
| --- | --- | --- | --- |
| D2 − R0 | DeiT-S | -0.0715 [-0.0733, -0.0698] lower | -0.0429 [-0.0461, -0.0396] lower |
| D2 − R0 | DeiT-B | -0.0100 [-0.0112, -0.0088] comparable | +0.0017 [-0.0007, +0.0041] comparable |
| D2 − R0 | DINO | -0.0845 [-0.0864, -0.0824] lower | -0.0396 [-0.0424, -0.0369] lower |
| D2 − shuffled stand-in | DeiT-S | -0.0404 [-0.0418, -0.0391] lower | -0.0230 [-0.0259, -0.0200] lower |
| D2 − shuffled stand-in | DeiT-B | -0.0135 [-0.0144, -0.0125] comparable | -0.0018 [-0.0040, +0.0005] comparable |
| D2 − shuffled stand-in | DINO | -0.0432 [-0.0447, -0.0417] lower | -0.0210 [-0.0235, -0.0188] lower |
| D2 − C2 | DeiT-S | +0.0272 [+0.0258, +0.0285] higher | +0.0296 [+0.0265, +0.0328] higher |
| D2 − C2 | DeiT-B | +0.0436 [+0.0423, +0.0449] higher | +0.0587 [+0.0561, +0.0613] higher |
| D2 − C2 | DINO | +0.0305 [+0.0292, +0.0320] higher | +0.0248 [+0.0223, +0.0275] higher |

Source: `experiments/2026-11-08_explore_d2_analysis/tests.csv`.

**Outcome by the rule.** D2 − R0: 0 cells "higher", 4 "lower", 2 "comparable" → branch B (`experiments/2026-11-08_explore_d2_analysis/branch.csv`). The combined reading is not reported.

**Consistency check.** R0, C2 and C3 recomputed in these runs against the formal runs, per image (`consistency.csv`): ClickMe maximum absolute difference 1.2e-5; same-ring enrichment identical except 1–2 images per reading (of about 12,150), maximum 0.0273 on one image, difference of means ≤ 3e-6.

## 5. Deviations, bugs and known caveats

### 5.1 Deviations from the plan

- **Implementation-check tolerance.** The first DeiT-B k-sweep run stopped: the pre-execution check (explicit-softmax attention reproduces the model's logits) required a maximum difference < 1e-4; DeiT-B gave 1.4e-4 (float32 rounding between fused and explicit softmax; predicted classes identical). The tolerance was raised to 5e-4 (commit `5faa503`, authored 2026-11-05 16:42, committed 17:02), keeping the requirement of identical classes, and the DeiT-B k sweep was rerun from the start. Stored maximum in the rerun: 1.40e-04. The failed folder was kept on the instance and not retrieved. No research definition or decision rule changed.
- **Preprocessing of the "no attention" baseline differs from its description.** The description of the attention-removal experiment says each model uses its own preprocessing, as in the execution figure, but its configuration sets `preprocessing: uniform` (`experiments/2026-11-03_ab_t7_*/config.yaml`). So the "no attention" baseline of Fig. 2(a) (and the single-layer removal column of App. B) used Resize 256 and a 0.875 centre crop for DeiT-S and DeiT-B, while the k sweep used their own 0.9 crop (DINO: 0.875 in both). Same 5,000 image indices. The setting was kept in the release so that the original numbers are reproduced, and it is stated in the configuration; the review version mentions it in App. A. See also Section 5.3.
- **Two code commits in the formal runs** (`94bf426`, `5faa503`): they differ only in that tolerance.
- **Operational incidents (experiment log, 2026-11-05):** an interrupted code upload left the queue starting on old code; it was stopped at the GPU-test stage before any experiment ran. Retrieval of runs was retried after timeouts. No result was produced by these incidents.
- **Registered claim corrected:** the ClickMe images were registered (2026-10-29) as ImageNet validation images; this was corrected before the formal ClickMe runs ended (2026-11-05 17:07, `7584ed2`); experiments and rules unchanged (5.3).
- **Earlier compute-driven reductions that affect reused runs:** DeiT-B head-vs-body on the val + test parts of PartImageNet only (3,614 images); SRG on a fixed 2,000-image subset.

### 5.2 Bugs found and fixed

- **Float32 far share (exploratory probe-error analysis).** The far share of errors used 1 − P(j) in float32; for tokens with P(j) close to 1 this rounds to about 0 and the ratio exceeded 1: in the pre-fix file `experiments/2026-11-07_explore_c2_deit_b/probe_errors.csv` the per-image far share exceeds 1 in 8,437 of 14,000 image × layer rows, and its mean over images at layer 6 is 231.53. Fix (`7015d89`, registered before the rerun): error mass computed in float64 as the sum of the probabilities off j; tokens without error mass excluded. Only the probe part was rerun (`experiments/2026-11-07_explore_c2_probefix_*`, code `534e333`) and only this column replaced; the other columns are per-token means and unchanged. No registered result uses this code.
- **MLP probe device.** When storing its weights the MLP probe was moved to the CPU, which failed on the GPU in a smoke run (CPU tests did not catch it). Fixed (`534e333`) before the exploratory runs; the smoke run was repeated. No registered result uses this code.

### 5.3 Known caveats

- **ClickMe images are likely ImageNet training images.** The Harmonization paper describes its ClickMe evaluation as using maps "captured on the ImageNet validation set"; the released validation shards (17,408 records) are only called the ClickMe validation set. In the held-out selection, 4,002 ImageNet validation candidates were compared with same-class ClickMe thumbnails and 2 matched (`heldout.json`). Two spot checks during the smoke runs (thumbnails under several crops; DeiT-S features) found no matching validation image (outputs not committed). The decision log records the conclusion "very likely from the ImageNet training set; the models have seen them". ClickMe experiments and rules were not changed.
- **MiTA-DeiT-S uses a different crop** (Resize 256, crop 0.875) from DeiT-S (Resize 248, crop 0.9). The top-k training row is paired with DeiT-S by image index; the two models see different crops of the same image.
- **Centred Gaussian subtraction on DeiT-B.** Absolute values (DeiT-B, layers 9–12): SRG 0.0238 [0.0145, 0.0331] vs R0 0.1694 [0.1595, 0.1785]; enrichment 0.0555 [0.0542, 0.0569] vs 0.0805 [0.0788, 0.0823]; ClickMe 0.0622 [0.0576, 0.0669] vs 0.1071 [0.1045, 0.1099]. Stand-in − R0: SRG −0.1456 [−0.1558, −0.1348] lower, enrichment −0.0250 [−0.0258, −0.0241] lower, ClickMe −0.0449 [−0.0498, −0.0401] lower. Consequently C1 − stand-in on DeiT-B: SRG 0.1201 [0.1094, 0.1305] higher, enrichment 0.0218 [0.0210, 0.0227] higher, ClickMe 0.0248 [0.0197, 0.0295] higher.
- **Head vs body uses ImageNet training images** (PartImageNet), seen in training; DeiT-B on a smaller subset (3,614 images).
- **Preprocessing of the attention-removal runs.** The "no attention" curve (all 12 layers) and the single-layer removal column used uniform preprocessing (Resize 256, crop 0.875; `preprocessing: uniform` in `experiments/2026-11-03_ab_t7_*/config.yaml`), whereas the k sweep and the single-layer restriction used each model's own preprocessing (DeiT-S and DeiT-B: crop 0.9; DINO: crop 0.875 in both). Same 5,000 image indices.
- **Value-norm ratio** comes from the attention-budget run with uniform preprocessing on the 11,919-image formal part of ImageNet-S (not the 12,419 images and own preprocessing of Table 1), uses ‖v‖ rather than ‖W<sub>O</sub>v‖, and has no interval. Per-layer values at layers 9–12: 0.214–0.387 (DeiT-S), 0.192–0.268 (DeiT-B) (Section 4.7).
- **Register units** were selected on the 500-image pilot split with uniform preprocessing and applied with DeiT-S's own preprocessing in Table 1.
- **Panels from earlier runs.** M3 (Fig. 3c) and the per-layer quality (Fig. 3d) come from the 2026-10-31 runs, not from the v5 runs; the single-layer restriction from 2026-10-28/29 runs (DeiT-B on an A100).
- **ClickMe-geometry mean maps** use the centre-square geometry because the two ClickMe duplicates found in the held-out check matched that geometry (experiment log, 2026-11-05).
- **Enrichment sample size** is 12,150 (DeiT-S, DeiT-B) or 12,144 (DINO) of 12,419 images; the remaining images have no defined value.
- **GAE** uses gradients of the top-1 logit; the readings do not use the class.
- **Earlier probe vs new probe.** The M2 panel uses the new probes (2,000 training images); the earlier probe (400 images) gives lower accuracies (Section 4.6).

## 6. Experiments not done, and recorded reasons

| Not done | Recorded reason |
| --- | --- |
| Model with trained registers (e.g. DINOv2 with registers, DeiT-III with registers) | Recorded 2026-10-29: public register models use 14-pixel patches or have no CLS token, which breaks the 14×14-grid assumption. Recorded again for v5 (plan, question 4). Paper-draft writing notes: high-norm tokens in DINOv2 appear only at ViT-L and above (Darcet et al. 2024), so no contrast at sizes S/B; listed as a limitation. |
| Test-time register for DeiT-B | Its unit search did not pass the registered gate (≥ 80% fewer high-norm patch-layers with ≥ 95% agreement, n ≤ 32 units): with 32 units the reduction was 0.4488 (`experiments/2026-11-03_ab_t2_deit_b/gate.csv`). |
| SRG for top-k training (MiTA-DeiT-S) | Summary slots pool fixed regions of the full 14×14 grid, undefined after patch removal (author decision, 2026-11-05). |
| SRG for the layer 1–4 negative control | Author chose to judge it on enrichment and ClickMe only (2026-11-05). |
| Rerun of the single-layer tests with the v5 code and data | Plan question 5: same 5,000 images and preprocessing as the earlier runs; "a rerun gives no new information". (The release manifest lists a rerun with new code as a later release step.) |
| Full per-layer matrix (12 layers × all readings) | Research adjustment note: not done "to control length and compute"; layer 12 alone and layers 1–4 are reported. |
| SRG on all 12,419 ImageNet-S images | Plan question 7: compute (each image needs 18 deletion forwards, 9 budgets × 2 orders, for each of the 21 SRG maps of the Table 1 configuration). Observed half-width of the R0 SRG interval on 2,000 images: 0.0103 (DeiT-S; `table1_long.csv`); margin 0.05. |
| Multiple-comparison correction | Plan question 1: kept the earlier statistics (no correction) for comparability with earlier reports. |
| Other blur matchings (equal entropy; fixed σ = 1 patch) | Plan question 2: entropy does not measure how far the score moves; a fixed σ is too weak in deep layers. |
| Excluding high-norm tokens from evaluation as the C3 stand-in | Plan question 9: it changes the metric's denominator, so rows would not be comparable. |
| Per-image map similarity (median Spearman ≥ 0.9) as the negative-control criterion | Plan question 8: the same verdict rule as for the other comparisons was chosen. |
| Other position-probe types beyond linear and the exploratory two-layer MLP | No reason recorded. |
| Object-shifted (corner) test with R0 | Deferred on 2026-11-07 "until results, author to decide"; no later decision recorded. The paper draft lists the corner-shift test as removed from the report. |
| Reported only under a branch that did not occur | C3 with centred Gaussian subtraction (pairwise branch A); layer × head table of C3 weights ("head weighting" outcome); SRG, matched blur and C3 + C2 for the MLP probe ("rescued" branch). Computed in the same runs, not reported. |

**Run earlier but not used in the paper** (paper draft: "removed from this report, kept in the repository"; no reason recorded in the draft): the corner-shift test, the full comparison with Grad-CAM, Integrated Gradients and Rollout, Oxford Pets and Hard ImageNet, distance-matched deletion, compute cost.

## 7. Reproduction status

A release package (branch `release/v1`, package `fastweight_vit`) was verified in three layers on 2026-11-08.

| Layer | What | Result |
| --- | --- | --- |
| 1 | Rebuild every table and figure from the saved outputs of the runs listed in this dossier with the release code; compare with the tables used here | 30 of 31 tables identical (same rows, maximum absolute difference 0). The single-layer table (Section 4.2): means identical, interval bounds differ by at most 0.0012 (the earlier analysis resampled the images in another row order). Main-text figures identical pixel by pixel. |
| 2 | Fresh clone, fresh environment, rerun of every experiment (`reproduce/full.sh`) on one NVIDIA L4, 11.5 GPU hours; tolerances: means 1e-4, every verdict identical, execution agreement 0.1 points, retrained probe accuracy 0.5 points, readings that use the probe 2e-3 | All 31 tables within tolerance; every verdict and prediction outcome identical. 26 tables identical to the last digit. Largest differences: per-layer high-norm share 1.2e-6; per-layer ClickMe quality 1.5e-7; head vs body 7.1e-6; top-k map similarity medians 2.5e-5; value-norm ratio 1.5e-8. In the first rerun the per-layer ClickMe quality differed by up to 3.0e-4 for DeiT-S and DINO because the release configuration used batch size 64 instead of the original 128 (the image-specific control pairs images within a batch); with the original batch sizes the difference is 1.5e-7. |
| 3 | Fresh environment from the pinned requirements: unit tests; 16-image smoke run on a GPU | 78 tests passed (CPU and L4); smoke run completed in 233 s. |

## 8. Internal id → paper name

| Internal id | Name / content in this dossier and the paper |
| --- | --- |
| B09–B16 (2026-10-29, `90e78d5`) | Earlier-study registrations: models and scale, statistics, SRG and enrichment, ClickMe protocol, PartImageNet head vs body, sanity check, cost, qualitative-gallery rule |
| B17–B19 | Earlier compute adjustments (token dropping for removal; fixed subsets; DeiT-B PartImageNet val + test only) |
| B20, B26 | Earlier results ledgers |
| B21 | Earlier positioning (top-k reading sets under a centre-bias control); replaced by B33 |
| B22 / S1 | Full-attention map (R0) vs top-k map: comparison, margins, map similarity (Section 4.4), reproduction reference |
| B23 / S2 | Per-layer curves; source of the per-layer ClickMe quality of R0 (Fig. 3d) |
| B24 / S3 | High-norm token definition and per-layer high-norm share of R0 (M3, Fig. 3c) |
| B27–B32 | Attention-budget study (register unit search, value-norm ratio, attention removal, earlier register/top-k verdicts) |
| B33 | v5 positioning: read → diagnose → correct |
| B34 | Readings R0, C1–C3, R_all, other order; stand-ins; references; change-the-model rows |
| B35 | k sweep (execution test) |
| B36 | Data and held-out set |
| B37 | Diagnostics M1–M3 (Fig. 3) |
| B38 | Predictions P1–P6, statistics, margins |
| B39 | Order of runs and stop rule |
| B40 | Reporting of earlier extensions |
| B41 (`94bf426`) | Author confirmation and freeze; primary comparisons (P7) |
| B42 (`5faa503`) | Implementation-check tolerance 1e-4 → 5e-4 |
| B43 (`7584ed2`) | ClickMe image-source correction |
| B44 (`b35fcf6`) | Formal results ledger |
| B45 (`65aba6c`) | Exploratory: probe error geometry, mean prediction, where C2 moves mass, ClickMe smoothing |
| B46 (`e5894a5`) | Exploratory: pairwise combinations, C3 mechanism, dose–response, globalisation, MLP probe, GAE gap, figures |
| B47 (`7015d89`) | Float32 far-share fix |
| B48 (`9453e46`) | Exploratory results ledger (incl. MLP-probe device bug) |
| B49 (`81bad1c`) | Author decisions after the exploratory results |
| B50 (`d9bae2d`) | Discount without relocation (D2) |
| P1–P7 | Registered predictions 1–7 (paper Appendix C uses P1–P7) |
| P4a / P4b | Negative controls: C3 on DINO / C2 at layers 1–4 |
| P5 / A6 | Last-layer recovery |
| A2 | Single-layer restriction and removal table |
| A7 | Combination orders |
| E3 | Earlier position probe (400 training images) |
| E4 | Earlier deletion and enrichment runs (S1 reruns: `2026-10-31_s_e4_*`) |
| E5 | Earlier object-shift (corner) test |
| E6 | Earlier centre-matched deletion |
| E7 | Earlier ClickMe runs (centre Gaussian 0.41: `2026-10-29_v3_e7_clickme_*`) |
| E8 | PartImageNet runs (head vs body: `2026-10-29_v3_e8_parts_*`) |
| Extension A / B / C / D / E | Few units per region (k sweep, map similarity) / single-layer tests / head vs body / corner shift / change the model |
| T0 | Attention-budget: 50,000-image accuracy check |
| T1 | Attention-budget: pilot budget accounts |
| T2 | Attention-budget: register unit search (`2026-11-03_ab_t2_*`) |
| T3 | Attention-budget: formal budget accounts; value-norm ratio (`2026-11-03_ab_t3_*`) |
| T4 | Attention-budget: high-norm token use and sink typing (`2026-11-03_ab_t4_*`) |
| T5 | Attention-budget: paired comparisons on CPU (no GPU run folder) |
| T6 | Attention-budget: register intervention on the formal split (`2026-11-03_ab_t6_deit`) |
| T7 | Attention removed in all / single layers (`2026-11-03_ab_t7_*`) |
| H1, H2, H6, H3, H4 | Attention-budget hypotheses (DINO vs supervised; top-k training; register effect; unit-search gate; register operation check) |
| E-C2 (incl. H-a, H-b) | Exploratory C2 analysis: probe errors mainly far (a), image-independent probe preference (b) — Sections 4.18.1–4.18.4 |
| X0 | Status table of the exploratory phase |
| X1 (X1-A / X1-B) | Pairwise combinations (branch A / B) — 4.18.5 |
| X2 | C3 mechanism checks — 4.18.6 |
| X3 (X3-A / X3-B) | Dose–response and per-layer — 4.18.7 |
| X4 (H-global) | Within-layer globalisation — 4.18.8 |
| X5 (M-weak / M-strong / version C) | MLP probe and C2 variants (weak / strong-still-fails / rescued) — 4.18.9 |
| X6 | Share of the GAE gap closed — 4.18.10 |
| X7 | Paper figures 2(a), 2(b), 3 |
| X8 | Check of the draft's numbers |
| X9 | Branch summary — 4.18.11 |
| D2, D2-shuffle, C3D2 | Discount without relocation, its stand-in, its combination with C3 — 4.18.12 |
| `attention_mass` | R0 (layers 9–12) in the earlier runs |
| `probe_mass` | Top-k reading-set map (layers 9–12) |
| `L3_spatial`, `rand` | Reading sets (16 regions, spatial routing, CLS own set) / random sets |
| `blur`, `no_high`, `centre_sub` | Matched blur / zero high-norm tokens / centred Gaussian subtraction |
| `R_all_alt` | Other order (C3→C1→C2) |
| `C1C3`, `C2C3`, `C1C2`, `C3v`, `C3shuffle`, `C3headmean`, `C2MLP`, `C2conf` | C1+C3, C2+C3, C1+C2, C3 with ‖v‖, C3 with shuffled weights, C3 with head-mean weights, C2 with the MLP probe, C2 moving only confident tokens |
| `deit`, `deit_b`, `dino`, `mita` | DeiT-S, DeiT-B, DINO, MiTA-DeiT-S |
| Run prefix `2026-11-05_read_ksweep_*` | Execution test, k sweep |
| `2026-11-05_diagnose_probe_*` | Held-out set and position probes (M2, blur widths) |
| `2026-11-05_diagnose_m1_*` | Mean maps, M1, radial share |
| `2026-11-05_correct_ablation_{imagenet_s,clickme}_*` | Table 1 runs |
| `2026-11-05_correct_register_*`, `2026-11-05_correct_mita_*` | Change-the-model rows |
| `2026-11-05_correct_order_*` | Combination orders |
| `2026-11-05_correct_sanity_*` | Sanity check |
| `2026-11-05_p4_*.csv` | Negative controls |
| `2026-11-05_v5_analysis` | Formal analysis tables |
| `2026-11-07_explore_c2_*`, `_probefix_*`, `_analysis` | Exploratory C2 analysis runs, far-share rerun, tables |
| `2026-11-07_supp_heldout_*`, `2026-11-07_supp_eval_*`, `2026-11-07_v5_supp_analysis` | Exploratory supplementary runs and tables |
| `2026-10-28_v2_*`, `2026-10-29_v3_*` | Earlier-study runs (single-layer restriction; PartImageNet; ClickMe 0.41) |
| `2026-10-31_s_*` | Earlier full-attention runs (map similarity, M3, per-layer quality, reproduction reference) |
| `2026-11-03_ab_*` | Attention-budget runs |
