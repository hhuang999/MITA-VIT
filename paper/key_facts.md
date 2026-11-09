# Key facts for writing the paper

Every statement below has been checked against the result files in `results/`. Where an earlier draft was wrong, this list gives the corrected wording; please do not reintroduce the old numbers. Names follow `naming.md`.

Conventions:
- Three numbers separated by " / " are DeiT-S / DeiT-B / DINO ViT-S.
- All maps are summed over layers 9–12 unless stated otherwise.
- Verdict words: a difference is **higher** or **lower** only if its 95% paired-bootstrap interval excludes 0 and it exceeds a margin fixed in advance (0.05 for deletion faithfulness, 0.02 otherwise). Otherwise it is **comparable**.

## Setup
- **Models.**
  - DeiT-S and DeiT-B (supervised; resize 248, crop 224, crop ratio 0.9).
  - DINO ViT-S/16 with its official linear head (resize 256, crop ratio 0.875).
  - Two model-side variants of DeiT-S:
    - + test-time register: 8 FFN units moved to one added token;
    - top-k training (MiTA): the MiTA authors' checkpoint, crop 0.875.
- **Data.**
  - Execution test: 5,000 ImageNet validation images.
  - ImageNet-S: all 12,419 images for same-ring object enrichment. The metric is defined for 12,150 / 12,150 / 12,144 of them; the rest have no valid object patch.
  - Deletion faithfulness: a fixed 2,000 ImageNet-S images.
  - ClickMe: 17,408 maps. These images are **very likely ImageNet training images**: 2 of 4,002 held-out validation candidates matched a ClickMe image. State the same caveat ("very likely") everywhere.
  - Held-out set: 4,000 ImageNet validation images. 2,000 train the position probes; 2,000 give the mean maps and the probe accuracy. None is evaluated.
- **Probes.** 36 linear position probes (12 layers × 3 models), 196 classes, on the output of the block's first layer norm. Adam, learning rate 0.01, batch 4,096, 30 epochs, seed 0.
- **Statistics.** 1,000 bootstrap resamples over images, seed 0; 95% percentile intervals.
- **Compute and reproduction.**
  - About 19 GPU hours in total, mostly NVIDIA L4. Three reused earlier runs ran partly on an A100: the single-layer restriction (DeiT-B), head versus body, and the centre-Gaussian ClickMe number.
  - A full rerun from a fresh clone (11.5 GPU hours, one L4) reproduced all 31 result tables. Every verdict was identical; the largest difference of a mean was 2.5e-5.

## Read (execution test)
- Each image region reads only its top-k units in all 12 layers. With k = 16 (about 8% of 196 units), the models keep **72% / 83% / 61%** of their top-1 predictions.
- Random sets of the same size keep **9% / 30% / 4%**; attention removed in all layers keeps **0.1%–0.2%**.
- Top-k reading sets beat random sets at every k from 4 to 128 in every model. This prediction holds 11/11 tests per model.
- **Caveat.** The attention-removed baseline used a uniform 0.875 crop for all models. For DeiT-S and DeiT-B this differs from the 0.9 crop of the k sweep. Same image indices.
- **One layer at a time** (appendix):
  - Restricting a single layer keeps at least 77% of the predictions (DINO layer 1) and at least 91.7% for DeiT-S and DeiT-B.
  - Removing attention from layer 1 alone leaves 0.2% for DINO, 69% for DeiT-S and 88% for DeiT-B. Removing any other single layer keeps at least 85.7%.
- **Head versus body** (PartImageNet, ImageNet training images; DeiT-B on a 3,614-image subset):
  - Removing 4 head patches (drawn at random within the head, seeds 0–4) hurts more than removing 4 body patches, in all three models.
  - The ratio of the drops is about 3.4 / 3.6 / 5.1.
- **Top-k map versus full map:** median per-image Spearman 0.90–0.91.

## The reading itself
- Received attention, layers 9–12:
  - same-ring object enrichment 0.140 / 0.080 / 0.199;
  - image-specific ClickMe alignment 0.157 / 0.107 / 0.208.
- Every image-independent reference scores about 0 on both metrics.
- **Centre-bias motivation:** a fixed centred Gaussian has a plain (not image-specific) Spearman of 0.41 with ClickMe.
- **GAE:** its mean is higher than received attention in all nine model × metric cells, by 0.022 to 0.138. For example, deletion faithfulness on DeiT-S is 0.345 against 0.236. No paired test is stored for GAE, so describe it as a difference of means.

## Diagnose (three premises)
- **Position-probe accuracy** is close to 1 in layers 1–5. At layer 12 it is **18% / 32% / 36%**.
- **Share of received attention on high-norm tokens:**
  - DeiT-B: **44%** at layer 11 (0.435). The earlier draft's "43%" is wrong.
  - DeiT-S: about 9% at layers 8–10 (largest 9.3% at layer 8).
  - DINO ViT-S: 0% at every layer (largest interval upper bound 0.0003).
- **Value-vector norm** of high-norm tokens relative to ordinary patches:
  - averaged over layers 9–12: 0.29 (DeiT-S) and 0.23 (DeiT-B);
  - per layer: 0.21–0.39 (DeiT-S) and 0.19–0.27 (DeiT-B);
  - computed on 11,919 ImageNet-S images with uniform preprocessing, using ‖v‖.
- **Share of variance explained by the mean map:**
  - supervised models, layers 6–12: 3%–6%;
  - DINO ViT-S, layers 6–12: up to 17% (16.7% at layer 11). Over all layers it reaches 19.7% at layer 3, so state the layer range.
  - In layers 9–12, 82%–98% of the mean map's variance is explained by distance from the centre. In layers 6–8 this goes down to 0.72, so state the layer range here too.
- No single premise lines up with the drop of received attention's quality at layer 12.

## Correct
- **Contribution weighting versus received attention:**
  - enrichment +0.038 / +0.065, higher;
  - ClickMe +0.033 / +0.058, higher;
  - deletion +0.017 / +0.029 / 0.000, comparable;
  - DINO ViT-S: all differences ≤ 0.004, comparable (pre-registered negative control).
- **Contribution weighting versus its stand-in (zero high-norm tokens):**
  - enrichment +0.026 / +0.025, higher in both supervised models;
  - ClickMe +0.022 (DeiT-S, higher) and +0.017 (DeiT-B, comparable);
  - deletion comparable in both models.
  - So: it beats its stand-in **on object enrichment**, and on ClickMe only in DeiT-S. Do not write that it beats the stand-in everywhere.
- **Last-layer recovery:** contribution weighting recovers 31%–62% of the last-layer drop in enrichment and ClickMe (supervised models), and 8%–29% in deletion. No other correction recovers any.
- **Contribution weighting, exploratory checks:**
  - Shuffling the weights among the tokens of a head, or using the head-mean weight, removes the gain. Contribution weighting minus the shuffled version: +0.038 / +0.073 enrichment, +0.035 / +0.067 ClickMe.
  - Dose–response: images with more received attention on high-norm tokens gain more (Spearman with the ClickMe gain +0.045 / +0.158).
  - Value-norm weighting (‖v‖ instead of ‖W_O v‖) gives the same result. So in practice the correction is value-norm weighting, which Kobayashi et al. (2020) proposed; the method itself is not new.
  - Share of the received-attention-to-GAE gap closed: 57%–59% (enrichment), 43%–54% (ClickMe), 16%–21% (deletion).
  - At single layers 8–11 of DeiT-B, zeroing high-norm tokens gains more than contribution weighting. Contribution weighting gains more only at layer 12, and when layers 9–12 are combined.
- **Mean-map subtraction:**
  - Comparable to received attention on enrichment in all models.
  - Its one gain, DINO ViT-S ClickMe +0.044, is matched by centre-Gaussian subtraction (0.252 vs 0.253): the mean map is almost radial.
  - On DeiT-B it is higher than its stand-in on all three metrics, but only because centre-Gaussian subtraction falls below received attention there. Mean-map subtraction itself is not higher than received attention on any metric (DeiT-B ClickMe −0.020, lower).
- **Probe relocation:**
  - Lower than received attention in all nine cells. Example: DINO ViT-S enrichment falls from 0.199 to 0.084, while the matched blur only falls to 0.189.
  - Exception to "worse than the blur": on DeiT-B enrichment the blur is −0.008 and probe relocation (0.027) is higher than the blur, though still lower than received attention.
- **Change the model:**
  - + test-time register: +0.012 / +0.013 against DeiT-S received attention, comparable;
  - top-k training (MiTA): −0.040 / −0.042, lower.
  - Neither is higher (pre-registered prediction holds, 4/4).
- **All three corrections:**
  - Order A (contribution weighting → probe relocation → mean-map subtraction, pre-registered) is higher than received attention in only 1 of the 4 supervised cells (DeiT-B ClickMe +0.030). The main prediction **partly holds**.
  - Order B (contribution weighting → mean-map subtraction → probe relocation) beats order A in all nine cells, but it was not fixed in advance; base no claim on it.
- **Pairwise combinations:**
  - Contribution weighting + mean-map subtraction versus contribution weighting alone: enrichment −0.003 / 0.000, ClickMe +0.019 / +0.017. Not beyond the margins.
  - Adding probe relocation to contribution weighting makes every cell worse.
  - Recommendation: contribution weighting alone.

## Why probe relocation fails (all exploratory; rules fixed before each run)
- **Far errors (rule not met).** At layer 12:
  - the probe puts 45% / 66% / 81% of its probability inside the 3×3 neighbourhood of the true patch;
  - only 4%–29% of its error mass lies more than 3 patches away (DINO 3.7%, DeiT-B 15.2%, DeiT-S 29.4%).
- **Fixed positional prior (rule met in 0 of 12 cells).** The probe's mean prediction is close to uniform: its peak is 1.1–1.3 times the uniform value at layers 9–12. It is at most weakly related to the mean map (Pearson −0.17 / −0.18 at DeiT-S layers 11 / 12).
- **Weak probe.**
  - A two-layer MLP probe raises layer-12 accuracy by 1.8 / 3.8 / 6.7 points.
  - Probe relocation with the MLP probe is still lower than received attention in all six cells.
  - Relocating only confident tokens (top probability ≥ 0.5) removes 37%–94% of the loss, not all of it.
- **Globalisation (rule holds in no model).** The probe is not less certain about tokens that are more similar to the image's mean token; its entropy falls with globality in all three models. Accuracy falls with globality only in DeiT-B (layers 10–12). Do not write "global tokens are not harder to place" without this qualification.
- **What remains is a description.** On DeiT-S, probe relocation moves about as much score off the object as the matched blur (0.041 vs 0.039), yet lowers enrichment twice as much (−0.099 vs −0.045). It scatters scores irregularly and so scrambles their ranking.
- **Probability discount** (keep each score in place, multiply by the probe's probability that the unit is still at its patch):
  - lower than received attention in 4 of 6 cells (DeiT-S and DINO ViT-S, both metrics), comparable on DeiT-B;
  - lower than the shuffled probability discount in DeiT-S and DINO ViT-S on both metrics (comparable on DeiT-B);
  - higher than probe relocation in all six cells.
  - This suggests that, in those two models, the probe is less sure about object patches than about background patches. This is an inference, not measured directly.
- **ClickMe rewards smoothing.** A matched blur loses 0.029–0.049 less on ClickMe than on enrichment, in every model.

## Checks
- **Negative controls:** contribution weighting on DINO ViT-S, and probe relocation at layers 1–4, are comparable to received attention in every cell (absolute differences ≤ 0.004).
- **Model-randomisation sanity check:** 1,000 ImageNet-S images, cumulative re-initialisation from the classifier down to the patch embedding. Every reading passes (median Spearman to the original ≤ 0.37 after full randomisation). The centred Gaussian, which does not depend on the model, is the designed failing example.
- **Reproduction check:** the new received attention matches the earlier full-attention runs; every difference is below the registered 0.01.
- **Pre-registration:** definitions, margins, statistics, primary comparisons and predictions were committed before the main runs. One implementation-check tolerance was raised (1e-4 → 5e-4) because DeiT-B gave 1.4e-4 with identical classes; no research definition changed. Every exploratory analysis had its rule committed before it ran.
