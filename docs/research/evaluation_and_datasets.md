# B. Evaluation protocols, centre bias, and datasets for the MiTA reading-set probe

Survey date: 2026-10-29. Scope: faithfulness/quality metrics for token-importance explanations of ViTs (RQ1, RQ4), centre-bias handling (RQ3), and datasets for "human common sense" tests (RQ3).

How to read the flags:
- **verified=yes**: I opened the paper or dataset page (or an official mirror) in this session and checked the facts quoted here.
- **verified=partial**: the page was found, but at least one quoted detail (usually the venue or a number) comes from a secondary source or from memory. The detail in question is flagged.
- **verified=no**: not checked in this session.
- "unknown" means I could not find the value. I did not estimate it.

---

## 0. Executive summary

1. **Use token dropping as the main removal operator.** For ViTs, the natural way to remove a patch is to drop its token from the sequence after patch embedding and positional embedding. Kept tokens retain their positions. This is the same operation the probe performs ("execute only through the reading sets"), so the probe and every baseline are measured on equal terms. Jain et al. (ICLR 2022) show token dropping avoids the "missingness bias" of blacking out pixels. TIS (ICCVW 2023) and SkipPLUS (CVPRW 2024) already evaluate ViT attributions with "true token masking". Use pixel-space removal (mean replacement, blur, or ROAD imputation) only as a robustness check.
2. **Compare at equal budgets on a common unit.** Convert every method to a score over the 196 patch tokens, then report:
   - (a) insertion/sufficiency and deletion/comprehensiveness curves, using top-1 agreement with the full model (label-free) and target-probability drop;
   - (b) fixed-budget ERASER comprehensiveness and sufficiency at k = the probe's own set size;
   - (c) SRG = LIF − MIF (Blücher et al. 2024), which is 0 for a random ranking and stable across occlusion strategies;
   - (d) SaCo (CVPR 2024) for methods with graded scores.

   Always include random, centre (nearest-centre-k, centre Gaussian, CB-CAM) and ground-truth-mask reference "explanations". Also run a cheap cascading-randomization sanity check.
3. **Centre bias is the main confound in localization and plausibility metrics.** The literature has measured it directly:
   - A "point at the image centre" baseline scores 69.5% (all images) / 42.6% (difficult subset) on the VOC07 pointing game. On COCO it scores 27.7% / 19.4% (Zhang et al. 2016; TorchRay reproduces 69.6/42.4 and 27.8/19.5).
   - A centre Gaussian gets 52.5 MaxBoxAcc on ImageNet, against 63.5 for CAM (Choe et al. 2020).
   - A mean-ClickMe-map centre control was beaten by only 42 of 84 CNNs (Fel et al. 2022).
   - CB-CAM scores ΔA^F = 1.26, close to Grad-CAM's 1.31, on 5,000 ImageNet images (Ayyar et al. 2025).

   Report centre-adjusted scores (gain over the centre baseline, as in Information Gain). Also report shuffled-mask scores (the sAUC idea) and radius-stratified foreground enrichment. Stratify results by object distance from the centre and object size. Add a shifted-crop intervention that moves the same object off-centre.
4. **Ranked dataset plan (all zero-shot with the ImageNet-1k heads; no training needed):**
   1. ImageNet-S val masks, plus ImageNet val boxes, ImageNet-X factor labels and shifted crops.
   2. Hard ImageNet val (750 images; objects less centred and smaller, with strong spurious context).
   3. ClickMe val human importance maps (about 17.4k ImageNet-val images; an estimate, see §3.11).
   4. PartImageNet for parts. Caveat: its images come from the ImageNet training set.

   Datasets that need training (CUB species, Waterbirds, Spawrious, VOC/COCO class heads) are low priority for a 4-week window.

---

## 1. Faithfulness and quality metrics for token-importance explanations

### 1.1 Paper entries (metrics and protocols)

**[M1] ERASER: A Benchmark to Evaluate Rationalized NLP Models.** DeYoung, Jain, Rajani, Lehman, Xiong, Socher, Wallace. ACL 2020. https://aclanthology.org/2020.acl-main.408/ (verified=yes)
- *What it does:* Defines two faithfulness scores for a rationale r.
  - comprehensiveness = m(x)_j − m(x \ r)_j, the confidence drop when the rationale is removed;
  - sufficiency = m(x)_j − m(r)_j, the confidence drop when only the rationale is kept.

  Soft scores are turned into top-k_d rationales. The aggregate "AOPC" version averages over bins of the top 1%, 5%, 10%, 20% and 50% of tokens. Random-score reference values are reported. The paper notes AOPC is like ROAR without retraining.
- *Relevance:* This is exactly the probe's sufficiency/necessity framing. Use it at a fixed budget equal to the probe's set size so the probe's sets and baselines' top-k sets have the same cardinality.
- *Pitfall:* With a fixed k, results depend on k. Report curves or several bins.

**[M2] RISE: Randomized Input Sampling for Explanation of Black-box Models.** Petsiuk, Das, Saenko. BMVC 2018. https://arxiv.org/abs/1806.07421 (verified=yes for the deletion definition)
- *What it does:* Introduces the deletion and insertion metrics.
  - Deletion: remove pixels in decreasing importance and take the area under the class-probability curve (lower is better).
  - Insertion: add pixels in decreasing importance and take the area (higher is better).

  Insertion starts from a blurred canvas and deletion replaces pixels with a constant. The canvas detail is as described by later papers such as CAAP 2026 [M17] (verified=partial).
- *Relevance:* The deletion/insertion AUC is the de-facto standard in ViT XAI papers (TIS, SkipPLUS, FovEx, RFEM).
- *Pitfalls:* Blur or constant canvases are out of distribution. Deletion and insertion can rank methods differently (see ROAD and Blücher).

**[M3] Transformer Interpretability Beyond Attention Visualization.** Chefer, Gur, Wolf. CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html (verified=yes)
- *What it does:* Sets the ViT-explanation evaluation convention.
  - Positive/negative perturbation: gradually mask pixels from most to least relevant (positive) or the reverse (negative), measure mean top-1 accuracy, and report the AUC over 10%–90% erased. Lower is better for positive, higher for negative. Run for both the predicted and the ground-truth class; class-agnostic methods behave the same in both.
  - Segmentation test on ImageNet-Segmentation (4,276 images, 445 categories): pixel accuracy after thresholding at the mean, mIoU, and threshold-free mAP.
  - Baselines: raw attention, rollout, GradCAM on the last block's CLS attention, LRP, partial LRP.
- *Relevance:* This is the protocol RQ4 reviewers will expect. Reproduce it, but swap pixel masking for token dropping (keep pixel masking as a secondary check).
- *Pitfall:* The official loader (`data/Imagenet.py`) reads only `/value/img` and `/value/gt`, a binary fg/bg mask with no class label. It resizes images to 224×224 without cropping (verified by reading the code), which distorts the aspect ratio.

**[M4] Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers.** Chefer, Gur, Wolf. ICCV 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Chefer_Generic_Attention-Model_Explainability_for_Interpreting_Bi-Modal_and_Encoder-Decoder_Transformers_ICCV_2021_paper.html (verified=yes)
- *What it does:* A gradient-weighted attention relevance rule ("GAE" / "Transformer-MM"). Code: https://github.com/hila-chefer/Transformer-MM-Explainability
- *Relevance:* A cheap, strong baseline for RQ4. It is easy to re-implement with forward/backward hooks on any timm ViT, including DeiT-S and DINO ViT-S.

**[M5] Quantifying Attention Flow in Transformers.** Abnar, Zuidema. ACL 2020. https://aclanthology.org/2020.acl-main.385/ (verified=yes)
- *What it does:* Introduces attention rollout and attention flow (residual-aware propagation of attention). Both correlate better with gradient and blank-out importance than raw attention.
- *Relevance:* A mandatory baseline for RQ4 and the natural "multi-layer attention" comparison for RQ2.

**[M6] A Benchmark for Interpretability Methods in Deep Neural Networks (ROAR).** Hooker, Erhan, Kindermans, Kim. NeurIPS 2019. https://arxiv.org/abs/1806.10758 (verified=yes)
- *What it does:* Remove And Retrain. Remove the top-x% features, retrain, and measure the accuracy drop. Many popular methods are no better than random; only VarGrad and SmoothGrad-Squared beat random.
- *Relevance:* Shows that "better than random" must be tested. Retraining is impossible for frozen DeiT-S and DINO and out of budget, so use ROAD or token dropping instead and say so explicitly.

**[M7] A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD).** Rong, Leemann, Borisov, Kasneci, Kasneci. ICML 2022 (PMLR 162). https://proceedings.mlr.press/v162/rong22a.html, code https://github.com/tleemann/road_evaluation (verified=yes)
- *What it does:*
  - Shows that pixel-perturbation results are driven by **class information leakage through the shape of the removed pixels**, not by their values.
  - Proposes Noisy Linear Imputation.
  - With fixed-value imputation, the MoRF and LeRF rankings are uncorrelated (Spearman about −0.01 / 0.01). With ROAD they correlate at 0.61 / 0.58.
  - ROAD makes ROAR retraining unnecessary (about 99% less compute).
- *Relevance:* Two lessons for token dropping:
  - Report both MoRF (deletion) and LeRF orders.
  - Token dropping also leaks shape, because kept tokens keep their positional embeddings. Treat the order disagreement as a diagnostic, and include random and centre-matched sets of the same size.

  ROAD is in Quantus, so it can serve as the pixel-space robustness check.

**[M8] On the Faithfulness of Vision Transformer Explanations (SaCo).** Wu, Kang, Tang, Hong, Yan. CVPR 2024. https://arxiv.org/abs/2404.01415, code https://github.com/adreamwu/SaCo (verified=yes)
- *How it is computed:*
  1. Sort pixels by salience and split them into K equal groups G_1…G_K.
  2. Compute each group's salience s(G_i) as the sum of its scores.
  3. Perturb each group **individually** (per-sample mean replacement) and record ∇pred(G_i) = p(ŷ|x) − p(ŷ|x with G_i replaced).
  4. For every pair i<j: if ∇pred(G_i) ≥ ∇pred(G_j), add s(G_i) − s(G_j); otherwise subtract it.
  5. Normalise by the total absolute weight, giving F ∈ [−1, 1].
- *Findings:* Cumulative-perturbation metrics often cannot separate state-of-the-art methods from Random Attribution. Adding gradients and multi-layer aggregation improves attention-based explanations. It evaluates 10 methods (Integrated Gradients, Grad-CAM, LRP, partial LRP, conservative LRP, Transformer Attribution, raw attention, rollout, Transformer-MM, ATTCAT) on 3 datasets and 3 ViTs.
- *Relevance:* The best ViT-specific complement to deletion/insertion. Use K = 10 groups of about 20 tokens, with **token dropping per group** instead of mean replacement.
- *Pitfall:* SaCo uses score magnitudes. A set-valued probe needs a graded score first (e.g., selection frequency across the m regions, heads or layers), otherwise its many ties make SaCo ill-defined.

**[M9] Top-down Neural Attention by Excitation Backprop (Pointing Game).** Zhang, Lin, Brandt, Shen, Sclaroff. ECCV 2016. https://arxiv.org/abs/1608.00507 (verified=yes)
- *What it does:*
  - Pointing game: a hit if the maximum of the map lies on an annotated instance of the cued class (15-pixel tolerance). Accuracy = hits / (hits + misses), averaged over categories.
  - Reports a **"Center" baseline that points at the image centre**: VOC07 69.5% (all) / 42.6% (difficult); COCO 27.7% / 19.4%.
  - "Difficult" subset: the class's total box (or segment) area is under 1/4 of the image **and** at least one distractor category is present.
- *Relevance:* The canonical precedent for a centre baseline and a hard, off-centre subset. Use a patch-level pointing game: the argmax token must lie in the mask, optionally with a one-patch tolerance.

**[M10] TorchRay pointing-game benchmark (with Fong, Patrick, Vedaldi, "Understanding Deep Networks via Extremal Perturbations and Smooth Masks", ICCV 2019).** https://facebookresearch.github.io/TorchRay/benchmark.html (verified=yes)
- *What it does:* A re-implementation on VOC07 test and COCO14 val with VGG16/ResNet50 **fine-tuned multi-label** on those datasets. Centre baseline: VOC07 69.6 / 42.4, COCO 27.8 / 19.5.
- *Relevance:* Confirms the standard VOC/COCO pointing game needs fine-tuned heads, so it is not zero-shot for ImageNet-1k models. COCO is far less centre-biased than VOC07.

**[M11] Score-CAM: Score-Weighted Visual Explanations for CNNs (Energy-Based Pointing Game).** Wang et al. CVPR Workshops 2020. https://openaccess.thecvf.com/content_CVPRW_2020/html/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.html (verified=yes)
- *What it does:* EBPG = Σ_{in box} L / Σ_all L, the share of saliency "energy" inside the box. They note ILSVRC objects often fill most of the image, so they **drop images where the object covers more than 50% of the image** and keep single-box images (500 val images).
- *Relevance:* Use energy-in-mask (also called "relevance mass") per token. Always report it against the mask-area fraction: a uniform map scores exactly the area fraction. Filter or stratify out large objects.

**[M12] Sanity Checks for Saliency Maps.** Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim. NeurIPS 2018. https://arxiv.org/abs/1810.03292 (verified=yes)
- *What it does:*
  - Model-parameter randomization: cascading from the top layer down, or one layer at a time.
  - Data (label) randomization.
  - Compare explanations with Spearman correlation (with and without absolute values), SSIM and HOG.
  - Gradients and GradCAM pass; Guided Backprop fails. Visual similarity can mislead.
- *Relevance:* A cheap must-have for RQ4. Randomize DeiT-S blocks 12→1 (cascading) on about 200 images and check that the probe's token scores decorrelate. Skip the label-randomization test, since it needs retraining.

**[M13] Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond.** Hedström et al. JMLR 24(34), 2023. https://jmlr.org/papers/v24/22-0142.html, code https://github.com/understandable-machine-intelligence-lab/Quantus (verified=yes)
- *What it does:* 30+ metrics (the README now says 35+) in 6 categories: faithfulness (pixel flipping, region perturbation, selectivity, Faithfulness Correlation and Estimate, Monotonicity, SensitivityN, IROF, Infidelity, ROAD, Sufficiency), robustness, localisation (pointing game, attribution localisation, top-k intersection, relevance rank/mass accuracy, AUC…), complexity, randomisation (MPRT and variants), and axiomatic. PyTorch support.
- *Relevance:* Use it for the pixel-space robustness check (ROAD, Region Perturbation) and for localisation metrics. Quantus works in pixel space with replacement baselines, so **the token-dropping curves must be custom code** (a few dozen lines with timm).

**[M14] Missingness Bias in Model Debugging.** Jain, Salman, Wong, Zhang, Vineet, Vemprala, Madry. ICLR 2022. https://arxiv.org/abs/2204.08945, OpenReview https://openreview.net/forum?id=Te5ytkqsnl (verified=yes)
- *What it does:*
  - Blacking out pixels skews the output distribution towards unrelated classes, and makes LIME explanations inconsistent and indistinguishable from random.
  - For ViTs, **dropping the tokens** of removed regions (positional encodings of the rest kept) largely removes this bias.
  - Appendix C.7: dropping tokens beats blacking out pixels even for ViTs.
  - Code handles variable-length batches with dummy tokens and attention masking.
- *Relevance:* The key citation for using token dropping as the primary removal operator in RQ1 and RQ4.

**[M15] The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations.** Hase, Xie, Bansal. NeurIPS 2021. https://arxiv.org/abs/2106.00786 (verified=yes)
- *What it does:* Argues that ablated inputs are OOD, which undermines sufficiency and comprehensiveness. Compares Replace functions (Attention Mask, Mask Token, Slice-Out, Zero Vector, Marginalize). Without counterfactual training, Attention Mask or Mask Token work best.
- *Relevance:* Supports token removal through attention masking (equivalent to dropping) and frames the OOD caveat to state in the paper.

**[M16] Explaining Through Transformer Input Sampling (TIS).** Englebert et al. ICCV Workshops (NIVT) 2023. https://openaccess.thecvf.com/content/ICCV2023W/NIVT/papers/Englebert_Explaining_Through_Transformer_Input_Sampling_ICCVW_2023_paper.pdf, code https://github.com/aenglebert/Transformer_Input_Sampling (verified=yes)
- *What it does:* A perturbation explainer that samples subsets of input tokens after projection and positional embedding, so no replacement values are needed. Evaluated with insertion, deletion and the pointing game on ViT and DeiT.
- *Relevance:* (a) A precedent for token-level perturbation in ViT explanation. (b) A strong, conceptually close RQ4 baseline: a token-subset method compared against the probe's token-subset reading.

**[M17] SkipPLUS: Skip the First Few Layers to Better Explain Vision Transformers.** Mehri, Fayyaz, Soleymani Baghshah, Pilehvar. CVPR Workshops 2024. https://doi.org/10.1109/cvprw63382.2024.00025 (verified=yes)
- *What it does:* Evaluates with deletion/insertion AUC, using **"True Token Masking"**: masked patches are excluded entirely and correct positional encodings are kept for the rest.
- *Relevance:* A second precedent for token-dropping evaluation of ViT attributions.

**[M18] Decoupling Pixel Flipping and Occlusion Strategy for Consistent XAI Benchmarks.** Blücher, Vielhaben, Strodthoff. arXiv 2401.06654 (2024); listed on OpenReview (https://openreview.net/pdf?id=bIiLXdtUVM). The TMLR venue is my reading of the OpenReview listing (verified=partial: content yes, venue not confirmed). https://arxiv.org/abs/2401.06654
- *What it does:*
  - The R-OMS score measures how far out of scope an occlusion strategy is.
  - MIF and LIF rankings disagree, and which is informative depends on R-OMS.
  - Proposes **SRG = LIF − MIF**, the area between the two curves: 0 for a random ranking, upper bound 1, consistent rankings across 40 occlusion strategies.
- *Relevance:* Report SRG with token dropping as the single headline faithfulness number for RQ4. It is robust to the choice of removal operator.

**[M19] On the stability, correctness and plausibility of visual explanation methods based on feature importance.** Xu-Darme et al. 2023. ACM DOI https://doi.org/10.1145/3617233.3617257, arXiv https://arxiv.org/abs/2311.12860 (verified=yes for content; venue name not checked; the DOI is an ACM proceedings, likely CBMI 2023)
- *What it does:* Proposes **CB-CAM**, a 7×7 map with a single 1 at the centre, upsampled, as a trivial centre-biased "explanation". CB-CAM correlates with human gaze maps (Pearson) better than most real methods. Metrics for stability, correctness and plausibility often disagree.
- *Relevance:* Add CB-CAM, or the probe's "nearest-centre tokens", as a baseline "method" in every RQ3/RQ4 table.

**[M20] There is More to Attention: Statistical Filtering Enhances Explanations in Vision Transformers (RFEM).** Ayyar, Benois-Pineau, Zemmari. arXiv 2510.06070 (2025). https://arxiv.org/html/2510.06070 (verified=partial: arXiv only, venue unknown)
- *What it does:* Compares AttnLRP, Grad×Relevance (Chefer), rollout, SAW, GradCAM and RFEM using insertion/deletion, AD/AI/AG, Blücher's ΔA^F, stability metrics, and plausibility against gaze maps.
- *Key number:* On 5,000 ImageNet-val images, **CB-CAM scores ΔA^F = 1.26 against GradCAM's 1.31**, while rollout gets 7.90 and AttnLRP 8.69. A centre prior can tie a real method even on a faithfulness metric.
- *Relevance:* A recent template for a mixed faithfulness and plausibility evaluation with a centre baseline.

**[M21] AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers.** Achtibat et al. ICML 2024 (PMLR 235). https://proceedings.mlr.press/v235/achtibat24a.html, library https://github.com/rachtibat/LRP-eXplains-Transformers (verified=yes)
- *Relevance:* The current strongest LRP-family baseline (RFEM's table has it on top for ImageNet). Include it if the `lxt` library supports timm ViT-S easily. I did not verify that support.

**[M22] Token Transformation Matters: Towards Faithful Post-hoc Explanation for Vision Transformer (TokenTM).** Wu et al. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Token_Transformation_Matters_Towards_Faithful_Post-hoc_Explanation_for_Vision_Transformer_CVPR_2024_paper.pdf (verified=partial: I saw only the evaluation snippet, which uses Chefer-style positive/negative tests)
- *Relevance:* An optional extra RQ4 baseline. Its idea of accounting for token transformations (norms) relates to the high-norm-token analysis.

**[M23] M4: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods across Metrics, Modalities and Models.** Li et al. NeurIPS 2023 Datasets & Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2023/file/05957c194f4c77ac9d91e1374d2def6b-Paper-Datasets_and_Benchmarks.pdf (verified=partial: abstract only; author list not checked)
- *Relevance:* A benchmark that includes Transformers. Mainly useful for metric taxonomy.

**[M24] A Meaningful Perturbation Metric for Evaluating Explainability Methods (Stratified Inpainting).** Cohen, Chefer, Wolf. arXiv 2504.06800 (2025). https://arxiv.org/abs/2504.06800 (verified=partial: arXiv; venue unknown)
- *What it does:* Inpainting-based perturbation. Deletion, blur and mean perturbations often fail to separate methods from random. For ViT, its results agree with pixel-blur and per-channel mean, because ViTs are more robust to OOD inputs than CNNs.
- *Relevance:* Supports the claim that ViTs are less sensitive to the replacement value, but still recommends checking against random.

**[M25] LibraGrad: Balancing Gradient Flow for Universally Better Vision Transformer Attributions.** Mehri, Soleymani Baghshah, Pilehvar. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Mehri_LibraGrad_Balancing_Gradient_Flow_for_Universally_Better_Vision_Transformer_Attributions_CVPR_2025_paper.html (verified=yes)
- *What it does:* Groups metrics into Faithfulness (MIF/LIF perturbation), Completeness Error, and Segmentation AP. Covers 8 architectures and 5 datasets.
- *Relevance:* A recent, broad ViT-attribution evaluation suite. Its metric families map directly onto RQ4.

**[M26] Metric-Driven Attributions for Vision Transformers (MDA).** ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/4e21153e79aff242492146d78d09fcdb-Paper-Conference.pdf (verified=partial: abstract snippet; authors not checked)
- *What it does:* Reports 12 attribution metrics on ImageNet for ViT-B/16, including MAS insertion/deletion with a "density response" alignment penalty.
- *Relevance:* A warning that methods can be optimised directly for a metric. Do not tune the probe's hyperparameters (m, k, layer) on the same images and metric used in the RQ4 comparison.

**[M27] Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?** Jacovi, Goldberg. ACL 2020. https://aclanthology.org/2020.acl-main.386/ (verified=yes)
- *What it does:* Separates faithfulness from plausibility. Faithfulness evaluation should not rely on human judgement or gold labels.
- *Relevance:* Frames RQ1 and RQ4 (faithfulness: agreement with the model, label-free) separately from RQ3 (plausibility and "common sense"). A centre focus can be faithful (the model really uses the centre) yet not plausible.

**[M28] Causal Attribution via Activation Patching (CAAP).** Banayeeanzade et al. arXiv 2603.13652 (2026). https://arxiv.org/html/2603.13652v2 (verified=partial: arXiv; author list not checked)
- *What it does:* A 2026 ViT attribution method that patches internal activations. Evaluated with deletion/insertion, Insertion−Deletion, AUPR and the pointing game. Points out that deletion underestimates redundant patches and that insertion into a blank image is OOD.
- *Relevance:* A recent example of the standard metric suite. Its redundancy argument matters for the probe: shared reading sets may be redundant across regions.

### 1.2 Pitfalls checklist (all sourced above)

| Pitfall | Source | What to do for the probe |
|---|---|---|
| Replacement values create OOD inputs ("missingness bias") | Jain 2022 [M14], Hase 2021 [M15], ROAD [M7] | Use token dropping as the primary operator; pixel mean/blur/ROAD as an appendix robustness check |
| The shape of the removed set leaks class information | ROAD [M7] | Report MoRF and LeRF, use SRG [M18], and compare with random and centre sets **of the same size and similar spatial spread** |
| MoRF and LeRF disagree | ROAD [M7], Blücher [M18] | Headline SRG = LIF − MIF |
| Cumulative removal conflates groups; random attribution is hard to beat | SaCo [M8], ROAR [M6], Cohen 2025 [M24] | Add SaCo-style per-group tests and always show a random band (≥5 seeds) |
| Visual plausibility ≠ model dependence | Adebayo [M12], Jacovi & Goldberg [M27] | Cascading-randomization sanity check; keep RQ3 (plausibility) separate from RQ4 (faithfulness) |
| Centre and large-object priors inflate localization | Zhang 2016 [M9], Score-CAM [M11], Choe 2020 [C5], Fel 2022 [C6], Xu-Darme [M19] | Centre baselines, difficult/off-centre subsets, centre-adjusted scores (§2) |
| Metric hacking via tuning | MDA [M26] | Tune probe hyperparameters on a held-out split |
| Resolution mismatch (pixel vs 16×16 patch) | Chefer [M3] maps are upsampled | Aggregate every method to 196 tokens before ranking |
| Preprocessing crops off-centre objects | (Choe 2020 [C5] uses held-out evaluation; IN-9 [D7] filters by crop) | Transform masks and boxes with the same Resize(256)+CenterCrop(224), or use a no-crop resize for off-centre subsets |

### 1.3 Recommended protocol for RQ4 (probe vs raw attention, rollout, Chefer LRP, GAE, Grad-CAM, IG/gradients, optional AttnLRP and TIS)

**Common unit and target**
- One score vector s ∈ ℝ^196 per image, over the 14×14 patch tokens, for the target ŷ = the original model's top-1. The probe's metric is agreement with the original model, so ground-truth labels are not needed. This also lets you use datasets whose labels do not map to ImageNet.
- Pixel-level methods (IG, gradients): sum the attribution inside each 16×16 patch, and state whether you use signed or absolute values.
- The probe: use a graded score, e.g. how many of the m regions' reading sets contain token t (optionally summed over heads and layers), with ties broken by a fixed random seed. Also report the probe's native set at its own size |S|.
- DeiT-S: make sure it is the non-distilled `deit_small_patch16_224` (CLS token only). DINO: the official linear head reads CLS tokens from the last `n_last_blocks` blocks, plus optionally average-pooled patch tokens (`eval_linear.py`, verified). Check which config ships with the ViT-S/16 head. Removing tokens affects the head only through those CLS tokens unless average pooling is on.

**Removal operator**
- Primary: drop tokens after patch embedding and positional embedding (Jain [M14], TIS [M16], SkipPLUS [M17]). Kept tokens keep their positions; CLS is always kept.
- Secondary (appendix): per-patch mean replacement (SaCo-style), RISE blur insertion, and ROAD noisy linear imputation through Quantus.

**Curves (per model, per dataset; about 1–2k images is typical)**
- Token budgets k ∈ {1, 2, 4, 8, 16, 32, 64, 98, 147}, log-spaced because reading sets are small.
- Insertion/sufficiency (keep top-k) and deletion/comprehensiveness (drop top-k).
- Per point, measure (i) top-1 agreement with the full model, (ii) p(ŷ) and its drop, and (iii) KL(full ‖ reduced).
- Report the AUC of each curve and **SRG = LIF − MIF** (Blücher [M18]).

**Fixed-budget head-to-head**
- ERASER comprehensiveness and sufficiency [M1] at k = |S_probe(x)| per image, so every method is held to the probe's own budget.
- Add references at the same k: random (5 seeds), nearest-centre-k, centre Gaussian or CB-CAM, ground-truth-mask tokens (where masks exist), and the high-norm tokens.

**Graded-score faithfulness**
- SaCo [M8] with K = 10 token groups and per-group token dropping, for methods with continuous scores.

**Sanity**
- Cascading randomization of DeiT-S blocks 12→1 on about 200 images. Report Spearman correlation (with and without absolute values) between original and randomized token scores [M12].

**Statistics**
- Per-image paired differences between methods, image-level bootstrap 95% CIs, Wilcoxon signed-rank with Holm correction. Report N and seeds.

**What not to claim**
- ROAR-style retraining (not done).
- Absolute "faithfulness" numbers comparable across papers (the operators differ).

**Cost**
- Rough count: 9 budgets × 2 orders × about 8 methods × 2 models × 2k images ≈ 0.6M forward passes of a reduced-length ViT-S, plus one explanation pass per method per image.
- I did not benchmark L4 throughput. Gradient methods with 50-step IG dominate the cost; consider 20 steps.

---

## 2. Centre bias: how the literature handles it, and what to do

### 2.1 Paper entries (saliency and centre bias)

**[C1] Learning to Predict Where Humans Look.** Judd, Ehinger, Durand, Torralba. ICCV 2009. https://people.csail.mit.edu/tjudd/WherePeopleLook/Docs/wherepeoplelook.pdf (verified=yes)
- *What it does:* Eye tracking on 1,003 images with 15 viewers. Includes a distance-to-centre feature and a centre baseline. "The model with all features except the distance to the center performs as well as the model based on the distance to the center."
- *Relevance:* The canonical demonstration that a centre prior alone is a strong predictor.

**[C2] What do different evaluation metrics tell us about saliency models?** Bylinskii, Judd, Oliva, Torralba, Durand. IEEE TPAMI 2019 (arXiv 2016). https://arxiv.org/abs/1604.03605, https://doi.org/10.1109/TPAMI.2018.2815601 (verified=yes)
- *What it does:*
  - **Shuffled AUC (sAUC)** samples negatives from other images' fixations, so a centre-only model scores 0.5 and centre-biased models are penalised.
  - **Information Gain (IG)** = mean over fixations of log2(ε+P_i) − log2(ε+B_i) with B = a centre-prior baseline. It measures image-specific saliency beyond dataset bias.
  - Table 3 (MIT300): the centre prior reaches AUC 0.78, sAUC 0.51, IG 0 by construction, NSS 0.92, CC 0.38.
  - Recommends choosing metrics according to whether centre bias should be credited.
- *Relevance:* The two tools to port into RQ3: a shuffled-mask score and a gain over a centre baseline.

**[C3] Top-down Neural Attention by Excitation Backprop.** See [M9]. Centre-pointing baseline and a difficult subset (small objects plus distractors). (verified=yes)

**[C4] Score-CAM.** See [M11]. Removes images where the object covers more than 50% before the energy-based pointing game. (verified=yes)

**[C5] Evaluating Weakly Supervised Object Localization Methods Right.** Choe, Oh, Lee, Chun, Akata, Shim. CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html (verified=yes)
- *What it does:* A **centre-Gaussian baseline** (an isotropic Gaussian at the image centre) as the no-learning reference:
  - ImageNet MaxBoxAcc 52.5, against CAM's mean of 63.5;
  - CUB 59.7 against CAM's 68.8;
  - OpenImages PxAP 45.8 against CAM's 59.1.

  Also argues hyperparameters must be tuned on held-out data.
- *Relevance:* Strong evidence that on ImageNet a centre blob already achieves most of the box localization. Use the same baseline for the probe's reading sets.

**[C6] Harmonizing the object recognition strategies of deep neural networks with humans.** Fel, Felipe, Linsley, Serre. NeurIPS 2022. https://arxiv.org/abs/2211.04533, project https://serre-lab.github.io/Harmonization/ (verified=yes)
- *What it does:*
  - Human-model alignment = Spearman correlation between ClickMe maps and model saliency, normalised by human inter-rater agreement (ρ = 0.66).
  - **Centre-bias control:** the mean ClickMe map (which is centre-biased) used as a "model". It was beaten by only 42 of 84 CNNs.
  - Humans focus on object parts (animal faces; functional parts of tools).
- *Relevance:* A ready-made human-alignment protocol with a built-in centre control (see dataset [D11]).

**[C7] Xu-Darme et al. 2023 (CB-CAM)** [M19] and **Ayyar et al. 2025 (RFEM)** [M20]. Both use the centre-biased CAM as a baseline in plausibility and faithfulness tables. In RFEM, CB-CAM ties Grad-CAM on ΔA^F on ImageNet. (verified=yes / partial)

**[C8] Emerging Properties in Self-Supervised Vision Transformers (DINO).** Caron et al. ICCV 2021. https://arxiv.org/abs/2104.14294 (verified=yes)
- *What it does:* Measures the Jaccard similarity between VOC12-val ground truth and the CLS self-attention thresholded to keep 60% of its mass (Fig. 4 caption; an ablation table in the same paper says 80% for the same numbers). For ViT-S/16: **random weights 22.0**, supervised 27.3, DINO 45.9.
- *Relevance:* (a) A direct precedent for **label-free** object-focus evaluation on VOC12 with an ImageNet model, with no class mapping. (b) A random-weights ViT already scores 22.0, which acts as a built-in prior or centre baseline. (c) It predicts a large DeiT-S vs DINO gap that the probe should reproduce.

**[C9] Hard ImageNet: Segmentations for Objects with Strong Spurious Cues.** Moayeri, Singla, Feizi. NeurIPS 2022 Datasets & Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2022/hash/4120362f930b0b683cd30b71af56fad1-Abstract.html (verified=yes)
- *What it does:* Reports that Hard ImageNet objects are **less centred and occupy much less space** than RIVAL10 objects, and that this goes with more reliance on spurious features. I did not extract the exact statistics.
- *Relevance:* An off-centre, small-object ImageNet subset with masks (see [D8]).

**[C10] Corner Cases: How Size and Position of Objects Challenge ImageNet-Trained Models.** Fatima, Jung, Keuper. arXiv 2505.03569 (2025). https://arxiv.org/abs/2505.03569 (verified=partial: arXiv; venue unknown)
- *What it does:*
  - Per-class "centeredness" score C_c = mean over images and objects of 1 − ‖image centre − object centre‖_∞, and a size score (box area fraction), both from ImageNet-1k **validation** boxes.
  - Finds most classes are highly centred, with objects covering about half the image on average; the bounding-box-centre histogram concentrates at the centre and rarely reaches corners.
  - Builds **Hard-Spurious-ImageNet**: synthetic images with varied object positions and sizes. Models rely more on background when objects are small and off-centre.
- *Relevance:* The most direct ImageNet centredness statistics. Its formula can be reused on the 5k val subset and on Pet trimaps. Availability of the dataset is unknown.

**[C11] ImageNet-Hard: The Hardest Images Remaining from a Study of the Power of Zoom and Spatial Biases in Image Classification.** Taesiri, Nguyen, Habchi, Bezemer, Nguyen. NeurIPS 2023 Datasets & Benchmarks. https://arxiv.org/abs/2304.05538 (verified=yes)
- *What it does:* Across 99 zoom anchors, the centre anchor gives the highest upper-bound accuracy on all six datasets. The centre gap is small for ImageNet (about −1 to −5 points) but large for ImageNet-A (about −25) and ObjectNet (about −23).
- *Relevance:* Quantifies that the ImageNet centre advantage is real but modest at the level of classification accuracy. ObjectNet is centre-biased (its authors also say objects are "highly centralized"), so it is **not** a good off-centre set.

**[C12] ImageNet Large Scale Visual Recognition Challenge.** Russakovsky et al. IJCV 2015. https://arxiv.org/abs/1409.0575 (verified=yes)
- *Statistics:*
  - The average ILSVRC classification/localization object covers **35.8%** of the image area, against 24.1% for PASCAL.
  - The average chance performance of localization (CPL: a random same-class box gives IoU ≥ 0.5) is **20.8%**, against 8.7% for PASCAL. It correlates with object scale at ρ = 0.9.
- *Relevance:* The size and position prior behind centre baselines. CPL is a direct "dataset-prior localization" baseline.

**[C13] Vision Transformers Need Registers.** Darcet, Oquab, Mairal, Bojanowski. ICLR 2024. https://arxiv.org/abs/2309.16588 (verified=yes for the quoted facts)
- *What it does:*
  - High-norm "artifact" tokens have about 10× the norm and make up about 2% of tokens. They appear in middle layers, only in sufficiently large and long-trained models.
  - They sit on low-information patches, **tend to appear near the border** because photos are object-centric, and carry global rather than local information.
  - I recall, but did not re-check, which specific checkpoints (DINO v1, DeiT-S) show them.
- *Relevance:* The project's "high-norm tokens" category is confounded with border/background position. Compare high-norm vs other tokens **at matched radius**.

**[C14] Learning to Adapt to Position Bias in Vision Transformer Classifiers.** Bruintjes, van Gemert. arXiv 2505.13137 (2025). https://arxiv.org/abs/2505.13137 (verified=partial: arXiv; venue unknown, the PDF uses BMVC-style copyright)
- *What it does:* Introduces Position-SHAP to measure how much a dataset rewards positional information. ImageNet is "often but not always" centred ("capture bias").
- *Relevance:* A centre focus can be learned through the positional embeddings. Interpret "centre explains reading sets" as a possibly faithful model property, not only an evaluation artifact.

**[C15] Is CLIP Cross-Eyed? Revealing and Mitigating Center Bias in the CLIP Family.** arXiv 2604.05971 (2026). https://arxiv.org/html/2604.05971v1 (verified=partial: arXiv; authors not checked)
- *What it does:* Shows CLS attention in CLIP ViTs concentrates on the centre region while patch tokens still attend to off-centre objects.
- *Relevance:* A close analogue: CLS-centred attention may coexist with off-centre information in patch tokens. A layer-wise probe (RQ2) could show the same pattern.

### 2.2 How often are ImageNet and Pet objects centred?

- **ImageNet:**
  - Objects are large (35.8% average area; Russakovsky [C12]).
  - Most per-class box centres concentrate at the image centre, with objects covering about half the image (Fatima 2025 [C10]).
  - The centre crop is the best single zoom location for accuracy, but the gap is only a few points (Taesiri 2023 [C11]).
  - Centre-Gaussian box localization reaches 52.5 MaxBoxAcc (Choe [C5]).
  - I found **no published single number** such as "x% of ImageNet val objects have their centre within the central y%". Compute it from the official val boxes (2.2 MB file, §3 [D1]) on the 5k subset, after applying the same Resize+CenterCrop.
- **Oxford-IIIT Pet:** no published centredness statistics found (unknown). Compute them from the trimaps the project already uses: fg centroid distance to the centre, fg area fraction, and the Fatima centeredness score.

### 2.3 Recommended centre-bias handling for RQ3 (concrete)

1. **Centre "explanations" as baselines everywhere.** Use nearest-centre-k tokens (already in use), a centre Gaussian (Choe [C5]), CB-CAM (Xu-Darme [M19]), and the dataset-mean ground-truth-mask prior (the analogue of Fel's mean ClickMe map [C6]). Run them through every plausibility and faithfulness metric.
2. **Centre-adjusted scores.** Report (score_method − score_centre) / (score_oracle − score_centre), or an IG-style log-ratio against the centre prior (Bylinskii [C2]). A method that only reproduces the centre prior scores 0.
3. **Shuffled-mask (sAUC-style) score.** For each image, compute the AUROC of the token scores against that image's foreground mask, and again against **another image's mask** from the same dataset. Report the difference: image-specific object focus beyond the dataset prior. This is the direct port of sAUC.
4. **Radius-stratified enrichment.** Split the 14×14 grid into rings by distance to the centre. Within each ring, compare the fraction of selected tokens that are foreground with the ring's foreground fraction. Average the enrichment across rings (Mantel–Haenszel-style). This tests object focus at fixed position and complements the centre-matched deletion control (about 73% of the effect survives).
5. **Stratify by object geometry.** Bin by mask-centroid distance (e.g. tertiles) and area fraction. Add a "difficult" subset in the style of Zhang [M9]: area < 1/4, centroid outside the central region, optionally with distractor objects (ImageNet-S multi-class masks allow this).
6. **Intervention: shifted crops.** For ImageNet-val images with boxes (or ImageNet-S masks), take several 224×224 crops of the resized image that place the same object at different positions (centre, the four quadrants), discarding crops that truncate the object by more than 50%, as the IN-9 filter does [D7]. If the reading set follows the object, it is object-driven; if it stays central, it is position-driven. This is the cleanest causal control and costs only extra forward passes.
7. **Position-matched necessity.** Extend the existing centre-matched deletion: for foreground, background and high-norm categories, draw random controls from the same radius rings (Darcet [C13] notes that outliers sit near borders).

---

## 3. Datasets for RQ3

Column legend: "Zero-shot?" means the ImageNet-1k head (DeiT-S head, or DINO linear head) gives a meaningful prediction without training.

### 3.1 Comparison table

| # | Dataset | Images used for eval | Annotation | Label mapping to IN-1k | Zero-shot? | Off-centre? | Download (size) | RQ3 use |
|---|---|---|---|---|---|---|---|---|
| D1 | ImageNet-1k val + official boxes | 50k (project uses 5k) | class, boxes for all val images | direct | yes | mostly centred (C10, C12); stratify | boxes 2.2 MB (image-net.org) | baseline + off-centre strata + shifted crops |
| D2 | ImageNet-S (919) val | 12,419 (+27,423 test, masks on server) | semantic masks, multi-class per image | direct (919 synsets; 5 merged) | yes | unknown; filter by mask | annotation zip from GitHub release (size unknown); images = your ImageNet val | object focus beyond centre; multi-object; difficult subset |
| D3 | ImageNet-Segmentation (Guillaumin) | 4,276 | binary fg mask | no label in Chefer's .mat; 445 categories | label-free only | unknown | `gtsegs_ijcv.mat`, 1.2 GB | reproduce the Chefer segmentation test for RQ4 |
| D4 | PartImageNet | 24,095 (158 classes) | part masks (11 super-categories) | direct (158 IN classes) | yes | unknown (single-object filtered) | HF `turkeyju/PartImageNet`, 7.18 GB | part-level plausibility |
| D5 | PartImageNet++ | 100k (1000 classes × 100) | 406,364 part masks, 3,308 part classes | direct | yes | unknown | HF `lixiao20/PartImageNetPP`, 515 MB JSON; **images from IN-1k train needed** | parts at full scale (needs train images) |
| D6 | CUB-200-2011 | 11,788 (5,794 test) | 15 part keypoints, bbox, 312 attributes, masks | partial (some species ↔ IN bird classes, unverified); otherwise needs a probe | partly | unknown | 1.2 GB + 39.3 MB masks (CaltechDATA) | part keypoints (head, beak, eyes) |
| D7 | ImageNet-9 / Backgrounds Challenge | 4,050 test (450/class) × 8 variants | GrabCut fg/bg composites | 9 super-classes via WordNet; 1000-class models supported | yes | inherits ImageNet | GitHub release (size unknown) | background reliance (only-FG, mixed-rand, no-FG) |
| D8 | Hard ImageNet | 750 val (19,097 train) | object masks | direct (15 synsets) | yes | **less centred, smaller** (C9) | Box zip (annotations; size unknown) + your ImageNet val | object focus beyond centre under spurious context |
| D9 | RIVAL10 | ~26k (10 classes from 20 synsets) | object masks + 18 attribute masks | 2 synsets per class | yes (with mapping) | unknown | website (size unknown) | attribute/part plausibility |
| D10 | Salient ImageNet | 52,521 (232 classes); extended 6.4M masks | **model-derived** soft core/spurious masks | direct | yes | unknown | Box zip (size unknown) | core vs spurious reliance (masks come from a robust model, not humans) |
| D11 | ClickMe (Harmonization) val | ~17.4k (estimated) | human importance maps | direct (1000-way labels) | yes | maps are centre-biased (C6) | 17 TFRecord shards × ~27 MB ≈ 465 MB (GCS) | human alignment with centre control |
| D12 | Clicktionary | 200 IN-val | human importance maps (more reliable) | direct | yes | unknown | unknown | small, high-quality human alignment |
| D13 | ImageNet-X | 50k val + 12k train | 16 human factor labels (pose/location, smaller, larger, partial view, multiple objects, background…) | direct | yes | label "pose/positioning: different pose **or placed in different location**" | toolkit (route not verified) | stratify ImageNet val by position, size, background |
| D14 | PASCAL VOC 2012 seg | 1,449 val (2,913 trainval) | semantic and instance masks, 20 classes | none direct | label-free only (DINO precedent C8) | VOC07 centre-pointing baseline 69.5% (so fairly centred) | 2 GB tar | label-free object focus; multi-object |
| D15 | COCO 2017 val | 5,000 | instance masks, 80 classes | none direct | label-free only | **much less centred** (centre-pointing 27.7% on COCO14 val) | 778 MiB images + annotations | off-centre, small, multi-object stress test (label-free) |
| D16 | Waterbirds | CUB × Places; 4,795 train | binary label, controlled background | none | needs training | unknown | not verified | controlled background reliance |
| D17 | Spawrious | ~152k SD-generated | class × background | dog breeds (mapping not verified) | needs training | unknown | not verified | controlled spurious background |
| D18 | Hard-Spurious-ImageNet | unknown | synthetic position/size/background | IN-1k | yes | **by design** | availability unknown | ideal centre control if released |

### 3.2 Dataset entries

**[D1] ImageNet-1k (ILSVRC2012) validation set and its bounding boxes.**
- Size: 50,000 val images, 1000 classes (image-net.org). Box annotations for the val set: "Validation bounding box annotations (all tasks), 2.2 MB" (official download page). Val images are 6.3 GB.
- Mapping: direct.
- License: ImageNet terms of access (non-commercial research); registration needed on image-net.org. The localization version is also on Kaggle as `LOC_val_solution.csv`.
- URL: https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php (verified=yes for sizes)
- Off-centre: see §2.2. Mostly centred and large.
- RQ3 use: compute centredness for the 5k subset; define off-centre/small/difficult strata; generate shifted crops. Zero cost, since the project already has the images.

**[D2] ImageNet-S (Large-scale Unsupervised Semantic Segmentation).** Gao, Li, Yang, Cheng, Han, Torr. TPAMI 2022. https://github.com/LUSSeg/ImageNet-S, paper https://arxiv.org/abs/2106.03149 (verified=yes)
- Size: ImageNet-S919 has 12,419 val and 27,423 test images with pixel-level masks (test masks are held on an online server). Subsets: S300 val 4,097; S50 val 752. 9,190 training images also have masks. The val/test masks were made on **images from the ImageNet validation set** ("we annotate 40k images in the validation set of ImageNet").
- Annotation: semantic masks stored as RGB PNG; class id = R + 256·G; "other category" = 0, ignore = 1000. Multi-label images use ImageNet-ReaL-style re-labelling.
- Mapping: direct to ImageNet synsets, with 919 of the 1000 classes; 5 merges (e.g. n04356056 → n04355933).
- Download: `datapreparation_val.py` extracts images from your ImageNet copy. `datapreparation_anno.sh` downloads `ImageNetS919-*.zip`, `ImageNetS300-*.zip` and `ImageNetS50-*.zip` from GitHub releases. **Zip sizes and license are unknown** (the LICENSE page timed out).
- Off-centre: no statistics published (unknown). Filter by mask centroid.
- RQ3 use: the best "ImageNet object masks on held-out images" source. Supports zero-shot evaluation, object vs "other object" focus in multi-object images, centre-stratified analyses and difficult subsets. Several thousand of its val images are probably in, or can replace, the current 5k subset.

**[D3] ImageNet-Segmentation (ImageNet Auto-annotation ground truth).** Guillaumin, Küttel, Ferrari, "ImageNet Auto-Annotation with Segmentation Propagation", IJCV 2014. Data page https://calvin-vision.net/datasets/imagenet-auto-annotation/ (verified=yes for size and file)
- Size: 4,276 images with ground-truth figure-ground segmentations, collected on AMT and cleaned (**1.2 GB**, `gtsegs_ijcv.mat`). Mirror used by others: `http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat`. Chefer et al. state 445 categories.
- Mapping: the .mat as used by Chefer contains only images and binary masks, no labels. Whether all 445 categories are ILSVRC-1k classes, and whether the images are train or val, is **unknown**.
- License: unknown.
- Off-centre: unknown.
- RQ3/RQ4 use: reproduce the Chefer segmentation test (pixel accuracy, mIoU, mAP) so RQ4 numbers are comparable with the literature. Report it next to a centre-Gaussian baseline. Pitfall: the loader resizes to 224×224 without cropping.

**[D4] PartImageNet: A Large, High-Quality Dataset of Parts.** Ju He et al. ECCV 2022. https://arxiv.org/abs/2112.00933, https://github.com/TACJu/PartImageNet, HF https://huggingface.co/datasets/turkeyju/PartImageNet (verified=yes)
- Size: 24,095 images, 158 ImageNet classes in 11 super-categories. Parts per super-category, e.g. Quadruped: head, body, foot, tail; Bird: head, body, wing, foot, tail; Car: body, tire, side mirror. COCO-style annotations.
- Splits: PartImageNet_Seg (same classes in train/val/test) and PartImageNet_OOD (109/19/30 classes; 16,540 / 2,957 / 4,598 images). Images with more than one qualifying object were discarded.
- Source: "we collect images from the ILSVRC-12 dataset". A secondary source (D-Feat Occlusions, arXiv 2504.06432) states that PartImageNet **overlaps with the ImageNet training set**, so DeiT-S and DINO saw these images in training. Verified=partial for the split.
- License: "All the images conform to licensing for research purposes"; dataset license unknown.
- Download: the HF dataset holds 7.18 GB, which appears to include the images.
- Mapping: direct.
- Off-centre: unknown.
- RQ3 use: part-level plausibility. Which parts (head, body, tail…) do reading sets select, compared with part area? Head enrichment can be compared with the ClickMe finding that humans focus on faces [C6]. Caveat: training-set images (memorisation; confident predictions). Say so, and maybe check that conclusions also hold for ImageNet-S val.

**[D5] PartImageNet++ (PIN++).** Li, Liu, Dong, Qin, Hu. ECCV 2024. https://arxiv.org/abs/2407.10918, HF https://huggingface.co/datasets/lixiao20/PartImageNetPP, code https://github.com/LixiaoTHU/PartImageNetPP (verified=yes)
- Size: 100k images (100 per class for all 1000 IN-1k classes), 3,308 part categories on the HF card (the paper says 3,310), 406,364 part masks.
- Source: **all images are from the IN-1k training set** (paper §3.2), split 9:1 into train/val.
- Download: HF holds JSON annotations, 515 MB total. Images must come from your ImageNet train copy (138 GB).
- License: blank on the HF card (unknown).
- Mapping: direct.
- RQ3 use: part analysis at full class coverage. Too heavy for Colab unless only the 100 images per class are fetched (fetching single images still needs ImageNet train access). Lower priority than D4.

**[D6] CUB-200-2011.** Wah, Branson, Welinder, Perona, Belongie. Caltech tech report CNS-TR-2011-001. https://www.vision.caltech.edu/datasets/cub_200_2011/, data https://data.caltech.edu/records/65de6-vp158, masks https://data.caltech.edu/records/w9d68-gec53 (verified=yes)
- Size: 11,788 images (5,994 train / 5,794 test), 200 species, 15 part locations, 312 binary attributes, 1 box per image. Segmentations (Farrell) 39.3 MB. Images and annotations 1.2 GB.
- License: "use … restricted to non-commercial research and educational purposes".
- **Warning from the dataset page:** images overlap with ImageNet. Mahajan et al. (ECCV 2018) state that about 5% of CUB val images appear in IN-1k train.
- Mapping: species-level labels are not IN-1k classes. Some species correspond to IN-1k bird classes, but I did not verify a mapping list. Species-level predictions need a linear probe (DINO features suit this) or fine-tuning.
- Off-centre: unknown.
- RQ3 use: keypoint-level plausibility (beak, crown, eyes, wings, tail…) without needing species labels. The reading sets and agreement metric are label-free, so CUB can be used zero-shot for "does the probe read the head?" on images the model classifies as some bird class. Low priority.

**[D7] ImageNet-9 / Backgrounds Challenge.** Xiao, Engstrom, Ilyas, Madry, "Noise or Signal: The Role of Image Backgrounds in Object Recognition", ICLR 2021. https://arxiv.org/abs/2006.09994, https://github.com/MadryLab/backgrounds_challenge (verified=yes)
- Size: 9 super-classes (dog 116 sub-classes, bird 52, vehicle 42, reptile 36, carnivore 35, insect 27, instrument 26, primate 20, fish 16). 5,045 training and **450 test images per class**. Variants: Original, Only-BG-B, Only-BG-T, No-FG, Only-FG, Mixed-Same, Mixed-Rand, Mixed-Next.
- Construction: foreground by GrabCut inside the single ImageNet box. Crops that remove more than 50% of the box are filtered out. The test set was manually cleaned.
- Evaluation: `in9_eval.py` accepts a **1000-class ImageNet model** directly (zero-shot through super-class mapping).
- Source of test images (train vs val): **unknown**.
- Download: GitHub release (archive size unknown). License: unknown.
- RQ3 use: background reliance.
  - Fraction of reading-set tokens on foreground in Mixed-Rand vs Original.
  - Does the probe's sufficiency hold on Only-FG?
  - Foreground masks follow from Only-FG vs Original.
  - The BG-Gap (Mixed-Same − Mixed-Rand) is the model-level reference.

**[D8] Hard ImageNet.** Moayeri, Singla, Feizi. NeurIPS 2022 Datasets & Benchmarks. https://mmoayeri.github.io/HardImageNet/, https://github.com/mmoayeri/HardImageNet (verified=yes)
- Size: 15 IN-1k synsets with strong spurious cues: dog sled, howler monkey, seat belt, ski, sunglasses, swimming cap, balance beam, horizontal bar, patio, hockey puck, miniskirt, keyboard space bar, volleyball, baseball player, snorkel. 19,097 train and 750 val images ("quintuply validated"), same train/val split as ImageNet, with object masks and train-image spurious-strength rankings.
- Download: Box link (annotations only; you need ImageNet, and only the val images for evaluation). A Box link for just these classes' images is also given. Archive size unknown. License unknown.
- Benchmark: ablation accuracy drop, Relative Foreground Sensitivity (noise in fg vs bg), and saliency alignment (Grad-CAM IoU with masks).
- Off-centre: objects are **less centred and smaller** than RIVAL10 (paper; numbers not extracted).
- Mapping: direct. Zero-shot.
- RQ3 use: the cheapest real-image test of "object focus beyond the centre" and of background reliance the model actually uses. 750 val images take minutes to process.

**[D9] RIVAL10.** Moayeri, Pope, Balaji, Feizi, "A Comprehensive Study of Image Classification Model Sensitivity to Foregrounds, Backgrounds, and Visual Attributes", CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Moayeri_A_Comprehensive_Study_of_Image_Classification_Model_Sensitivity_to_Foregrounds_CVPR_2022_paper.html, site https://mmoayeri.github.io/RIVAL10/index.html (verified=yes for content)
- Size: about 26k images, 10 classes (CIFAR-10-like) built from 20 IN-1k synsets (2 per class). Object masks plus 18 attribute masks (beak, ears, floppy-ears, horns, mane, tail, wheels, wings, …).
- Split source, download size and license: unknown.
- RQ3 use: attribute/part-level plausibility on IN-1k classes. An alternative to PartImageNet if its images turn out to be val images (unknown).

**[D10] Salient ImageNet.** Singla, Feizi. ICLR 2022. https://arxiv.org/abs/2110.04301, site https://salient-imagenet.cs.umd.edu/ (verified=yes)
- Size: 232 classes × 5 neural features = 1,160 features (160 spurious). 52,521 images (about 226 per class) with core/spurious soft masks from neural activation maps of an adversarially robust ResNet-50. The extended version (Core Risk Minimization, Singla, Moayeri, Feizi, arXiv 2203.15566) has 5,000 features and 6,405,835 masks.
- Download: Box zip or the GitHub loaders (size unknown). License unknown. Images come from top-activating samples, likely ImageNet train (unverified).
- RQ3 use: does the probe read "core" vs "spurious" regions? Caveat: the masks come from another model, not from humans, so they are weak ground truth for "common sense". Low priority.

**[D11] ClickMe human feature-importance maps (via Harmonization).**
- Linsley, Shiebler, Eberhardt, Serre, "Learning what and where to attend", ICLR 2019. https://arxiv.org/abs/1805.08819
- Fel et al., NeurIPS 2022 [C6]. Library https://github.com/serre-lab/Harmonization (MIT license for the code).
- verified=yes, except where noted below.
- Size: 472,946 ClickMe maps on 196,499 unique ILSVRC12 images (train and val) from 1,235 players. Harmonization uses the maps on **ImageNet val** for evaluation.
- Validation loader: 17 TFRecord shards at `https://storage.googleapis.com/serrelab/prj_harmonization/dataset/click-me/val/val-{0..16}.tfrecords`. I checked two shards with HEAD requests: 27.6 MB and 27.1 MB. I downloaded shard 16 and counted **1,024 records**, so roughly 17 × 1,024 ≈ 17.4k images. This is an estimate: the other shards' counts were not checked.
- Format: images stored at 256×256 (resized to 224 by the loader); heatmaps 256×256, blurred.
- Labels: 1000-way (direct mapping).
- Data license: unknown (code is MIT).
- Protocol: Spearman correlation between the model's map and the human map, divided by inter-rater agreement (0.66). Centre control: the mean ClickMe map.
- Off-centre: maps are centre-biased; the control is part of the protocol.
- RQ3 use: "human common sense" alignment of reading sets and baselines, scored against the centre control. Humans favour faces and functional parts, which bridges to the part analysis.
- Practical: parse with the `tfrecord` pip package or TensorFlow on Colab; heatmaps can be pooled to 14×14.

**[D12] Clicktionary.** Linsley et al. (cited in Fel 2022). 200 ImageNet-val images with human-human game maps, more reliable than ClickMe (Fel 2022, verified=yes for this description). Download route unknown. Low priority.

**[D13] ImageNet-X.** Idrissi et al., "ImageNet-X: Understanding Model Mistakes with Factor of Variation Annotations", ICLR 2023. https://arxiv.org/abs/2211.01866 (verified=yes for content)
- Size: human annotations for **all 50k IN-1k val images** plus 12k train images.
- Factors (16), assigned against class prototypes: pose (defined as "different pose **or is placed in different location** within the image"), partial view, object blocking, person blocking, multiple objects, smaller, larger, brighter, darker, background, color, shape, texture, pattern, style, subcategory.
- Download route and license: not verified (a toolkit is released).
- RQ3 use: free human labels to stratify the existing 5k val images (e.g. "smaller", "multiple objects", "pose/location", "background").

**[D14] PASCAL VOC 2012 segmentation.** https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/ (mirror https://pjreddie.com/projects/pascal-voc-dataset-mirror/) (verified=yes)
- Size: segmentation trainval 2,913 images (1,464 train / 1,449 val), 6,929 objects, 20 classes. Train/val tar about 2 GB (1.9 GB on the mirror).
- Mapping: no direct IN-1k mapping. The standard pointing game uses fine-tuned models [M10].
- Zero-shot route: label-free, as in DINO's Jaccard evaluation [C8]. Compare reading sets with the union of object masks, and use agreement-with-original as the faithfulness target.
- Off-centre: the VOC07 centre-pointing baseline is 69.5%, so VOC is fairly centred. License: VOC terms (Flickr images; not checked).

**[D15] COCO 2017 val.** https://cocodataset.org (verified=yes for license and size)
- Size: 5,000 images (val2017 zip 777.8 MiB per the NCHC mirror), 80 classes, instance masks.
- License: annotations CC BY 4.0; images under Flickr terms.
- Mapping: no direct IN-1k mapping. Use label-free, or fine-tune a multi-label head (TorchRay).
- Off-centre: the centre-pointing baseline is only 27.7% (COCO14 val, Zhang 2016), so COCO is much less centre-biased.
- RQ3 use: a qualitative or label-free off-centre stress test. The ImageNet prediction may refer to any object in the scene, so restrict to images whose predicted IN class plausibly matches one annotated COCO category (manual mapping needed; effort).

**[D16] Waterbirds.** Sagawa, Koh, Hashimoto, Liang, "Distributionally Robust Neural Networks for Group Shifts", ICLR 2020. https://arxiv.org/abs/1911.08731 (verified=yes for construction)
- Construction: CUB birds cut with masks and pasted onto Places backgrounds (ocean or natural lake vs bamboo or broadleaf forest). Binary waterbird/landbird label with a 95% correlated background in train. 4,795 training examples; the smallest group has 56.
- Needs training a binary head (linear probe). Download route not verified (commonly via the group_DRO repo or WILDS).
- Value for this project: low. It is a controlled background-reliance test, but needs training, and ImageNet-1k features already contain both birds and backgrounds.

**[D17] Spawrious.** Lynch, Dovonon, Kaddour, Silva. arXiv 2303.05470 (2023); https://github.com/aengusl/spawrious (verified=partial: venue not checked)
- Size: about 152k images generated with Stable Diffusion v1.4: 4 dog-breed classes × 6 background locations, in O2O/M2M × Easy/Medium/Hard splits.
- Needs training (4-way). Breed-to-IN-1k mapping not verified. License unknown.
- Value: low for a 4-week window.

**[D18] Hard-Spurious-ImageNet.** Fatima et al. 2025 [C10]. Synthetic images with varied object positions, sizes and backgrounds, derived from IN-1k. Availability, size and license unknown.
- Value: it would be the ideal centre control if released. Otherwise build your own shifted crops (§2.3 item 6).

---

## 4. Recommended dataset plan (4-week window, Colab L4, DeiT-S and DINO ViT-S frozen)

All recommended items are **zero-shot** with the existing ImageNet-1k heads. No fine-tuning or linear probe is needed. Effort counts working days for one person and includes data wrangling, runs and figures. These are my estimates, not measured.

| Rank | Dataset(s) | Why (RQ3 angle) | Training? | Effort |
|---|---|---|---|---|
| **1** | **ImageNet-S919 val** (12,419 images, masks) + ImageNet val boxes + ImageNet-X labels + **shifted-crop intervention** on IN-val | Held-out ImageNet images with real masks; object focus beyond centre through stratification, shuffled-mask and radius-stratified enrichment, plus a causal position intervention; multi-object images test "right object vs other object" | none | 4–6 days (1 data/mapping, 1 shifted crops, 2–3 runs and analysis, 1 figures) |
| **2** | **Hard ImageNet val** (750 images, masks) | Objects documented as less centred and smaller, with strong spurious context. Tests whether reading sets follow the object or the context the model actually uses, and whether necessity tests detect background reliance | none | 1–2 days |
| **3** | **ClickMe val** (about 17.4k IN-val images, human maps; about 465 MB) | "Human common sense": Spearman alignment normalised by inter-rater agreement, with Fel's mean-map centre control. Humans favour faces and functional parts (a qualitative bridge to parts). Use a 2–5k subset if time is short | none | 2–3 days (TFRecord parsing, pooling to 14×14, alignment) |
| **4** | **PartImageNet** (Seg split; 7.18 GB on HF) | Part-level plausibility: enrichment of head/body/tail tokens in reading sets relative to part area; animals vs vehicles. Caveat: ImageNet train images | none | 2–3 days |
| optional | ImageNet-9 (Only-FG, Mixed-Rand, Mixed-Same) | Background reliance with a standard model-level reference (BG-Gap) | none | 1–2 days |
| optional | ImageNet-Segmentation (`gtsegs_ijcv.mat`, 1.2 GB) | Only to report the Chefer segmentation test for RQ4 comparability (with a centre baseline) | none | 1 day |
| not recommended now | CUB (species), Waterbirds, Spawrious, VOC/COCO class-specific | Need training or label mapping; marginal benefit over D2/D8 | linear probe or fine-tune | 3–6 days each |

**Suggested order**
- Week 1: protocol code (token dropping, curves, SRG, SaCo, sanity) on the existing 5k IN-val and Pet sets, to settle RQ4.
- Week 2: Rank 1 (ImageNet-S plus shifted crops).
- Week 3: Ranks 2 and 3.
- Week 4: Rank 4, figures and writing.

Keep Pet as the existing in-domain set, but report it with the centre-adjusted and radius-stratified scores from §2.3.

---

## 5. Items I could not verify (to check before citing)

- Venues: Blücher et al. (TMLR assumed from OpenReview); Fatima et al. 2025, Bruintjes & van Gemert 2025, Cohen et al. 2025, Ayyar et al. 2025, CAAP 2026 and the "Is CLIP Cross-Eyed?" 2026 paper (arXiv only, venue unknown); Xu-Darme 2023 (ACM DOI; conference name assumed); Spawrious (venue unknown).
- Dataset facts:
  - ImageNet-S license and annotation zip sizes.
  - ImageNet-Segmentation split (train/val), class mapping and license.
  - IN-9 test-image source and archive size.
  - Hard ImageNet, RIVAL10 and Salient ImageNet archive sizes, licenses and image sources.
  - PartImageNet train-set origin (secondary source only) and license.
  - ClickMe val image count (estimated from one shard) and data license.
  - CUB species ↔ IN-1k mapping.
  - Waterbirds download route; Spawrious breeds; ImageNet-X download route and license; Clicktionary download route.
- Method facts:
  - Whether AttnLRP's `lxt` supports timm DeiT-S and DINO ViT-S out of the box.
  - Whether Chefer's original repo supports DeiT-S (the GAE rule is easy to re-implement either way).
  - DINO ViT-S/16 linear-head config (`n_last_blocks`, avgpool): confirm in the DINO README.
  - Which checkpoints show Darcet-style high-norm artifacts.
- No published centredness statistic for Oxford-IIIT Pet was found: compute it.
