# A. Methods, positioning and baselines — annotated bibliography

Project: a MiTA-style probe (landmark queries, shared top-k "reading sets", optional summary slots) that interprets pretrained ViTs (DeiT-S/16, DINO ViT-S/16; MiTA-DeiT-S as a control). DSAA 2012 workshop paper, due 2026-12-11.
Survey compiled 2026-10-29 with Exa search and fetch.

**Verification legend.** `verified=yes` means I fetched the arXiv, venue or proceedings page during this session and confirmed the title and authors. Where I also saw the venue on a page (CVF, PMLR, NeurIPS proceedings, ACL Anthology, ICLR proceedings, or the official repo README), it is written plainly. A dagger (†) means the **venue/year comes from memory** and was not re-checked on a page; the paper itself is verified. `verified=partial` means only some fields were confirmed (noted inline). I have not invented any number. Numbers quoted below appear verbatim on the fetched pages.

---

## 0. TL;DR

1. **The biggest novelty risk is the MiTA paper itself** (Wen et al., arXiv 2602.01219, v6 Sep 2026). Here is what it already does:
   - It frames attention as an N-width fast-weight MLP.
   - It visualises each expert's gathered key-value pairs and routed queries (Fig. 3).
   - It reports an "**emergent token-pruning effect**": in later layers most tokens are not selected, and selection "concentrates on class-relevant regions" (Fig. 4, on MiTA-trained models).
   - It measures per-layer overlap (mIoU) between gathered key-values and routed queries (Fig. 8).
   - In App. C / Fig. 9 it swaps standard attention for MiTA **at inference without training** on DeiT-T/S models trained with standard attention. This "retains over 95% of the original performance".
   - Its ablation finds "performance is more sensitive to k than to m". This is the trained-model analogue of our finding that sparsity, not sharing, is the main cost.

   We must not claim any of these as new. Our contribution has to be the **interpretability methodology** applied to official pretrained checkpoints:
   - agreement-based sufficiency;
   - decomposing the cost into sparsity, sharing and compression;
   - necessity tests with random and **centre-matched** controls;
   - quantitative plausibility beyond a centre prior. Our negative DeiT-S result directly qualifies MiTA's qualitative "class-relevant" claim.
   - head-to-head comparison against standard ViT explanation methods under the same tests.
2. **Several earlier analyses already reported that deep ViT attention shares a few tokens**:
   - "query-irrelevant" deep-layer attention (Ma et al. 2022, FCViT);
   - attention collapse (DeepViT);
   - query-independent non-local attention (GCNet);
   - massive-token attention sinks (Darcet 2024; Lu et al. 2025).

   So RQ2's "deep layers share few tokens" is **confirmatory**. Frame it as a quantification through the probe, not a discovery.
3. **The "neuron = patch" premise has a known weak point.** Meister et al. (ACL 2021) argue that sparse attention is not more interpretable because intermediate tokens are not the inputs. Darcet et al. (ICLR 2024) show high-norm tokens lose local and position information. The project needs a token-identity check to justify reading deep-layer neurons as patches.
4. **Recommended RQ4 baselines** (details in §8):
   - raw CLS attention (per layer, plus the last layer);
   - Attention Rollout;
   - Chefer et al. ICCV 2021 (GAE, gradient-weighted rollout);
   - AttnLRP (LXT), with Chefer CVPR 2021 as the fallback;
   - one gradient-free token-dropping perturbation method (TIS or ViT-CX).

   Grad-CAM is an optional "weak but popular" reference.
5. **DINO caveat.** The official DINO linear head for ViT-S/16 reads the **concatenation of [CLS] from the last 4 blocks** (`n_last_blocks=4`, `avgpool_patchtokens=False`; eval_linear.py, verified). Methods that read the last-layer CLS (raw attention, rollout, GAE) need adapting. Gradient and LRP methods handle it automatically.

---

## 1. Attention as fast weights, an MLP, key-value memory or associative memory

**1.1 MiTA: Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights** (v1–v3 title: *MiTA Attention: Efficient Fast-Weight Scaling via a Mixture of Top-k Activations*)
- Authors: Qishuai Wen, Zhiyuan Huang, Xianghan Meng, Wei He, Chun-Guang Li (BUPT).
- Venue: arXiv 2602.01219 (v1 Feb 2026, v6 27 Sep 2026). The project brief says NeurIPS 2026; the arXiv page I saw does not state a venue.
- URL: https://arxiv.org/abs/2602.01219 · code https://github.com/QishuaiWen/MiTA · verified=yes
- What it does:
  - Writes softmax attention as a two-layer MLP: scaled keys are the input weights, exp is the activation, and normalised values are the output weights. The width equals N.
  - MiTA = (i) m landmark queries built by **average pooling over uniform windows**, (ii) top-k keys per landmark forming "deformable routed experts", (iii) landmark values forming a compressed "shared expert", and (iv) each query routed to its argmax landmark (s=1).
  - Reports these findings:
    - an emergent token-pruning effect concentrated on class-relevant regions (Fig. 4, MiTA-trained);
    - modest per-layer overlap between an expert's keys and its routed queries (Fig. 8);
    - routing matters more than compression;
    - "performance is more sensitive to k than to m";
    - standard→MiTA inference swap keeps >95% performance (App. C, Fig. 9);
    - average pooling beats learned landmark extractors, and ToMe-based landmarks help.
- Relevance: this is the source of our probe and the closest prior work. See §7.

**1.2 Linear Transformers Are Secretly Fast Weight Programmers** — Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber. ICML 2021 (PMLR 139). https://arxiv.org/abs/2102.11174 · verified=yes
- What: proves that linearised attention is formally equivalent to 1990s fast-weight programmers, where keys and values are written as outer-product updates. Proposes a delta-rule update.
- Relevance: the canonical citation for the fast-weight view. Note it concerns *linear* attention. The softmax "N-width MLP" reading is spelled out explicitly in MiTA (and earlier in Sukhbaatar 2019 and Gupta 2021).

**1.3 Hopfield Networks is All You Need** — Hubert Ramsauer et al. (Hochreiter group). ICLR 2021†. https://arxiv.org/abs/2008.02217 · verified=yes
- What: modern continuous Hopfield networks whose one-step update equals softmax attention. Stored patterns are keys, and retrieval is a softmax-weighted average.
- Relevance: the associative-memory reading of a reading set: the top-k keys are the "stored patterns" a query retrieves. Useful for discussing metastable versus single-pattern retrieval, which links to sparsity.

**1.4 Transformer Feed-Forward Layers Are Key-Value Memories** — Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy. EMNLP 2021†. https://arxiv.org/abs/2012.14913 · verified=yes
- What: shows each FFN hidden unit acts as a key (a textual pattern detector) paired with a value (a distribution over outputs). Interprets single neurons.
- Relevance: the reverse direction (FFN → key-value). It is the methodological precedent for neuron-level interpretation, which we transfer to attention, where the "neurons" are patches.

**1.5 Augmenting Self-attention with Persistent Memory** — Sainbayar Sukhbaatar, Edouard Grave, Guillaume Lample, Hervé Jégou, Armand Joulin. arXiv 2019. https://arxiv.org/abs/1907.01470 · verified=yes
- What: replaces the FFN with attention over learned persistent key-value vectors, making FFN and attention the same operator.
- Relevance: an early explicit "FFN = attention over fixed memory" equivalence. Cite it so that the equivalence is not presented as ours.

**1.6 Memory-efficient Transformers via Top-k Attention** — Ankit Gupta, Guy Dar, Shaya Goodman, David Ciprut, Jonathan Berant. SustaiNLP workshop @ EMNLP 2021. https://arxiv.org/abs/2106.06899 · verified=yes
- What: per-query top-k attention as a drop-in replacement for vanilla attention in *pretrained* models **without corrective training**. Also casts FFN layers into the query-key-value framework.
- Relevance: a direct precedent for our "execute a pretrained model only through top-k sets" (text and LRA rather than vision, and per-query rather than shared). It shows that training-free top-k is known to work.

**1.7 Birth of a Transformer: A Memory Viewpoint** — Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Hervé Jégou, Léon Bottou. NeurIPS 2023†. https://arxiv.org/abs/2306.00802 · verified=yes
- What: studies how transformers store knowledge as associative memories in weight matrices and in context (induction heads).
- Relevance: cited by MiTA. A memory-viewpoint background citation.

**1.8 Attention Approximates Sparse Distributed Memory** — Trenton Bricken, Cengiz Pehlevan. NeurIPS 2021†. https://arxiv.org/abs/2111.05498 · verified=yes
- What: shows softmax attention approximates Kanerva's Sparse Distributed Memory under conditions met in GPT-2.
- Relevance: a theoretical reason to expect effectively sparse reading sets (SDM reads from a small neighbourhood), which supports RQ1.

**1.9 The Dual Form of Neural Networks Revisited: Connecting Test Time Predictions to Training Patterns via Spotlights of Attention** — Kazuki Irie, Róbert Csordás, Jürgen Schmidhuber. ICML 2022†. https://arxiv.org/abs/2202.05798 · verified=yes
- What: a linear layer trained by gradient descent equals unnormalised key-value attention over all training patterns. They inspect these "spotlights of attention" to interpret predictions.
- Relevance: the **closest precedent for using the attention↔MLP duality *for interpretability*** (reverse direction: MLP → attention over training data). Worth one sentence in related work.

**1.10 Attention as a Hypernetwork** — Simon Schug, Seijin Kobayashi, Yassir Akram, João Sacramento, Razvan Pascanu. ICLR 2025 (oral). https://proceedings.iclr.cc/paper_files/paper/2025/hash/abfa542f546df3c6c35695ec8d5bf4b9-Abstract-Conference.html · verified=yes
- What: multi-head attention is a linear hypernetwork whose latent code (the attention scores across heads, per key-query pair) configures a value network. The code is interpretable and structured.
- Relevance: another "attention = dynamically generated network" view used for interpretation. It supports head-level analysis (Idea 2).

**1.11 A Mathematical Framework for Transformer Circuits** — Nelson Elhage, Neel Nanda, Catherine Olsson, et al. (Anthropic). Transformer Circuits Thread, Dec 2021. https://transformer-circuits.pub/2021/framework/index.html · verified=yes
- What: decomposes attention heads into QK (where to read) and OV (what to write) circuits. The residual stream is a sum of component outputs.
- Relevance: our "key = input weight, value = output weight" maps onto QK and OV. Use the OV circuit to weight neurons by what they write (Idea 1).

**1.12 Every Component is a Lookup: Token Attribution and Composition from a Single Decomposition** — Po-Kai Chen, Niki van Stein, Aske Plaat (Leiden). arXiv 2605.23393, 2026. https://arxiv.org/abs/2605.23393 · verified=yes
- What: treats attention and the MLP as one key-value template φ(S)U. A backward recursion yields token attributions and composition paths from a single forward pass. Evaluated on GPT-2 and Pythia.
- Relevance: a 2026 **language-model** paper that uses the unified key-value-lookup view for attribution. It is conceptually the nearest "interpretability via key-value template" work. It is not vision and does not test reading-set sufficiency.

**1.13 Towards Interpretable and Efficient Attention: Compressing All by Contracting a Few (CBSA)** — Qishuai Wen, Zhiyuan Huang, Chun-Guang Li. NeurIPS 2025 (spotlight). https://arxiv.org/abs/2509.16875 · verified=yes
- What: derives "Contract-and-Broadcast Self-Attention" by unrolling an MCR²-style objective. A few representative tokens are contracted and broadcast back. It is presented as inherently interpretable and linear-time.
- Relevance: the same group as MiTA. Its "few representatives" are the white-box analogue of our landmarks or summary slots. Cite it so reviewers do not think we missed it.

**1.14 Self-Attention Attribution: Interpreting Information Interactions Inside Transformer** — Yaru Hao, Li Dong, Furu Wei, Ke Xu. AAAI 2021. https://arxiv.org/abs/2004.11207 · verified=yes
- What: integrated gradients on attention edges, then prunes to an "attribution tree" of important token-to-token interactions (BERT).
- Relevance: an edge-level (query→key) importance precedent in NLP. Our reading sets are a training-free, structure-constrained alternative.

**1.15 Holistically Explainable Vision Transformers (B-cos ViTs)** — Moritz Böhle, Mario Fritz, Bernt Schiele. arXiv 2301.08669 (2023). https://doi.org/10.48550/arxiv.2301.08669 · verified=yes
- What: makes every ViT component dynamic-linear, so the whole model is a single input-dependent linear map and explanations are exact.
- Relevance: the "inherently interpretable" contrast to our post-hoc probe. It requires retraining, whereas we do not.

**1.16 Interpret Vision Transformers as ConvNets with Dynamic Convolutions** — Chong Zhou, Chen Change Loy, Bo Dai. arXiv 2309.10713 (2023). https://doi.org/10.48550/arxiv.2309.10713 · verified=yes
- What: rewrites self-attention as a ConvNet block with static and dynamic convolutions.
- Relevance: another "attention = dynamic standard layer" reinterpretation for vision, but at the architecture level, not for explanation.

---

## 2. Sparse, top-k, clustered and grouped attention, and token reduction (one line each)

| Work | Citation / URL | verified | How it relates to our probe |
|---|---|---|---|
| **Clustered Attention** — Vyas, Katharopoulos, Fleuret | NeurIPS 2020†, https://arxiv.org/abs/2007.04825 | yes | Clusters *queries* and computes attention once per centroid. This is our "route queries to m regions with shared reading" idea, but with full keys per centroid (we additionally top-k the keys). |
| **Routing Transformer** — Roy, Saffar, Vaswani, Grangier | TACL 2021†, https://arxiv.org/abs/2003.05997 | yes | Online k-means puts queries and keys in the same cluster, and attention is restricted within a cluster. A content-based shared sparse pattern; "routing ≈ clustering" (MiTA's Fig. 8 checks that MiTA does not degenerate into this). |
| **Explicit Sparse Transformer** — Zhao et al. | arXiv 2019, https://arxiv.org/abs/1912.11637 | yes | Per-query top-k attention, trained. This is the "query-specific reading set" end of our sharing axis. |
| **Top-k attention** — Gupta et al. | SustaiNLP 2021, https://arxiv.org/abs/2106.06899 | yes | Training-free per-query top-k on pretrained models (see 1.6). The baseline for our "no sharing" condition. |
| **Nyströmformer** — Xiong et al. | AAAI 2021†, https://arxiv.org/abs/2102.03902 | yes | Builds **landmarks by segment-means of queries and keys** and approximates softmax via the Nyström method. It is the origin of the average-pooled landmark idea that MiTA reuses, and it is close to our summary slots. |
| **KVT: k-NN attention** — Wang et al. | ECCV 2022, https://arxiv.org/abs/2106.00515 | yes | Per-query top-k keys in ViTs (trained). Claims it filters noisy background tokens. This is a direct hypothesis for RQ3 (do top-k sets drop background?). |
| **BiFormer (Bi-level Routing Attention)** — Zhu et al. | CVPR 2023†, https://arxiv.org/abs/2303.08810 | yes | Region-to-region top-k routing, then token attention within routed regions. It is a region-level reading set; MiTA calls BRA a "routing-only" degenerate case. |
| **Agent Attention** — Han et al. | ECCV 2024†, https://arxiv.org/abs/2312.08874 | yes | Pooled "agent" tokens aggregate from keys and broadcast to queries. This is the "compression-only" counterpart: our summary slots alone. |
| **Sparsifiner** — Wei et al. | CVPR 2023†, https://arxiv.org/abs/2303.13755 | yes (pdf) | Learns instance-dependent sparse connectivity (top-k on a low-rank attention estimate). A learned analogue of per-query reading sets. |
| **MoBA** — Lu et al. (Moonshot AI) | arXiv 2025, https://arxiv.org/abs/2502.13189 | yes | Block-partitioned key-values as experts with gated routing. MiTA's "rigid expert" predecessor. |
| **NSA (Native Sparse Attention)** — Yuan et al. (DeepSeek) | ACL 2025†, https://arxiv.org/abs/2502.11089 | yes | Combines compression, top-block selection and a sliding window. MiTA calls it the pioneer of routing plus compression. |
| **EViT** — Liang et al. | ICLR 2022†, https://arxiv.org/abs/2202.07800 | yes | Keeps the top-k tokens by CLS attention and **fuses inattentive tokens into one token** (attention-weighted average). This is almost exactly our "summary slot" for the discarded part, so cite it when describing summaries. |
| **DynamicViT** — Rao et al. | NeurIPS 2021†, https://arxiv.org/abs/2106.02034 | yes | Learned progressive token pruning. Its abstract states "the final prediction in vision transformers is only based on a subset of most informative tokens", a prior claim our sufficiency test quantifies without training. |
| **A-ViT** (arXiv title "AdaViT: Adaptive Tokens…") — Yin et al. | CVPR 2022†, https://arxiv.org/abs/2112.07658 | yes | Adaptive per-token halting. Its halting maps are often shown as object-focused, a counterpoint for RQ3. |
| **ToMe (Token Merging)** — Bolya et al. | ICLR 2023†, https://arxiv.org/abs/2210.09461 | yes | Training-free bipartite merging of similar tokens. MiTA reports that ToMe-based landmarks improve MiTA. This is a candidate for content-aware landmarks (Idea 4). |
| **TokenLearner** — Ryoo et al. | NeurIPS 2021†, https://arxiv.org/abs/2106.11297 | yes | A handful of learned spatial-attention-pooled tokens. A learned analogue of summary slots. |
| **IA-RED²** — Pan et al. | NeurIPS 2021, https://arxiv.org/abs/2106.12620 | yes | An "interpretability-aware" policy network drops uninformative patches. It claims the kept-patch maps beat raw attention and Grad-CAM on ImageNet-Segmentation. This is a precedent for "efficiency mechanism doubles as explanation". |
| **Which Tokens to Use?** — Haurum, Escalera, Taylor, Moeslund | ICCV Workshops 2023, https://arxiv.org/abs/2308.04657 | yes | Compares the reduction patterns of 10 token-reduction methods. Pruning patterns "significantly differ from **fixed radial patterns**", and top-K pruning is a strong baseline. **A direct precedent for our centre-prior control.** |
| **Artifacts and Attention Sinks: Structured Approximations for Efficient ViTs (Fast Nyström Attention)** — Lu, Liao, Wang, Yang, Shi | arXiv 2507.16018 (2025) | yes | Massive (sink) tokens and latent "artifact" tokens structure attention. They are exploited for a *training-free* landmark (Nyström) approximation on pretrained ViTs, and masking sinks late helps. Very close in spirit to our training-free probe; it explains why deep sets share a few tokens. |
| **SparsePR: Partition the Support, Reconstruct the Residual** — Taghavi, Langari, Pandey | arXiv 2608.18484 (2026) | yes | Training-free block-sparse attention for video DiTs. Shows "per-query concentration does not determine shared-route support" (median support 6.2% per query → 22.9% when 8 queries share a route) and that retained mass does not predict post-softmax error. **An independent analogue of our "sparsity vs sharing" decomposition** in another domain. |
| **Accelerating ViTs Based on Heterogeneous Attention Patterns** — Yu et al. (Baidu) | arXiv 2310.07664 (2023) | yes | Early-layer attention is similar across images and later layers are low-rank. Layer-structure evidence for RQ2. |

---

## 3. Explanation methods for ViTs (RQ4 candidates)

The cost column counts passes per image per explained class. "Needs grad/label" means it backpropagates from a class logit; the class is usually the predicted top-1, which is what our agreement-based tests use.

**3.1 Raw attention (CLS → patches, last layer)**
- Folklore baseline, used for example in DINO (Caron et al., ICCV 2021†, https://arxiv.org/abs/2104.14294, verified=yes).
- 1 forward pass; no gradient; class-agnostic.
- Implementation: in timm, attention weights are hidden when fused SDPA is on, so disable fused attention to read them (practical note, not a citation).
- Relevance: the "attention-as-explanation" foil. **For DINO+linear head, use the CLS attention of blocks 9–12**, since the head reads those four CLS tokens.

**3.2 Quantifying Attention Flow in Transformers (Attention Rollout and Attention Flow)** — Samira Abnar, Willem Zuidema. ACL 2020†. https://arxiv.org/abs/2005.00928 · verified=yes
- Mechanism: Rollout multiplies head-averaged (A+I) matrices across layers. Flow solves max-flow on the layer graph.
- Cost: Rollout is 1 forward pass, no gradient, class-agnostic. Flow needs a max-flow solve per image (much slower; rarely used for ViTs).
- Code: a Rollout baseline is in the Chefer repo (3.3). No official ViT Flow code was verified.
- Relevance: the cleanest gradient-free, class-agnostic comparator, like our probe. Ethayarajh & Jurafsky (5.6) prove that attention flows are Shapley values at the layer level.

**3.3 Transformer Interpretability Beyond Attention Visualization** — Hila Chefer, Shir Gur, Lior Wolf. CVPR 2021. https://arxiv.org/abs/2012.09838 · code https://github.com/hila-chefer/Transformer-Explainability · verified=yes
- Mechanism: LRP-based relevance per attention map, weighted by attention gradients, aggregated with rollout. Class-specific.
- Cost: 1 forward pass plus 1 LRP/backward pass; needs grad and class.
- Code: uses its own ViT re-implementation (`ViT_LRP.py`) based on timm, with a **DeiT notebook** (README: "Jan 5 2021: A Jupyter notebook for DeiT added"). Includes ImageNet-Segmentation and perturbation-test scripts.
- DINO: the linear head that concatenates four CLS tokens needs custom LRP rules (moderate effort).

**3.4 Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers (GAE)** — Hila Chefer, Shir Gur, Lior Wolf. ICCV 2021 (oral). https://arxiv.org/abs/2103.15679 · code https://github.com/hila-chefer/Transformer-MM-Explainability (has a ViT notebook) · verified=yes
- Mechanism: rollout of head-averaged (∇A ⊙ A)⁺ without LRP.
- Cost: 1 forward plus 1 backward pass; needs grad and class.
- Implementation: easy on timm with hooks on attention probabilities and their gradients.
- Relevance: **the best quality/effort ratio**. A standard strong baseline in TokenTM, SaCo and LibraGrad comparisons.

**3.5 Grad-SAM: Explaining Transformers via Gradient Self-Attention Maps** — Oren Barkan et al. CIKM 2021. https://arxiv.org/abs/2204.11073 (DOI 10.1145/3459637.3482126) · verified=yes
- Mechanism: averages ReLU(∂y/∂A) ⊙ A over heads and layers. Developed for BERT.
- Cost: 1 forward plus 1 backward pass; needs grad and class.
- Code: none verified. It is a few lines to implement.

**3.6 AttCAT: Explaining Transformers via Attentive Class Activation Tokens** — Yao Qiang, Deng Pan, Chengyin Li, Xin Li, Rhongho Jang, Dongxiao Zhu. NeurIPS 2022. Proceedings: https://proceedings.neurips.cc/paper_files/paper/2022/hash/20e45668fefa793bd9f2edf19be12c4b-Abstract-Conference.html · code https://github.com/qiangyao1988/AttCAT · verified=yes
- Mechanism: token features × their gradients (CAM-style), weighted by attention and summed over layers.
- Cost: 1 forward plus 1 backward pass; needs grad and class.
- Code: **NLP-only** (IMDB, SST-2, MNLI and similar). A ViT adaptation was used as a baseline in TokenTM.

**3.7 Beyond Intuition: Rethinking Token Attributions inside Transformers** — Jiamin Chen, Xuhong Li, Lei Yu, Dejing Dou, Haoyi Xiong. **TMLR 2023, not CVPR** (OpenReview rm0zIzlhcX). https://openreview.net/forum?id=rm0zIzlhcX · code https://github.com/jiaminchen-1031/transformerinterp (also in InterpretDL) · verified=yes
- Mechanism: chain-rule decomposition into "attention perception" (head-wise and token-wise approximations) and "reasoning feedback" (integrated gradients on the last attention map).
- Cost: 1 forward pass plus a few IG backward steps (step count not checked); needs grad and class.
- Code: has ViT segmentation and perturbation scripts adapted from Chefer's codebase.

**3.8 ViT-CX: Causal Explanation of Vision Transformers** — Weiyan Xie, Xiao-Hui Li, Caleb Chen Cao, Nevin L. Zhang. IJCAI 2023. https://arxiv.org/abs/2211.03064 · code https://github.com/vaynexie/CausalX-ViT · verified=yes
- Mechanism: clusters last-layer patch-embedding feature maps into masks, scores each masked image's target-class output, and corrects for pixel coverage. Perturbation-based and causal.
- Cost: 1 forward pass plus one forward per mask (batched; `gpu_batch=50` by default; the mask count depends on the clustering threshold, roughly 10²–10³, **not verified**). No gradient; needs the class score.
- Code: explicitly documents a reshape function for DeiT (drops CLS and DIST) and takes `target_layer=model.blocks[-1].norm1`. It is timm-style.

**3.9 Token Transformation Matters: Towards Faithful Post-hoc Explanation for Vision Transformer (TokenTM)** — Junyi Wu, Bin Duan, Weitai Kang, Hao Tang, Yan Yan. CVPR 2024. https://arxiv.org/abs/2403.14552 · verified=yes
- Mechanism: rollout-like aggregation over MHSA **and FFN**, where attention is reweighted by token length ratios ‖Ẽ‖/‖E‖ and a normalised-exponential cosine (NECC) between pre- and post-transformation tokens, together with (∇A p(c))⁺.
- Cost: 1 forward plus 1 backward pass; needs grad and class.
- Code: **no official repo found** (search returned only unrelated "TokenTM" accounts). It is reimplementable from Eqs. 9–17.
- Relevance: it uses **value and token norms**, which is the same intuition as Idea 1.

**3.10 AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers** — Reduan Achtibat et al. (Fraunhofer HHI). ICML 2024. https://arxiv.org/abs/2402.05602 · code https://github.com/rachtibat/LRP-eXplains-Transformers (`pip install lxt`, docs https://lxt.readthedocs.io) · verified=yes
- Mechanism: new LRP rules for softmax, matrix multiplication and LayerNorm, implemented as modified gradients.
- Cost: **a single forward plus a single backward pass**; needs class.
- Code: docs show a **torchvision ViT-B/16** example via `monkey_patch(vision_transformer)` plus zennit Gamma rules (the gamma needs tuning). timm DeiT/DINO support was **not verified** and will need a monkey patch of the timm attention module.

**3.11 Grad-CAM** — Ramprasaath R. Selvaraju et al. ICCV 2017†. https://arxiv.org/abs/1610.02391 · verified=yes
- Code: https://github.com/jacobgil/pytorch-grad-cam (ViT tutorial verified). Pass `reshape_transform` to drop CLS and reshape to 14×14, and use `target_layers=[model.blocks[-1].norm1]`. The last block's output patch tokens get zero gradient for CLS-readout models.
- Cost: 1 forward plus 1 backward pass; needs grad and class.
- Relevance: popular but typically weak on ViTs. Optional.

**3.12 Axiomatic Attribution for Deep Networks (Integrated Gradients)** — Mukund Sundararajan, Ankur Taly, Qiqi Yan. ICML 2017†. https://arxiv.org/abs/1703.01365 · verified=yes
- Cost: m backward passes (m≈20–50 interpolation steps), aggregated per patch; needs grad and class.
- Code: Captum. Noisy on ViTs (cf. LibraGrad). Optional.

**3.13 Explaining Through Transformer Input Sampling (TIS)** — Alexandre Englebert et al. ICCV Workshops (NIVT) 2023. https://openaccess.thecvf.com/content/ICCV2023W/NIVT/html/Englebert_Explaining_Through_Transformer_Input_Sampling_ICCVW_2023_paper.html · verified=yes (code stated public; the exact URL was truncated on the page)
- Mechanism: RISE-like, but **drops token subsets** instead of masking pixels (masks guided by embeddings). Saliency = score-weighted mask average normalised by mask frequency. Evaluated on ViT and DeiT.
- Cost: N forward passes on reduced token sets; no gradient.
- Relevance: uses the same perturbation as our necessity tests. **Caution:** token-dropping evaluation metrics will favour it.

**3.14 Vision DiffMask: Faithful Interpretation of Vision Transformers with Differentiable Patch Masking** — Angelos Nalmpantis et al. CVPR Workshops 2023. https://arxiv.org/abs/2304.06391 · verified=yes
- Mechanism: per-layer gates trained to find the **minimal patch subset that preserves the output**.
- Cost: needs training an interpretation network.
- Relevance: a conceptual neighbour of our *sufficiency* test, but it is learned. Not a recommended baseline (training cost).

**3.15 Learning to Estimate Shapley Values with Vision Transformers (ViT-Shapley)** — Ian Covert, Chanwoo Kim, Su-In Lee. ICLR 2023†. https://arxiv.org/abs/2206.05282 · verified=yes
- What: uses **attention masking** to evaluate ViTs with partial inputs, then trains an amortised Shapley explainer.
- Relevance: its masking mechanism is relevant. The method needs training, so exclude it as a baseline.

**3.16 RISE** — Vitali Petsiuk, Abir Das, Kate Saenko. BMVC 2018†. https://arxiv.org/abs/1806.07421 · verified=yes
- Cost: thousands of masked forward passes. Expensive; TIS is the ViT-native version.

**3.17 LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity** — Walid Bousselham, Angie Boggust, Sofian Chaybouti, Hendrik Strobelt, Hilde Kuehne. ICCV 2025. https://arxiv.org/abs/2404.03214 · code https://github.com/WalBouss/LeGrad (`pip install legrad_torch`; an **OpenCLIP wrapper**; timm/DeiT not verified) · verified=yes
- Mechanism: gradients with respect to the attention maps of every layer, aggregated.
- Cost: about 1 forward plus 1 backward pass; needs a target (class or text).

**3.18 LibraGrad: Balancing Gradient Flow for Universally Better Vision Transformer Attributions** — Faridoun Mehri, Mahdieh Soleymani Baghshah, Mohammad Taher Pilehvar. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Mehri_LibraGrad_Balancing_Gradient_Flow_for_Universally_Better_Vision_Transformer_Attributions_CVPR_2025_paper.html · code page https://nightmachinery.github.io/LibraGrad/ · verified=yes
- Mechanism: prunes and scales backward paths so that gradient methods regain FullGrad-completeness. The forward pass is unchanged.
- Cost: about 1 forward plus 1 backward pass; needs grad and class.
- Relevance: claims to beat transformer-specific methods (GAE, TokenTM, AttCAT) across 8 architectures. A strong modern option if its code supports timm (not verified).

**3.19 Other 2024–2026 ViT attribution methods** (verified to exist; shallow read only):
- **CoIBA**, Comprehensive Information Bottleneck (Hong et al., CVPR 2025; CVF page verified).
- **MDA**, Metric-Driven Attributions for ViTs (ICLR 2025; proceedings PDF seen, authors not captured → verified=partial).
- **GMAR**, Gradient-Driven Multi-Head Attention Rollout (Sehyeong Jo et al., arXiv 2504.19414, 2025).
- **AG-CAM**, Attention Guided CAM (Saebom Leem, Hyunseok Seo, AAAI 2024; DOI 10.1609/aaai.v38i4.28077).
- **RFEM**, "There is More to Attention: Statistical Filtering Enhances Explanations in ViTs" (Ayyar, Benois-Pineau, Zemmari; Springer 2025, arXiv 2510.06070). Uses human-gaze plausibility.
- **DAP**, Decision-Aware Attention Propagation (arXiv 2604.18094, 2026).
- **HiLRP** (arXiv 2609.01282, 2026; only a snippet seen → verified=partial).

None is needed as a baseline. CoIBA and MDA are too costly (optimisation per image).

**3.20 Explainability of Vision Transformers: A Comprehensive Review and New Perspectives** — Rojina Kashefi, Leili Barekatain, Mohammad Sabokrou, Fatemeh Aghaeipoor. arXiv 2311.06786 (2023). https://arxiv.org/abs/2311.06786 · verified=yes
- Relevance: a survey to cite for the taxonomy of ViT explanation methods.

---

## 4. Mechanistic and structural analyses of ViT attention (RQ2, RQ3)

**4.1 Do Vision Transformers See Like Convolutional Neural Networks?** — Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, Alexey Dosovitskiy. NeurIPS 2021. https://arxiv.org/abs/2108.08810 · verified=yes
- What: ViT lower layers mix local and global heads (attention distance grows with data scale), strong skip connections propagate features, and spatial information is preserved through the network.
- Relevance: RQ2's early-local / late-global structure. Also supports the "token = patch" premise (spatial localisation is preserved).

**4.2 An Image is Worth 16x16 Words (ViT)** — Alexey Dosovitskiy et al. ICLR 2021†. https://arxiv.org/abs/2010.11929 · verified=yes
- Relevance: introduces the "mean attention distance per head per layer" analysis, which is the canonical layer-structure plot.

**4.3 What do Vision Transformers Learn? A Visual Exploration** — Amin Ghiasi, Hamid Kazemi, Eitan Borgnia, Steven Reich, Manli Shu, Micah Goldblum, Andrew Gordon Wilson, Tom Goldstein. arXiv 2212.06727 (2022). https://arxiv.org/abs/2212.06727 · verified=yes
- What: feature visualisation of ViT neurons (on FFN activations), and patch-wise activation maps. ViTs use **background** information more effectively than CNNs. A foreground/background masking table (bounding boxes) shows ViT-B16 keeps 95.64% (normalised top-5) with foreground only and 31.59% with background only.
- Relevance: RQ3 prior evidence that ViTs exploit background.

**4.4 Analyzing Vision Transformers for Image Classification in Class Embedding Space** — Martina G. Vilas, Timothy Schaumlöffel, Gemma Roig. NeurIPS 2023†. https://arxiv.org/abs/2310.18969 · verified=yes
- What: projects intermediate tokens onto the class-embedding space (a logit-lens for ViTs) to track how class information builds up across layers and tokens, and which attention and MLP components contribute.
- Relevance: a cheap per-layer readout we could apply to the reading-set tokens (do the selected deep tokens already carry the class?).

**4.5 Interpreting CLIP's Image Representation via Text-Based Decomposition** — Yossi Gandelsman, Alexei A. Efros, Jacob Steinhardt. ICLR 2024†. https://arxiv.org/abs/2310.05916 · verified=yes
- What: decomposes CLIP-ViT's final representation as a sum over layers × heads × **tokens**. The last few attention layers carry most of the direct effect. Heads are labelled with text.
- Relevance: the per-(head, token) direct-effect decomposition is the right tool to weight our neurons by their *contribution* rather than attention (Idea 1).

**4.6 Decomposing and Interpreting Image Representations via Text in ViTs Beyond CLIP** — Sriram Balasubramanian, Samyadeep Basu, Soheil Feizi. NeurIPS 2024. https://arxiv.org/abs/2406.01583 · code https://github.com/SriramB-98/vit-decompose · verified=yes
- What: automates Gandelsman-style decompositions for **DeiT, DINO**, DINOv2, Swin and MaxViT, and maps component contributions to CLIP space. Gives token-importance heatmaps.
- Relevance: the **same model families as ours**, with ready code for per-head, per-token contributions.

**4.7 Vision Transformers Need Registers** — Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. ICLR 2024†. https://arxiv.org/abs/2309.16588 · verified=yes
- What: high-norm outlier tokens (~2% of tokens, roughly 10× the norm) appear in low-information background patches. They hold global rather than local information. Registers remove them.
- Key verified details:
  - "all models but DINO exhibit peaky outlier values". DINO v1 is the **exception**.
  - Outliers appear "only [in] the three largest models" for DINOv2.
  - DeiT-III B and L show them.
- Relevance (important for our high-norm category): **DINO ViT-S/16 and DeiT-S may have few or no Darcet-style artifacts.** Define "high-norm" relatively (e.g., the per-image top-q% of norms, or tokens attracting the most incoming attention), report how many such tokens exist, and do not rely on the Darcet cutoff.

**4.8 Vision Transformers Don't Need Trained Registers** — Nick Jiang, Amil Dravid, Alexei A. Efros, Yossi Gandelsman. NeurIPS 2025. https://arxiv.org/abs/2506.08010 · verified=yes
- What: a sparse set of "register neurons" (in the MLP) creates the high-norm outlier tokens. Shifting those activations into an extra untrained token gives training-free registers. High attention appears after a specific layer's MLP.
- Relevance: a training-free tool to remove sinks and test whether deep shared reading sets are just sinks (RQ2/RQ3).

**4.9 Efficient Streaming Language Models with Attention Sinks** — Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. ICLR 2024†. https://arxiv.org/abs/2309.17453 · verified=yes
- Relevance: the origin of "attention sinks". Use it for the language of shared, uninformative attractor tokens.

**4.10 Massive Activations in Large Language Models** — Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. COLM 2024†. https://arxiv.org/abs/2402.17762 · verified=yes
- What: very few huge activations act as biases and attention sinks. The paper also covers ViTs (from memory: they report massive activations in CLIP/DINOv2 ViTs; not re-checked).
- Relevance: background for sink tokens.

**4.11 Register and [CLS] tokens induce a decoupling of local and global features in large ViTs** — Alexander Lappe, Martin A. Giese. arXiv 2505.05892 (2025). https://arxiv.org/abs/2505.05892 · verified=yes
- What: even with registers, clean attention maps "do not accurately reflect the integration of local image information". The [CLS] token causes a similar decoupling.
- Relevance: a caution for RQ3/RQ4. Clean-looking maps are not proof of faithful local use, which motivates our sufficiency and necessity tests.

**4.12 Artifacts and Attention Sinks (Lu et al. 2025)** — see §2. Massive and artifact tokens mutually suppress each other; masking sinks late helps. Relevance: RQ2.

**4.13 A Close Look at Spatial Modeling: From Attention to Convolution (FCViT)** — Xu Ma, Huan Wang, Can Qin, Kunpeng Li, Xingchen Zhao, Jie Fu, Yun Fu. arXiv 2212.12552 (2022). https://arxiv.org/abs/2212.12552 · verified=yes
- What: on timm ViT and DeiT, deep-layer attention maps are "nearly identical ... regardless of the query patch" (query-irrelevant, even head-irrelevant) and "intrinsically sparse, few tokens dominate". Confirmed statistically on ImageNet.
- Relevance: **directly anticipates our RQ2 finding** that deep layers share few tokens. Must cite.

**4.14 DeepViT: Towards Deeper Vision Transformer** — Daquan Zhou et al. arXiv 2103.11886 (2021). https://arxiv.org/abs/2103.11886 · verified=yes
- What: "attention collapse": attention maps become similar across deeper layers.
- Relevance: RQ2 support.

**4.15 GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond** — Yue Cao, Jiarui Xu, Stephen Lin, Fangyun Wei, Han Hu. ICCVW 2019†. https://arxiv.org/abs/1904.11492 · verified=yes
- What: non-local (self-attention) blocks learn **query-independent** attention maps, so one global map is shared.
- Relevance: the oldest precedent for the "sharing is cheap in deep and global layers" finding.

**4.16 Dissecting Query-Key Interaction in Vision Transformers** — Xu Pan, Aaron Philip, Ziqian Xie, Odelia Schwartz. arXiv 2405.14880 (2024; venue not checked). https://arxiv.org/abs/2405.14880 · verified=yes
- What: SVD of W_qᵀW_k. Early layers attend to *similar* tokens (grouping) and late layers attend more to *dissimilar* or contextual tokens (background, other objects), especially in classification-trained ViTs. Singular-vector pairs are often semantic (object↔part, foreground↔background).
- Relevance: explains why deep DeiT-S reading sets need not be object-focused (RQ3). Also a head-level tool.

**4.17 How Do Vision Transformers Work?** — Namuk Park, Songkuk Kim. ICLR 2022†. https://arxiv.org/abs/2202.06709 · verified=yes
- Relevance: background for multi-head self-attention as low-pass, aggregating filters and for late-stage attention behaviour.

**4.18 Intriguing Properties of Vision Transformers** — Muzammal Naseer et al. NeurIPS 2021. https://arxiv.org/abs/2105.10497 · verified=yes
- What: PatchDrop robustness. "DeiT-S ... obtains 70% accuracy when 50% of the image content is removed". Also robust to removing salient (foreground) and non-salient content.
- Relevance: **a sufficiency baseline expectation.** A random 50% of tokens already gives high accuracy, so our sufficiency numbers must be compared against random and centre sets of equal size.

**4.19 A Comprehensive Study of Image Classification Model Sensitivity to Foregrounds, Backgrounds, and Visual Attributes (RIVAL10)** — Mazda Moayeri, Phillip Pope, Yogesh Balaji, Soheil Feizi. CVPR 2022†. https://arxiv.org/abs/2201.10766 · verified=yes
- What: noise-based relative foreground sensitivity across models. Verified quote: "DeiTs with much lower RFS than ViTs, suggesting that the heavy augmentations DeiTs leveraged ... may have also made the models much more sensitive to backgrounds".
- Relevance: **an independent explanation for our DeiT-S result** that deep sets are not more object-focused than a centre prior. Cite it in RQ3.

**4.20 Visualizing and Understanding Patch Interactions in Vision Transformer** — Jie Ma, Yalong Bai, Bineng Zhong, Wei Zhang, Ting Yao, Tao Mei. IEEE TNNLS 2023/2024. https://arxiv.org/abs/2203.05922 · verified=yes
- What: quantifies the impact of patch-to-patch attention and finds each patch's "responsive field" (the set of high-impact patches it reads), which is semantically related. Uses it for adaptive windows and to drop "indiscriminative patches" that every patch reads.
- Relevance: **the closest vision precedent to per-query "reading sets"**, though aimed at architecture design rather than faithfulness tests. Their "indiscriminative patches read by everyone" resemble our deep shared tokens and sinks.

**4.21 Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video** — Sonia Joseph et al. CVPR 2025 MIV workshop (oral and tutorial). https://arxiv.org/abs/2504.19475 · code https://github.com/Prisma-Multimodal/ViT-Prisma · verified=yes
- What: HookedViT for 75+ timm, OpenCLIP and HF models (ViT, CLIP, **DINO**), activation patching and ablation, ACDC integration, and 80+ pretrained SAEs (CLIP-B, DINO-B).
- Relevance: a ready-made hooking and patching infrastructure for Idea 3, if we want it. Our own hooks are probably simpler for ViT-S.

**4.22 Sparse autoencoders for ViTs**
- **Interpretable and Testable Vision Features via Sparse Autoencoders** (v1 title: *Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models*) — Samuel Stevens, Wei-Lun Chao, Tanya Berger-Wolf, Yu Su. arXiv 2502.06755 (2025). https://arxiv.org/abs/2502.06755 · verified=yes. SAEs on CLIP and DINOv2 ViT activations with patch-level causal edits.
- **Interpreting vision transformers via residual replacement model** — Jinyeong Kim, Junhyeok Kim, Yumin Shim, Joohyeok Kim, Sunyoung Jung, Seong Jae Hwang. NeurIPS 2025. https://neurips.cc/virtual/2025/poster/118006 · code https://github.com/rubato-yeong/RRM · verified=yes. SAE features across all layers (6.6K annotated) plus a "residual replacement model" for faithful sparse circuits. Their stated limitation is that it "only indirectly captures the roles of attention ... modules", which is exactly where our probe adds something.
- Relevance: the SAE/feature axis is complementary to our token axis. Mention it as an alternative granularity (features vs. tokens), not as a baseline.

**4.23 Explaining the "few tokens suffice" line in token-reduction papers** — see DynamicViT and Haurum et al. in §2.

---

## 5. "Attention is (not) explanation" and evaluation methodology

**5.1 Attention is not Explanation** — Sarthak Jain, Byron C. Wallace. NAACL 2019†. https://arxiv.org/abs/1902.10186 · verified=yes
- What: attention weights correlate weakly with gradient or leave-one-out importance, and different (adversarial) attention distributions give the same predictions.
- Relevance: the core objection our sufficiency and necessity tests answer empirically.

**5.2 Attention is not not Explanation** — Sarah Wiegreffe, Yuval Pinter. EMNLP 2019†. https://arxiv.org/abs/1908.04626 · verified=yes
- What: rebuttal. Proposes tests, including whether the model needs the learned attention at all (uniform attention baseline) and adversarial-training checks. Whether attention explains is task- and model-dependent.
- Relevance: frames our claim as "a faithful *readout* of where computation reads", not "attention = importance".

**5.3 Is Attention Interpretable?** — Sofia Serrano, Noah A. Smith. ACL 2019†. https://arxiv.org/abs/1906.03731 · verified=yes
- What: zeroing the highest-attention items often fails to flip decisions. Gradient-based rankings flip decisions faster.
- Relevance: the necessity test design (delete top-attended vs. random vs. gradient-ranked) maps onto our RQ1 necessity design.

**5.4 The elephant in the interpretability room: Why use attention as explanation when we have saliency methods?** — Jasmijn Bastings, Katja Filippova. BlackboxNLP 2020†. https://arxiv.org/abs/2010.05607 · verified=yes
- Relevance: argues saliency methods are the right comparators, which justifies RQ4.

**5.5 Is Sparse Attention more Interpretable?** — Clara Meister, Stefan Lazov, Isabelle Augenstein, Ryan Cotterell. ACL 2021 (short). https://aclanthology.org/2021.acl-short.17/ · verified=yes
- What: "only a weak relationship between inputs and co-indexed intermediate representations exists — under sparse attention and otherwise". Inducing sparsity "may make it less plausible" to interpret.
- Relevance: **the most direct critique of our premise.** Address it with a token-identity check (Idea 7) and by stressing that we test *behavioural* sufficiency and necessity, not plausibility alone.

**5.6 Attention Flows are Shapley Value Explanations** — Kawin Ethayarajh, Dan Jurafsky. ACL 2021†. https://arxiv.org/abs/2105.14652 · verified=yes
- What: attention weights and leave-one-out values are not Shapley values, but attention flows are (layer-wise).
- Relevance: theoretical support for using flow and rollout as principled attention baselines.

**5.7 Attention is Not Only a Weight: Analyzing Transformers with Vector Norms** — Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, Kentaro Inui. EMNLP 2020†. https://arxiv.org/abs/2004.10102 · verified=yes
- What: uses ‖α_ij f(x_j)‖ (attention × transformed value norm) instead of α_ij. High attention to special tokens comes with tiny value norms.
- Relevance: **Idea 1.** Rank neurons by write magnitude. Directly relevant to sink tokens in our sets.

**5.8 Measuring the Mixing of Contextual Information in the Transformer (ALTI)** — Javier Ferrando, Gerard I. Gállego, Marta R. Costa-jussà. EMNLP 2022†. https://arxiv.org/abs/2203.04212 · verified=yes
- What: token-to-token contributions that account for multi-head attention, the residual and LayerNorm, aggregated across layers.
- Relevance: a norm-based alternative to rollout. A cheap gradient-free baseline or neuron-scoring rule.

**5.9 Information Flow Routes: Automatically Interpreting Language Models at Scale** — Javier Ferrando, Elena Voita. EMNLP 2024†. https://arxiv.org/abs/2403.00824 · verified=yes
- What: builds per-prediction graphs keeping only important token-to-token edges via attribution, in a single forward pass.
- Relevance: the language-model analogue of extracting a sparse read graph. Our reading sets are a constrained, landmark-shared version. Their edge-importance threshold is an alternative to a fixed k.

**5.10 On Identifiability in Transformers** — Gino Brunner et al. ICLR 2020†. https://arxiv.org/abs/1908.04211 · verified=yes
- What: attention weights are not identifiable when sequence length exceeds head dimension. Studies whether tokens stay identifiable across layers.
- Relevance: supports a token-identity check. Note N=197 > d_head=64 for ViT-S, so attention weights are non-identifiable. This is another argument for behavioural tests over raw weights.

**5.11 Rethinking Attention-Model Explainability through Faithfulness Violation Test** — Yibing Liu, Haoliang Li, Yangyang Guo, Chenqi Kong, Jing Li, Shiqi Wang. ICML 2022†. https://arxiv.org/abs/2201.12114 · verified=yes
- What: attention explanations cannot express the polarity of impact (suppressive vs. supportive). Raw attention violates faithfulness most often, and Attention⊙Gradient does best.
- Relevance: our necessity test should report signed effects (deleting some in-set tokens may *raise* confidence).

**5.12 On the Faithfulness of Vision Transformer Explanations (SaCo)** — Junyi Wu, Weitai Kang, Hao Tang, Yuan Hong, Yan Yan. CVPR 2024. https://arxiv.org/abs/2404.01415 · verified=yes
- What: current metrics "struggle to differentiate between advanced explanation methods and Random Attribution". SaCo compares salience-ranked pixel groups. Raw attention underperforms, while gradients and multi-layer aggregation help.
- Relevance: an **RQ4 faithfulness metric** and a strong argument that our *random* controls are essential.

**5.13 Missingness Bias in Model Debugging** — Saachi Jain, Hadi Salman, Eric Wong, Pengchuan Zhang, Vibhav Vineet, Sai Vemprala, Aleksander Madry. ICLR 2022†. https://arxiv.org/abs/2204.08945 · verified=yes
- What: blacking out pixels biases predictions. **Dropping tokens in ViTs side-steps missingness bias.**
- Relevance: justifies token dropping (or attention masking) rather than pixel masking for both our ablations and baseline evaluations. Apply the same deletion mechanism to all methods.

**5.14 Sanity Checks for Saliency Maps** (Adebayo et al., NeurIPS 2018†, https://arxiv.org/abs/1810.03292, verified=yes), **ROAR: A Benchmark for Interpretability Methods in Deep Neural Networks** (Hooker et al., NeurIPS 2019†, https://arxiv.org/abs/1806.10758, verified=yes), and **ROAD: A Consistent and Efficient Evaluation Strategy for Attribution Methods** (Rong et al., ICML 2022†, https://arxiv.org/abs/2202.00449, verified=yes)
- Relevance: the standard evaluation caveats: model-randomisation checks, retrain-vs-no-retrain removal, and information leakage through mask shape.

**5.15 When Random Saliency Looks Trained: Architectural Center Bias in CNN Interpretability** — Keying Kuang, Iain Carmichael, Elizabeth Purdom. ICML 2026 (PMLR 306). https://proceedings.mlr.press/v306/kuang26a.html · verified=yes
- What: CNN saliency has an architecture-induced centre prior that is "largely absent" in ViTs. Proposes a **corner-shift benchmark** and a **Center-Shift Index**.
- Relevance: **RQ3.** In ViTs the centre bias we see is data- and learning-induced, not architectural. Their corner-shift protocol is the cleanest way to separate object from centre (Idea 5).

**5.16 New Definitions and Evaluations for Saliency Methods: Staying Intrinsic, Complete and Sound** (arXiv 2211.02912, seen via ar5iv; authors not captured → verified=partial)
- What: includes a "Centered Gaussian" baseline map whose insertion and saliency-metric scores are flagged as "likely due to centrality bias".
- Relevance: a precedent that a centre map is a competitive saliency baseline. Include a centre-Gaussian "explanation" in RQ4.

**5.17 Head analysis classics: Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned** (Voita et al., ACL 2019†, https://arxiv.org/abs/1905.09418, verified=yes) and **Are Sixteen Heads Really Better than One?** (Michel, Levy, Neubig, NeurIPS 2019†, https://arxiv.org/abs/1905.10650, verified=yes)
- Relevance: head-level importance and pruning, for Idea 2.

**5.18 Activation patching: How to use and interpret activation patching** (Heimersheim & Nanda, arXiv 2404.15255, 2024, verified=yes) and **Towards Best Practices of Activation Patching in Language Models: Metrics and Methods** (Zhang & Nanda, ICLR 2024†, https://arxiv.org/abs/2309.16042, verified=yes)
- Relevance: the methodology for Idea 3: noising vs. denoising, metric choice, and zero, mean or resample ablation.

---

## 6. Datasets and annotations useful for RQ3 (object, parts, background)

- **ImageNet-S** (from *Large-scale Unsupervised Semantic Segmentation*) — Shanghua Gao, Zhong-Yu Li, Ming-Hsuan Yang, Ming-Ming Cheng, Junwei Han, Philip Torr. TPAMI 2022†. https://arxiv.org/abs/2106.03149 · verified=yes. Pixel masks for ImageNet classes on val and test. Gives foreground masks for real ImageNet val images.
- **PartImageNet** — Ju He et al. ECCV 2022†. https://arxiv.org/abs/2112.00933 · verified=yes. Part-level masks (158 classes, from memory). Enables the "parts" half of RQ3.
- **ImageNet-Segmentation (Guillaumin et al., IJCV 2014)** — the `gtsegs_ijcv.mat` file used by Chefer et al., Beyond Intuition and TokenTM for segmentation tests. The file link appears in the verified repos; the paper page was not fetched → verified=partial. Using it makes RQ4 numbers comparable to the literature.
- **Salient ImageNet** — Sahil Singla, Soheil Feizi. ICLR 2022†. https://arxiv.org/abs/2110.04301 · verified=yes. Core vs. spurious feature masks; lets us split "object" into core and spurious-context tokens.
- **ImageNet-9 backgrounds (Noise or Signal: The Role of Image Backgrounds in Object Recognition)** — Kai Xiao, Logan Engstrom, Andrew Ilyas, Aleksander Madry. ICLR 2021†. https://arxiv.org/abs/2006.09994 · verified=yes. Foreground and background swaps for counterfactual patching (Idea 3).
- **RIVAL10** — Moayeri et al. (4.19). Segmentation and attribute masks for 10 classes.

---

## 7. Closest prior work and how we differ (candid)

**(a) MiTA (Wen et al. 2026). High overlap; must be addressed head-on.**

What it already shows (verified in the v6 text):
- the fast-weight-MLP framing with tokens as hidden units;
- visualisations of gathered key-values per expert and routed queries (Fig. 3);
- a qualitative "token-pruning effect" in late layers concentrated on "class-relevant regions" (Fig. 4; MiTA-*trained* models, ImageNet training images);
- per-layer key-value/query overlap (Fig. 8);
- that **standard-attention-trained DeiT-T/S evaluated with MiTA at inference "retains over 95% of the original performance"** (App. C, Fig. 9; these are their own Tab. 2 models, not official timm checkpoints);
- that k matters more than m (ablation; trained models).

What we add:
1. An explicit **interpretability question**: faithfulness through *agreement with the original top-1*, rather than accuracy against labels, on official DeiT-S and DINO ViT-S checkpoints.
2. A **decomposition of the faithfulness cost** into sparsity (per-query top-k), sharing (landmark-shared sets) and compression (summary slots). Note that MiTA's k-more-important-than-m ablation is consistent with our "sparsity not sharing" finding; present ours as the training-free, faithfulness-level version.
3. **Necessity tests** with random and **centre-matched** controls.
4. A quantitative RQ3 test against a centre prior. Our result for DeiT-S (deep sets ≈ centre prior; only DINO layer 11 beats it) *qualifies* MiTA's qualitative "class-relevant" impression. This is a genuine, reportable contribution.
5. Comparison with ViT explanation methods (RQ4).

Wording: present the probe as "MiTA's routing and compression applied post hoc as a measurement instrument", and credit MiTA for the training-free transfer observation.

**(b) Patch-interaction and query-irrelevance analyses. Moderate overlap with RQ2.**
- Ma et al. 2022 (FCViT) and DeepViT already report deep-layer query-irrelevant, sparse, collapsed attention in timm ViT/DeiT. GCNet reports the same for non-local blocks.
- Ma et al. TNNLS 2023 defines per-patch "responsive fields" and patches read by everyone.
- Our RQ2 finding (early query-specific, deep shared and few tokens) is therefore **confirmatory**. The new parts are measuring it as a *faithfulness cost of sharing* per layer, and checking whether the shared tokens are sinks or object tokens.

**(c) Training-free sparse attention on pretrained models. Low–moderate overlap.**
- Gupta et al. 2021: training-free top-k in pretrained language models.
- Lu et al. 2025: training-free Nyström with sink-aware landmarks on pretrained ViTs.
- SparsePR 2026: per-query vs. pooled support in video DiTs.
- These are all efficiency papers. None runs necessity or plausibility tests or compares with explanation methods.

**(d) Token reduction as explanation. Moderate overlap with RQ3 framing.**
- IA-RED² claims kept-patch maps are interpretable.
- Haurum et al. compare reduction patterns with **fixed radial patterns** (a centre prior!).
- DynamicViT and Naseer et al. establish that few tokens suffice.
- Cite all three. Our centre-matched necessity control is in the same spirit as Haurum's radial comparison, but applied to deletion effects inside reading sets.

**(e) Duality-for-interpretability lineage. Conceptual overlap, different domain.**
- Geva 2021 (FFN neurons), Irie et al. 2022 (spotlights of attention over training data), Schug et al. 2025 (attention as hypernetwork), and Chen et al. 2026 (every component is a lookup; language-model attribution) use key-value/MLP equivalences for interpretation.
- **I found no paper that treats ViT tokens as the hidden neurons of an attention-MLP and tests sufficiency and necessity of the resulting neuron sets.**
- Caveat: absence of evidence after about 15 targeted searches, not proof.

**(f) Threats from the "attention is not explanation" literature.**
- Meister et al. 2021: intermediate tokens are not inputs.
- Brunner et al. 2020: non-identifiability.
- Lappe & Giese 2025: clean maps do not imply local integration.
- SaCo 2024: metrics cannot tell methods from random.
- Our behavioural tests and controls are the answer to these, but **we need a token-identity check** to justify "each neuron IS an image patch" at deep layers. Darcet 2024 shows high-norm tokens lose position and pixel information.

**Novelty risk overall: medium.**
- The equivalence, the probe mechanics, the training-free transfer and the qualitative late-layer pruning are all prior work.
- The defensible novelty is the controlled evaluation protocol and its findings, especially the centre-prior negative result for DeiT-S versus DINO, and the head-to-head with explanation baselines under identical deletion tests.

---

## 8. Recommended RQ4 baselines

Guiding principles for a fair comparison:
- **Same budget.** Convert every map to a top-k token set at the same k, and convert our reading sets to a map by membership frequency or attention×value-norm mass.
- **Same perturbation.** Token dropping or attention masking, with no pixel blackout (Jain et al. 2022).
- **Same target.** The original top-1 prediction.
- **Same controls.** Random and centre-matched sets, plus a centre-Gaussian "explanation".
- **DINO.** The head reads the CLS of blocks 9–12, so CLS-attention methods must aggregate over those blocks.

| # | Method | Code (verified) | Cost per image | Needs grad / label | Notes |
|---|---|---|---|---|---|
| 1 | **Raw CLS attention** (last layer, and layer-matched to each probe layer) | none needed (timm hooks; disable fused attention) | 1 fwd | no / no | The "attention = explanation" foil. Layer-matched maps make RQ2 comparisons apples-to-apples. For DINO, average blocks 9–12. |
| 2 | **Attention Rollout** (Abnar & Zuidema 2020) | baseline inside https://github.com/hila-chefer/Transformer-Explainability | 1 fwd | no / no | Class-agnostic and gradient-free, like our probe. Cheapest multi-layer baseline. Skip Attention *Flow* (max-flow per image is too slow, and rollout is the standard proxy). |
| 3 | **Chefer et al. ICCV 2021 (GAE)**: rollout of (∇A⊙A)⁺ | https://github.com/hila-chefer/Transformer-MM-Explainability (ViT notebook) | 1 fwd + 1 bwd | yes / yes (top-1) | Best quality/effort and widely reported. Easy to implement on timm DeiT-S and DINO (backprop through the 4-CLS linear head works automatically). |
| 4 | **AttnLRP** (Achtibat et al., ICML 2024), *or fallback* **Chefer CVPR 2021** (LRP + grad + rollout) | https://github.com/rachtibat/LRP-eXplains-Transformers (`lxt`; torchvision ViT example, timm needs a patch) / https://github.com/hila-chefer/Transformer-Explainability (own timm-based ViT with a DeiT notebook) | 1 fwd + 1 bwd | yes / yes | The modern "faithful" representative. If LXT patching of timm takes more than a day, use Chefer CVPR21. Check its DINO-head handling (custom LRP for the concatenated CLS). |
| 5 | **TIS** (token-sampling RISE) *or* **ViT-CX** | ViT-CX: https://github.com/vaynexie/CausalX-ViT (documents DeiT reshape; timm `blocks[-1].norm1`). TIS: code stated public, URL not captured. | ~10²–10³ fwd (approximate, not verified) | no / needs class score | A gradient-free causal/perturbation representative. Note TIS optimises the same token-dropping signal used in deletion metrics, so it gets a home advantage. Report it but flag this. ViT-CX is safer and has verified code. |
| (opt) | **Grad-CAM** on `blocks[-1].norm1` | https://github.com/jacobgil/pytorch-grad-cam (ViT tutorial) | 1 fwd + 1 bwd | yes / yes | Popular but weak on ViTs. Include only if space allows. |
| (opt) | **Centre-Gaussian map** | trivial | 0 | no / no | Essential sanity baseline for RQ3 and RQ4 (cf. 5.16). |

Not recommended as main baselines:
- **TokenTM**: no official code found; reimplementation risk. Cite it as related.
- **AttCAT**: official code is NLP-only.
- **Beyond Intuition**: it is TMLR 2023, not CVPR. Code exists but is less standard.
- **Grad-SAM**: no code. Trivial, so optional.
- **LibraGrad and LeGrad**: strong and recent, but timm support is unverified (LeGrad's code is an OpenCLIP wrapper).
- **IG**: noisy on ViTs and m× the cost.
- **ViT-Shapley and Vision DiffMask**: need training.
- **RISE**: thousands of forward passes.
- **CoIBA and MDA**: per-image optimisation.

Suggested metrics for RQ4:
1. Sufficiency: agreement when keeping only the top-k tokens (by each method), plus insertion AUC with token dropping.
2. Necessity: agreement drop and logit drop when deleting the top-k, against random and centre-matched deletions.
3. SaCo (Wu et al. CVPR 2024).
4. Plausibility: pointing game and IoU on ImageNet-S or ImageNet-Segmentation, **reported as gain over the centre-Gaussian map**.

---

## 9. Ideas to borrow (concrete, with the RQ each serves)

1. **Weight neurons by what they write, not only by attention (value and OV norms).** From Kobayashi 2020, ALTI 2022, TokenTM 2024, Gandelsman 2024 and Balasubramanian 2024.
   - Score key j for a landmark by α_ij·‖W_O v_j‖, or by its direct contribution to the head output projected onto the predicted class direction (Vilas 2023 class-embedding projection), rather than by α_ij alone. Select top-k on that score.
   - *Why:* sink tokens get high α but tiny value norm (Kobayashi). Norm-aware sets should be **more faithful at the same k** (RQ1) and change the composition of deep sets (RQ3). This also makes the RQ4 comparison fairer against norm-aware baselines.
2. **Head-level reading sets and head typing.**
   - Compute landmarks and top-k per head (MiTA's algorithm is per head) instead of head-averaged sets.
   - Report per-head mean attention distance (ViT; Raghu 2021), per-head sharing cost and per-head object/centre ratio.
   - Use Pan et al.'s QK-SVD to label heads as grouping or contextual.
   - *Why:* RQ2's early-vs-deep story is likely a mixture of local and global heads (Raghu). Averaging can hide object-focused heads in DeiT-S. Head-averaging may also be what makes deep sets look like a centre prior.
3. **Causal (activation) patching instead of only deletion.** From Heimersheim & Nanda 2024, Zhang & Nanda 2024, ImageNet-9, and Prisma for tooling.
   - In the necessity test, replace the keys and values of in-set tokens with (a) the mean over images (mean-ablation), (b) the same position from a background-swapped or object-removed counterfactual (ImageNet-9 or inpainting), or (c) a random other image (resample-ablation). Report the effect in logit difference.
   - *Why:* deletion shrinks the softmax denominator and is off-distribution. Patching isolates "content of the object neurons" from "number of neurons", which strengthens RQ1 necessity and the object-vs-centre claims in RQ3.
4. **Content-aware landmarks instead of a fixed grid.**
   - Build the m regions by k-means or ToMe bipartite merging of queries (MiTA reports ToMe landmarks help), by clustering DINO keys, or from ImageNet-S/PartImageNet masks (an "oracle object/part landmark").
   - *Why:* a uniform 14×14 grid partition bakes in spatial structure (the central region is always a landmark). Semantic landmarks test whether the sharing cost (RQ2) and object focus (RQ3) are artefacts of the grid. A part-mask landmark condition directly addresses "object/parts" in RQ3.
5. **De-confound object from centre.**
   - Use the corner-shift idea (Kuang et al., ICML 2026): evaluate on shifted or off-centre crops, or on the subset of ImageNet-S val images whose object centroid is far from the centre.
   - Report the object score as **gain over a centre prior** (an information-gain-over-centre-baseline style, as in fixation-saliency evaluation), and compare with Haurum's radial-pattern analysis.
   - *Why:* this is exactly the failure mode we observe for DeiT-S. Moayeri et al. 2022 give a plausible cause (DeiT's heavy augmentation increases background sensitivity).
6. **Sink- and register-aware analysis.**
   - Detect sinks by incoming attention or CLS→token attention exceeding CLS→CLS (Lu et al. 2025's criterion) or by relative norm, not Darcet's absolute cutoff (Darcet: DINO v1 has essentially no artifacts and small models rarely do).
   - Report reading-set composition with and without sinks. Optionally apply Jiang et al.'s training-free register trick.
   - *Why:* tests whether "deep layers share few tokens" (RQ2) is just a shared sink plus a few content tokens. It also gives a principled high-norm category for necessity tests.
7. **Token-identity check to justify "neuron = patch".**
   - Per layer, measure how well each token predicts its own position and patch content (linear probes, as in Darcet), or its cosine to its layer-0 embedding.
   - *Why:* this pre-empts the Meister et al. 2021 and Brunner 2020 critique. If identity decays in deep layers, report RQ3 plausibility only where tokens remain identifiable.
8. **Evaluation hygiene from the ViT XAI literature.**
   - Use token dropping or attention masking for every perturbation (Jain et al. 2022). Add SaCo, signed effects (Liu et al. 2022) and a model-randomisation sanity check (Adebayo 2018).
   - Include the centre-Gaussian and random maps as baselines (SaCo shows metrics often cannot beat random).
   - *Why:* RQ4 fairness and reviewer-proofing.
9. **Report sparsity vs. sharing with SparsePR's support statistics.** Per layer, report per-query support τ_i(α) (the smallest key set holding α of the attention mass) against pooled support over the region's queries.
   - *Why:* it gives the RQ2 sharing cost a model-free, attention-mass description to place next to our behavioural agreement numbers.

---

## 10. Items not fully verified

- MiTA's venue (NeurIPS 2026) is from the project brief. The arXiv v6 page does not state it.
- Venues marked † are from memory: Geva, Ramsauer, Bietti, Bricken, Irie, Vyas, Roy, Xiong, BiFormer, Agent, NSA, Sparsifiner, EViT, DynamicViT, ToMe, A-ViT, TokenLearner, Abnar, ViT-Shapley, RISE, Grad-CAM, IG, ViT/Dosovitskiy, Vilas, Gandelsman, Darcet, Xiao, Sun, GCNet, Park & Kim, Moayeri, Jain & Wallace, Wiegreffe, Serrano, Bastings, Kobayashi, ALTI, Ferrando & Voita, Brunner, Ethayarajh, Liu 2022, Jain 2022, Adebayo, Hooker, ROAD, Zhang & Nanda, Voita, Michel, ImageNet-S, PartImageNet, Salient ImageNet, ImageNet-9, DINO, DeiT.
- Not verified: TokenTM official code (not found); LXT and LeGrad support for timm DeiT/DINO; the TIS code URL; ViT-CX mask count and cost; the Beyond Intuition IG step count; MDA authors; arXiv 2211.02912 authors; HiLRP details; the Guillaumin 2014 paper page; PartImageNet class count; and the ViT results in Sun et al. 2024.
- Not verified, practical: the timm `fused_attn` switch for exposing attention weights. Check it in the installed timm version.
