<!--
文件 03：论文草稿（2026-11-07 版，从作者的在线文档导出）

给 Claude Code 的说明：
- 这份草稿告诉你“论文主张什么、用到哪些数字”。它是第一阶段数字核对（01 号文件 X8）和第二阶段“保留哪些实验”（02、04 号文件）的依据。
- 不要修改这份草稿。发现问题时写进你的汇报。
- 标记含义：[PENDING] 等第一阶段结果；[VERIFY] 作者自己核对文献；[TO DRAW] 作者手画；[TO RENDER] 由你出图。
- 开头“写作说明”和每节斜体“中文要点”是写给作者的，不是论文内容。
-->

# Attention as Fast Weights: Reading, Diagnosing, and Correcting Attention Maps in Vision Transformers

Nov 7, 2026 · @deng

## 写作说明（中文，定稿前删除）

这是 4 页正文的第一版英文草稿。章节按课程 rubric 第 2.2 节的建议排列；叙事沿用你的参考文档：读 → 诊断 → 修正 → 为什么有的修不了。

### 结构决定

| 决定 | 理由 |
| --- | --- |
| 章节按 rubric：摘要、引言、相关工作、方法、实验、讨论、结论、附录 | 老师按 rubric 打分。参考文档的页数表漏了“相关工作”，而文献综述占五分之一的分数，所以单独成节 |
| 正文 3 图 1 表，分别对应读、诊断、修正 | 沿用参考文档 |
| 图 2(b) 改为同一张图的人类图、R0、C3、C2 对比 | rubric 要求定性结果；一张图同时展示“哪个修正有效、哪个无效”，也适合视频 |
| 表 1 每个修正行下紧贴它的替身行；SRG 移到附录，正文一句话 | 沿用参考文档 |
| 主张用有条件的写法：不写“两个可分离的原因”；“变成全局信息”等补充分析出来再定 | 目前没有证据支持“可分离”；C2 失败的机制还没验证 |
| 不做 DINOv2-reg | DINOv2 只有 Large 及以上才出现高范数 token（Darcet 等 2024），S/B 上形成不了对比；写进局限 |
| 讨论加伦理考虑，结论加未来工作和更广泛影响 | rubric 第 6、7 项明确列出 |
| 引言末尾主动说明 Kobayashi 2020 和 SNNA 的先例 | 自己先说，比审稿人指出好 |

### 页数预算（正文 4 页，含图表）

| 部分 | 页 |
| --- | --- |
| 标题 + 摘要 | 0.35 |
| 1 引言 | 0.5 |
| 2 相关工作 | 0.35 |
| 3 方法（含图 1） | 0.85 |
| 4 实验（含图 2、图 3、表 1） | 1.65 |
| 5 讨论 | 0.2 |
| 6 结论 | 0.1 |

草稿目前比 4 页长约 15%，转成 LaTeX 后再统一压缩。图 2、图 3 要做成宽而矮的横排图。

### 标记

- **\[PENDING\]**：等补充分析结果。
- **\[VERIFY\]**：写进终稿前核对原文。
- **\[TO DRAW\] / \[TO RENDER\]**：要画或要出的图。
- 每节开头斜体的“中文要点”，定稿前删除。

## Abstract

*中文要点：问题 → 视角 → 读、诊断、修正各一句结果 → 意义。*

Attention maps are the most common way to explain what a vision transformer (ViT) looks at, yet how to read them is rarely checked, and photographers' centre bias can inflate their apparent agreement with humans. We revisit attention as a fast-weight network: for every image, fixed (slow) weights write a temporary two-layer network whose hidden units are the tokens, so an attention map is a reading of hidden activations. This lens lets us test whether the read units are used, state the premises behind reading them, and correct the reading. On DeiT-S, DeiT-B and DINO ViT-S, restricting every region to the 16 units it reads keeps 72%, 83% and 62% of top-1 predictions, against 9%, 30% and 4% for random units. Of three premises, two fail in deep layers: tokens lose their positional identity in all models, and in supervised models a high activation does not mean a high contribution. Weighting attention by each unit's output contribution improves object focus and human alignment in supervised ViTs, beats removing high-norm tokens and test-time registers, and recovers 31–62% of the last-layer degradation; relocating tokens with position probes instead makes maps worse **\[PENDING: because …\]**. Part of what makes deep attention maps look poor is how we read them.

## 1 Introduction

*中文要点：注意力图难以验证、会被中心偏差夸大；核心问题是“变差有多少来自读法”；把注意力看成快权重网络，读法的三个前提就能检验；三个研究问题、三个贡献；主动说明先例。*

Attention maps are the most common window into what a vision transformer (ViT) \[Dosovitskiy et al., 2021\] attends to \[Abnar and Zuidema, 2020; Caron et al., 2021\]. Two problems make them hard to trust. First, a map is a picture: it does not show that the model computes with the highlighted tokens \[Jain and Wallace, 2019; Wiegreffe and Pinter, 2019\]. Second, agreement with human importance maps can be inflated by photographers' centre bias. In our setup, a fixed centred Gaussian that never looks at the image reaches a Spearman correlation of 0.41 with ClickMe human maps \[Linsley et al., 2019\], in the same range as values reported for ViT attention rollout \[Fel et al., 2022\] **\[VERIFY: their protocol differs\]**.

These problems matter most in deep layers. Attention maps of supervised ViTs become noisy and drift away from objects near the output \[Darcet et al., 2024\], and the usual response is to change the model, for example by adding register tokens \[Darcet et al., 2024; Jiang et al., 2025\]. We ask a different question: **how much of what makes an attention map look poor comes from how we read it, rather than from where the model looks?**

To make the reading itself testable, we revisit attention as a fast-weight network \[Schmidhuber, 1992; Schlag et al., 2021\]. For each query, an attention head computes a two-layer network whose first-layer weights are the keys, whose activation is the softmax and whose second-layer weights are the values. The keys and values are written anew for each image by fixed (slow) projection weights. Each token is therefore a hidden unit, and an attention map is a reading of hidden activations drawn at the units' patch positions. This reading silently assumes three premises: the activations are specific to the image (M1), unit *j* still stands for patch *j* (M2), and a large activation means a large contribution (M3).

We use this lens to run one complete loop on three pretrained ViTs (DeiT-S, DeiT-B and DINO ViT-S), organised by three research questions. **RQ1 (read):** are the units read from attention used by the model? **RQ2 (diagnose):** where do the three premises fail? **RQ3 (correct):** can corrections derived from the lens improve the reading, and how do they compare with changing the model? Every correction is paired with a simpler stand-in that it must beat, all main metrics give zero credit to a centre prior, and all decision thresholds were fixed before the main runs.

Our contributions are:

1. **A lens that makes reading premises measurable.** An execution test shows that the units read from attention are used: restricting each region to 16 of 197 units in all layers keeps most predictions, while random units do not.
2. **A diagnose–correct protocol with matched controls.** We measure each premise layer by layer and pair each lens-derived correction with a stand-in, under centre-controlled metrics.
3. **Findings on what can be fixed by reading.** Positional identity is lost in deep layers of all models, and activation differs from contribution in supervised models. Weighting attention by each unit's output contribution helps exactly where the second failure occurs, beats removing high-norm tokens and test-time registers, and recovers 31–62% of the last-layer degradation; relocating tokens with position probes makes maps worse **\[PENDING: mechanism\]**.

Weighting attention by value-vector norms was proposed by Kobayashi et al. \[2020\] for language models and used in ViT explanations by SNNA \[2024\]; we do not claim it as new. What is new is deriving it from the lens, testing it against a matched control and against model-side fixes, and predicting where it should not help.

## 2 Related Work

*中文要点：三组文献，每组先讲前人做了什么，再写空白和我们的位置（rubric 要求的 gap analysis 和 positioning）。*

**Attention as explanation.** Whether attention explains predictions has long been debated \[Jain and Wallace, 2019; Wiegreffe and Pinter, 2019\]. For transformers, attention rollout \[Abnar and Zuidema, 2020\] and gradient-weighted propagation (GAE) \[Chefer et al., 2021\] combine attention across layers, norm-based analyses weight attention by value-vector norms \[Kobayashi et al., 2020; SNNA, 2024\], and recent work filters ViT attention maps statistically \[arXiv 2510.06070\]. Received attention also serves as a token-importance score in pruning \[Goyal et al., 2020; Kim et al., 2022\]. Explanations are evaluated by deletion faithfulness \[Blücher et al., 2024\] and by agreement with human maps \[Fel et al., 2022\], and centre bias is known to confound saliency evaluation \[Kuang et al., 2026\]. *Gap:* these works compare readings, but do not test the premises behind a reading, and human-alignment results for ViTs are rarely controlled for photographers' centre bias.

**Fast weights and attention as memory.** Linear attention is equivalent to fast-weight programming \[Schmidhuber, 1992; Schlag et al., 2021\], softmax attention is retrieval in a modern Hopfield network \[Ramsauer et al., 2021\], and many sequence layers can be seen as test-time regressors \[Wang et al., 2026\]. Ramsauer et al. also measured how many tokens each BERT head averages over, and MiTA \[2026\] trains models whose regions read only their top-*k* keys. *Gap:* the lens has been used to design or characterise models, not to audit how attention maps are read. We apply MiTA-style reading sets post hoc to standard ViTs as an execution test, and use the lens to derive the premises of a reading.

**Sinks, registers and token identity.** Supervised and large ViTs repurpose low-information patches as high-norm "scratch" tokens that attract attention \[Darcet et al., 2024\]; registers, trained or added at test time, take over this role \[Darcet et al., 2024; Jiang et al., 2025\], and attention sinks move rather than vanish \[Fesser et al., 2026\]. Deep tokens also lose their identity \[Brunner et al., 2020\], become more similar to each other \[Wang et al., 2022\], and absorb global semantics \[Su, 2026\]. *Positioning:* we compare such model-side fixes with reading-side corrections on the same metrics, and test whether lost identity can be repaired by relocating scores.

## 3 Methodology

*中文要点：用公式说清视角；由视角推出三样东西：读取名单与执行检验、三个前提的测量、三种修正及替身；最后交代模型、数据和唯一训练的探针。*

**Figure 1 \[TO DRAW\].** *Attention as a fast-weight network.* Left: for each image, the slow weights write keys (first-layer weights) and values (second-layer weights); each token is a hidden unit. Middle: an attention map reads the units' activations and draws unit *j* at patch *j*. Right: the three premises of this reading (M1–M3), how each is measured, and the correction (C1–C3) and stand-in derived from it.

### 3.1 Attention as a fast-weight network

Let the 197 tokens entering block ℓ be the CLS token and 196 patch tokens. Head *h* computes queries, keys and values by fixed projections, and for query *i*:

```latex
o_i = W_O^{h} \sum_{j} a_{ij} \, v_j, \qquad a_{ij} = \operatorname{softmax}_j\!\left(q_i^{\top} k_j / \sqrt{d_h}\right)
```

Read as a function of the input q\_i, this is a two-layer MLP: first-layer weights {k\_j}, activation softmax, second-layer weights {v\_j}, output projection W\_O^h. The weights K and V are written anew for every image by the slow weights W\_Q, W\_K, W\_V; they are fast weights, and token *j* is hidden unit *j*. The standard "attention received" map reads the total activation of each unit and draws it at patch *j*:

```latex
s^{\ell}_j = \sum_{h} \sum_{i} a^{\ell h}_{ij}
```

We call this reading **R0** and, unless stated otherwise, sum it over layers 9–12. Unlike in an MLP, softmax makes the units compete, so the analogy is approximate (Section 5).

### 3.2 Read: reading sets and the execution test

We split the 14×14 patch grid into 16 regions. Each region averages its queries, scores all keys, and keeps the top-*k* as its reading set; CLS keeps its own top-*k*. We then rerun the model with every query allowed to attend only to its region's set, in all 12 layers at once, and record whether the top-1 prediction matches that of the unmodified model. Baselines are random sets of the same size (five seeds) and removing attention altogether; *k* ranges from 4 to 128.

### 3.3 Diagnose: three premises

- **M1, the reading is image-specific.** We estimate the model's mean map s̄ per layer on held-out images and report the share of within-image spatial variance it explains: one minus the variance of s − s̄ over the variance of s, pooled over images.
- **M2, unit *j* stands for patch *j*.** A linear probe per layer predicts each token's original patch index (196 classes) from its normalised representation; we report top-1 accuracy on held-out images.
- **M3, a large activation means a large contribution.** We report the share of R0 that falls on high-norm tokens: patches whose block-input norm exceeds three times the image's median at that layer.

### 3.4 Correct: three corrections and their stand-ins

Each premise gives one correction:

```latex
\text{C3: } c^{\ell}_j = \sum_{h}\sum_{i} a^{\ell h}_{ij}\,\lVert W_O^{h} v^{\ell h}_j \rVert_2 \qquad \text{C2: } r^{\ell}_p = \sum_j s^{\ell}_j \, P_\ell(p \mid x^{\ell}_j) \qquad \text{C1: } s^{\ell} - \bar{s}^{\ell}
```

C3 (for M3) follows directly from the network view: a hidden unit's contribution is its activation times the size of its outgoing weight. C2 (for M2) moves each unit's score to where the probe says the unit now stands. C1 (for M1) removes the image-independent part. A correction could help for a simpler reason, so each has a **stand-in** it must beat: for C3, setting high-norm tokens to zero; for C2, a Gaussian blur whose mean squared displacement matches that of C2 in each layer; for C1, subtracting a centred Gaussian scaled to the mean map's spatial spread. The pre-registered combination of all three corrections is reported in Appendix D.

### 3.5 Models, data and training

We study frozen DeiT-S and DeiT-B \[Touvron et al., 2021\] and DINO ViT-S with its official linear head \[Caron et al., 2021\], each with its own preprocessing. Two model-side variants are evaluated with the same protocol: DeiT-S with a test-time register that takes over the 8 FFN units producing high-norm tokens \[Jiang et al., 2025\], and MiTA-DeiT-S trained with top-*k* attention \[MiTA, 2026\]. The only trained components are the 36 linear probes (12 layers × 3 models; Adam, learning rate 0.01, batch 4,096, 30 epochs, seed 0), trained on 2,000 held-out ImageNet validation images; another 2,000 held-out images give the mean maps and the probes' test accuracy. No held-out image is used for evaluation.

## 4 Empirical Analysis

*中文要点：先交代数据、指标和统计规则（评分标准里的 Experimental Setup）；然后按 读 → 诊断 → 修正 的顺序给结果，每节第一句就是结论。*

### 4.1 Experimental setting

**Data.** The execution test uses a fixed set of 5,000 ImageNet validation images. Object alignment uses ImageNet-S \[Gao et al., 2022\]: all 12,419 images for enrichment and a fixed 2,000 for deletion. Human alignment uses the 17,408 released ClickMe maps \[Linsley et al., 2019; Fel et al., 2022\]. We found that these images are almost certainly from the ImageNet *training* set (only 2 of 4,002 held-out validation images reappear in them), so the models have seen them; we report ClickMe numbers with this caveat.

**Metrics.** *Deletion faithfulness* is the SRG score of Blücher et al. \[2024\]: the area between the least- and most-important-first deletion curves. *Same-ring object enrichment* compares the score on object and background patches at the same distance from the image centre. *Image-specific human alignment* is the Spearman correlation with the ClickMe map after removing each map's dataset mean. Both alignment metrics give 0 to any map that does not depend on the image, such as a centred Gaussian, so a high score cannot come from centre bias alone. GAE \[Chefer et al., 2021\] is reported as a reference, not as a competitor: it uses gradients and class information that our readings do not.

**Statistics.** All differences are paired over images, with 1,000 bootstrap resamples (seed 0) and 95% intervals. A difference counts as *higher* or *lower* only if its interval excludes 0 and it exceeds a margin fixed in advance (0.02; 0.05 for deletion); otherwise it counts as *comparable*. Predictions, margins and the main comparisons were fixed before the runs (Appendix C). Two negative controls behave as predicted: C3 leaves DINO unchanged, and C2 leaves layers 1–4 unchanged (all absolute differences ≤ 0.004). All readings pass the model-randomisation sanity check \[Adebayo et al., 2018\] (median Spearman to the original ≤ 0.37 after full randomisation).

### 4.2 RQ1: the units an attention map reads are the ones the model executes

**Each region needs to read very few units, and the units the attention map picks are the right ones.** With *k* = 16 units per region (about 8% of 196) in all 12 layers, the model keeps 72% / 83% / 62% of its top-1 predictions (DeiT-S / DeiT-B / DINO). Random sets of the same size keep 9% / 30% / 4%, and removing attention leaves 0.1%–0.2%. Reading sets beat random sets at every *k* from 4 to 128 in every model (Figure 2a). Restricting one layer at a time costs much less (Appendix B): the test is strict because it changes all layers at once.

**The map is meaningful, but it is a weak heat map.** After centre bias is removed, R0 still prefers object patches (enrichment 0.140 / 0.080 / 0.199) and agrees with humans (0.157 / 0.107 / 0.208), while every image-independent reference scores about 0. GAE is higher in all nine cells (for example, deletion 0.345 vs 0.236 on DeiT-S). The next two sections ask why the gap exists.

**Figure 2.** *Read.* (a) Top-1 agreement with the unmodified model as the number of units read per region grows, for reading sets, random sets and no attention; three models. (b) \[TO RENDER\] Two randomly chosen ClickMe images: human map, R0, C3 and C2 (layer 12).

### 4.3 RQ2: two of the three premises fail, in different places

**Unit *j* stops standing for patch *j* in every model.** The linear probe recovers the patch index for almost every token in layers 1–5, then accuracy falls: 18% / 32% / 36% at layer 12 (DeiT-S / DeiT-B / DINO).

**Large activation stops meaning large contribution, but only in supervised models.** In DeiT-B, 43% of R0 at layer 11 falls on high-norm tokens; in DeiT-S, about 9% at layers 8–10; in DINO, 0% at every layer. These tokens have values only 0.2–0.3 times as long as ordinary patches, so they receive attention but write little.

**The image-independent part is small.** The model's mean map explains only 3%–6% of the spatial variation in supervised layers 6–12, and up to 17% in DINO. It is almost a set of concentric rings around the centre (82%–98% of its variance is explained by distance from the centre).

No single premise lines up with the drop in R0's quality at the last layer (Figure 3d): M2 declines smoothly, M3 peaks at layer 11 in DeiT-B, and M1 is smallest at layer 12. The premises describe *where* the reading can mislead; whether fixing them helps is a separate, testable question.

**Figure 3.** *Diagnose,* per layer for three models. (a) M1: share of spatial variance explained by the mean map. (b) M2: position-probe accuracy. (c) M3: share of R0 on high-norm tokens. (d) R0's image-specific ClickMe alignment.

### 4.4 RQ3: only contribution weighting beats its stand-in

*中文要点：这一节是全文的主结果。用一张表说清：每个修正正下方就是它的替身；只有 C3 既胜过 R0 又胜过替身，而且只在前提失效的模型上。*

**Table 1.** Object enrichment / image-specific ClickMe alignment (layers 9–12; higher is better). Each correction is followed by its stand-in (└). Intervals are about ±0.002 (enrichment) and ±0.003 (ClickMe). Deletion scores, intervals and the combined readings are in Appendix D.

| Reading | DeiT-S | DeiT-B | DINO |
| --- | --- | --- | --- |
| Centre Gaussian / mean map | 0.000 / 0.000 | 0.000 / 0.000 | 0.003 / 0.000 |
| **R0** (attention received) | 0.140 / 0.157 | 0.080 / 0.107 | 0.199 / 0.208 |
| **C3** contribution weighting | **0.178 / 0.190** | **0.146 / 0.165** | 0.196 / 0.207 |
| └ zero high-norm tokens | 0.152 / 0.168 | 0.121 / 0.148 | 0.199 / 0.208 |
| **C2** relocation by probe | 0.041 / 0.084 | 0.027 / 0.050 | 0.084 / 0.143 |
| └ matched blur | 0.095 / 0.152 | −0.008 / 0.068 | 0.189 / 0.226 |
| **C1** subtract mean map | 0.138 / 0.165 | 0.077 / 0.087 | 0.201 / 0.252 |
| └ subtract centre Gaussian | 0.133 / 0.168 | 0.056 / 0.062 | 0.200 / 0.253 |
| + test-time register | 0.152 / 0.169 | — | — |
| top-*k* training (MiTA) | 0.100 / 0.115 | — | — |
| GAE (reference) | 0.204 / 0.234 | 0.196 / 0.213 | 0.221 / 0.234 |

**C3 is the only correction that helps, and it helps where the lens says it should.** On both supervised models, C3 raises enrichment by +0.038 / +0.065 and ClickMe alignment by +0.033 / +0.058 (DeiT-S / DeiT-B), all *higher*. It is more than removing the high-norm tokens: C3 beats that stand-in on enrichment in both models (+0.026 / +0.025) and on ClickMe in DeiT-S (+0.022). On DINO, which has no high-norm tokens, C3 changes nothing (all differences ≤ 0.004), as predicted. On deletion, C3 is *comparable* everywhere (+0.017 / +0.029 / 0.000). C3 also recovers 31%–62% of the last-layer drop in the supervised models; no other correction recovers any (Appendix D). \[PENDING: C3 with ‖v‖ instead of ‖W\_O v‖; C3 with value norms shuffled within each image.\]

**C1 does nothing beyond removing centre bias.** It is *comparable* to R0 on enrichment in all models. Its one gain, on DINO ClickMe (+0.044), is matched by subtracting a plain centred Gaussian (0.252 vs 0.253): because the mean map is almost radial, "remove the image-independent part" here means "remove centre bias". This agrees with the small M1 share in Section 4.3.

**C2 makes every reading worse, usually worse than a blur of the same size.** C2 is *lower* than R0 in all nine cells; for example, DINO enrichment falls from 0.199 to 0.084, while a blur that moves mass the same distance only lowers it to 0.189. So the probe does not just spread the score out; it sends the score to wrong places. Section 4.5 asks why.

**Changing the model does not beat changing the reading.** A test-time register on DeiT-S is only *comparable* to R0 (+0.012 / +0.013), against +0.038 / +0.033 for C3 on the same model. Training with top-*k* attention (MiTA) is *lower* (−0.040 / −0.042).

**Combining corrections is not robust.** The pre-registered order C3 → C2 → C1 is *higher* than R0 in only one of four supervised cells (DeiT-B ClickMe, +0.030). The order C3 → C1 → C2 beats the pre-registered order in all nine cells, but we did not fix it in advance and use it for no claim. \[PENDING: pairwise combinations, especially C3 + C1.\] Our recommendation is therefore to use C3 alone.

### 4.5 Why relocation fails \[PENDING\]

*中文要点：这一节待补充实验结果。按结果二选一保留下面两段之一；页数不够时压成一句话并入 4.4，完整分析放附录 E。所有内容都要标为“事后、探索性”。*

The lens offers a simple explanation. If a deep token has *moved*, a position probe should find its new place. If instead the token has become *global*, mixing information from the whole image, it has no single place, and any relocation spreads its score over wrong patches. The probe's outputs point to the second case: at layer 12 of DeiT-S, the largest probe probability averages only 0.27. All analyses in this section were added after the main results and are exploratory.

**\[Version A, if supported\]** Within each layer, tokens that are more similar to the image's mean token (excluding high-norm tokens) are less accurately located by the probe and have flatter probe outputs, and C2 moves their scores farther. A two-layer MLP probe raises accuracy but does not make C2 better than R0. This suggests that deep tokens become global rather than relocated: M2 fails because the token has lost its place, not because it has moved to another one.

**\[Version B, if not supported\]** We could not find a single cause. Probe errors are \[spatially structured / not structured\], and a stronger probe \[does / does not\] change the result. We conclude only that, with probes of this capacity, relocation is not a safe correction, and we leave the mechanism open.

## 5 Discussion

*中文要点：用“诊断—修正是否一致”把三个结果串起来；给出可操作的建议；坦白写局限和伦理。*

**Diagnosis predicts correction, in two of three cases.** The loop gives a simple check: a correction should help only where its premise fails, and only by more than its stand-in. This holds for two premises. M1 holds (the image-independent part is small), and C1 adds nothing beyond removing centre bias. M3 fails only in supervised models, and C3 helps only there, by more than zeroing the high-norm tokens. M2 is the exception: it fails in every model, yet its correction hurts everywhere. A failed premise is therefore not enough; the correction must also match *how* it fails. A token that has become global cannot be put back at one place (Section 4.5).

**What this means in practice.** (i) When reading attention from a supervised ViT, weight each unit by its contribution, ‖W\_O v\_j‖; it costs one norm per token and needs no gradients. (ii) Check a new correction against a stand-in of the same strength, not only against the uncorrected map: in our data, a matched blur alone raises DINO's ClickMe score from 0.208 to 0.226, so a metric can reward smoothing. (iii) Do not read deep attention maps as "patch *j* is important"; past layer 6, the probe says unit *j* often no longer stands for patch *j*.

**Limitations.** First, softmax makes the units compete, so attention is not exactly a two-layer MLP; our tests check what the analogy predicts, not that it is exact. Second, we study three ImageNet classifiers at sizes S and B, each from a single trained run. High-norm artifacts in DINOv2 appear only at ViT-L and above \[Darcet et al., 2024\], so we could not test trained registers at our scale. Third, our probes are linear; a stronger probe may find position information that ours misses. Fourth, the ClickMe images were probably seen during training. Finally, our readings are class-independent and remain well below GAE; we aim to explain attention maps, not to replace gradient methods.

**Ethical considerations.** Better-looking explanations can raise trust in a model more than they should. We therefore report where every correction fails, compare each with a simple stand-in, and fixed our tests in advance. We use only public models and datasets; the human maps were collected by Linsley et al. \[2019\] under their own consent procedures.

## 6 Conclusion

*中文要点：三句话总结 + 未来工作 + 更广影响。*

Viewing an attention head as a per-image two-layer network turns attention maps into a reading of hidden units, with three premises that can be measured. The units this reading picks are the ones the model executes, but two premises fail in deep layers: tokens lose their patch identity in every model, and high-norm tokens absorb attention without contributing in supervised models. Only the correction for the second failure, weighting each unit by its contribution, improves the map beyond a matched stand-in, and only where that failure occurs.

**Future work.** Test the lens on larger models with trained registers (DINOv2 at ViT-L), replace the position probe with a reading that does not assume one place per token, and apply the same read–diagnose–correct loop to the FFN, which is also a two-layer network.

**Broader impact.** The fast/slow-weight view gives a common language for explaining what attention computes. Its main use is to warn when an attention map should not be trusted, and to give a cheap fix when it can be.

## Appendix plan

*中文要点：正文 4 页放不下的都放这里。正文每次引用附录都要对得上。*

- **A. Implementation and reproducibility.** Code and run folders; hardware (Colab L4, float32, deterministic); preprocessing per model; probe training details; statement on AI assistance (code and experiments run with an AI coding assistant under the author's design; all claims checked by the author).
- **B. Extensions of RQ1.** One layer at a time: only layer 1 cannot lose attention (Extension B). Head versus body patches (Extension C), with ratios reported per model. Top-*k* maps versus full maps (median Spearman 0.90–0.91).
- **C. Pre-registration and checks.** Predictions P1–P7 and their outcomes; margins; negative controls P4a and P4b; reproduction (maximum difference 0.0003); randomisation sanity check (15/15 pass).
- **D. Full Table 1.** All three metrics with intervals; references (random, centre Gaussian, mean map, GAE); combined readings in both orders; last-layer recovery table.
- **E. Exploratory analysis of C2 \[PENDING\].** Probe error structure, probe prior, where C2 moves mass, within-layer globalisation analysis, MLP probe.

**Removed from this report** (kept in the repository): the corner-shift test, a full comparison with Grad-CAM / IG / Rollout, Oxford Pets and Hard ImageNet, distance-matched deletion, compute cost.

## To-do before submission

- [ ] Fill every \[PENDING\] when the supplementary results arrive: Section 4.5 (choose version A or B), C3 checks and C3 + C1 in 4.4, Appendix E; then update the abstract, contribution 3 and Section 5.
- [ ] Figure 1 \[TO DRAW\] (author); Figure 2b \[TO RENDER\] (human map, R0, C3, C2 at layer 12; two random ClickMe images from the saved gallery); check Figures 2a and 3 for font size at print width.
- [ ] Verify every citation, especially the MiTA and SNNA author lists and the protocol behind Fel et al.'s 0.38–0.39.
- [ ] Port to the NeurIPS LaTeX template (`paper/main.tex`) and trim the main text to 4 pages; move detail to appendices before cutting claims.
- [ ] Write the AI-assistance statement; make the repository public (or give the TA access) before the deadline.
- [ ] Final read: every number in the text matches Table 1 or a cited appendix table.
