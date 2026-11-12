# v10 投稿前全面审稿与编辑检查

审阅日期：2026-11-12。对象：`output/pdf/attention_fast_weights_paper_v10.pdf`，SHA-256
`76f18c207cd5d64f18fc526ae0e9edde6cc6d30e329200677e1382f6c1dfdb0a`。
审读覆盖全部 12 页，并对照 v10 LaTeX、已有结果 CSV、统计实现、方法实现、配置和历史注册记录。
以下是审稿建议；论文、图表、实验结果和注册记录保持原状。

## 1. 总体判断与修改优先级

**建议完成一轮定点的科学定义和编辑修订，再作为最终稿提交。** 论文的主线已经清楚：
固定模型，比较 received attention 的不同读出规则；范数加权在 DeiT-S/B 上改善对齐，
index-probe relocation 失败，组合修正没有一致收益。这是一个有边界、正负结果并列的实证故事。

按 [NeurIPS 2026 审稿标准](https://neurips.cc/Conferences/2026/ReviewerGuidelines)，原创性可以体现为
对已有方法的新认识，并不要求提出新算法。本稿最值得强调的是 alignment、deletion score 和 diagnostic
proxies 的分离。主会层面的主要风险仍是：已有范数读出的增量认识是否足够重要，三个 checkpoint 是否足以
支持一般性结论，以及失败 correction 的设计是否真正对应其诊断量。这些风险不能仅靠语言润色消除。
当前材料适合定位为范围明确的受控实证研究；不宜提升为通用诊断理论或优于模型侧干预的方法。

| 方面 | 评价 | 对当前提交的影响 |
| --- | --- | --- |
| 问题与主线 | 清楚，正负结果完整 | 保留现有主线，精简重复解释 |
| 贡献定位 | 较诚实，仍有“框架已被验证”的误读空间 | 明确提出的是组织测试的框架，而非有效诊断器 |
| 技术定义 | 主体正确，但有几处实现与文字不完全对应 | 修正 affine biases、对照标定、指标命名 |
| 主张与结果 | 主要数字和结论匹配原始结果 | 不改结果；弱化少量绝对措辞 |
| 统计透明度 | registered/post hoc、margin/comparable 区分较好 | 补明 primary/descriptive、固定 pairing 与 checkpoint 条件性 |
| 图表 | Figure 2 的证据组织明显有效 | 保留结构；补自包含图注、改善附录字号 |
| 参考文献 | 31 条均找到原始或官方记录 | 2 项作者顺序/版本信息待核，数处引用范围需收窄 |
| 复现说明 | 代码与证据保存较扎实，PDF 细节仍不完整 | 增补 GAE、MLP、deletion budgets、seeds 等已有信息 |
| PDF | 正文 4 页，完整 12 页 | 页数合规，无裁切或坏的内部跳转；仍有可读性改进空间 |

优先级：P1 = 提交前应改；P2 = 建议改；P3 = 可选润色。下述建议全部可依据已有材料完成，
不要求补跑实验、改变注册预测或修改统计规则。

## 2. P1：技术定义、方法归属与证据范围

### P1-1. “exact representation” 应明确 affine biases 和表示边界

位置：§3.1、Eq. (1)、Figure 1a。当前写 `q=W_Q x, k=W_K x, v=W_V x`，并称 attention update 的表示
是 exact。实际 `split_qkv()` 调用模型的 affine QKV，模型的 attention output projection 也含 bias。
因此无偏置写法需要明确是简记，或把 bias 写入公式；两层网络的第一层权重严格为 `k_j/sqrt(d)`。
这不否定 fast-weight 视角，但 “exact” 对符号精度提出了更高要求。

建议公式：

\[
q_i^h=W_Q^h x_i+b_Q^h,\quad k_j^h=W_K^h x_j+b_K^h,\quad
v_j^h=W_V^h x_j+b_V^h,
\]
\[
o_i=b_O+\sum_h\sum_j a_{ij}^h W_O^h v_j^h,\qquad
a_{ij}^h=\operatorname{softmax}_j(q_i^{h\top}k_j^h/\sqrt d).
\]

可替换英文：

> Here, \(o_i\) is the attention-branch output before residual addition, with dropout inactive at evaluation.
> For a fixed image, each head is a two-layer network in \(q_i^h\), with input weights
> \(k_j^h/\sqrt d\) and output weights \(W_O^h v_j^h\); softmax couples the hidden activations.
> This representation is exact, whereas the proposed reading assumptions require empirical validation.

篇幅紧张时，正文注明 “affine Q/K/V projections; the output bias is omitted from Eq. (1)”，
把完整式子放附录；不要让省略项在 “exact” 下变成未声明的假设。C3 仍按现有定义，
不需要人为给 output bias 分配 patch attribution。

### P1-2. C1 Gaussian 对照实际在层求和之后标定

位置：Appendix A “Corrections and matched controls”。当前总述 “All are applied per layer and then summed”
对 C1 的 Gaussian 对照不准确。`readings.compute()` 先求和 R0，然后调用
`centre_matched(lens.mean('R0', layers))`；即以**求和后的 held-out mean map** 的标准差标定 Gaussian。
`sd(sum_l mean_l)` 通常不同于 `sum_l sd(mean_l)`，这会影响复现。

可替换英文：

> Corrections and the C2/C3 controls are computed per layer and summed over layers 9–12.
> For the C1 control, we first sum the received-attention maps and then subtract a centred Gaussian
> scaled to the spatial standard deviation of the summed held-out mean map.

建议附录给出明确式子。令 `S=sum_l s_l`、`bar S=sum_l bar s_l`，对照为
`S - G * sd_j(bar S)/sd_j(G)`，其中 Gaussian 宽度仍为现有 3.5 patches。

### P1-3. C2 blur 匹配的是未按 attention 加权的 token 位移

位置：Appendix A、Appendix C “moves the score equally far”。`readings.displacement()` 计算
`mean_{x,j} sum_p P(p|x_j)||p-j||^2`，均匀平均 held-out images 和 source token slots。
它没有用 `s_j` 作为权重，因此不能称为实际 score mass 的平均位移完全一致。

可替换英文：

> For each layer, the Gaussian width matches the probe's mean squared displacement, averaged uniformly
> over held-out images and source patch indices. The matching does not weight source patches by their attention scores.

正文的 “equal-displacement blur” 可以保留为简名，但附录需给此定义；Appendix C 改为
“a Gaussian control matched in mean squared token displacement”。该对照控制了一个平滑尺度，
并没有消除 relocation 的全部替代解释；现有失败结论仍成立。

### P1-4. Deletion score 是 SRG-style adaptation，应准确署名

位置：Appendix A “The score is the symmetric relevance gain of ...”。现有实现使用 top-1 prediction
是否保持的二元指标，横轴为删除比例，在预算 `[0,1,2,4,8,16,32,64,98,147]/196` 上梯形积分，
计算 LIF–MIF。Blücher 等人的原文以被解释类别的预测值/概率为 value function。
面积差结构相同，但观测值和 token-deletion 设置是本研究的具体化。
[原文 §2.2、Eq. (5)](https://openreview.net/pdf?id=bIiLXdtUVM)

可替换英文：

> We use an SRG-style area difference between least- and most-important-first token-removal curves,
> replacing class probability with a binary indicator that the original top-1 prediction is preserved.
> Areas are computed by trapezoidal integration over the removed-patch fraction, from 0 to 147/196.

继续使用现有 score、0.05 margin 和所有结果；只修正名称和复现定义。

### P1-5. C3 与既有 norm-based methods 共享成分，不能写成完全相同

位置：§2.1、§4.4、Appendix B 最后一段。C3 是
`sum_{h,i} a_ij^h ||W_O^h v_j^h||`；Kobayashi 的单头项支持这一设计，但其整层多头分析还使用
`||sum_h a_ij^h W_O^h v_j^h||`，先求范数再跨头求和与先跨头求和再求范数不同。
Zhu 的完整方法还包含梯度、残差处理和平滑。因此 “in practice the correction is the value-norm weighting of ...”
过于等同；value-only variant 的 comparable verdict 也不是方法等价证明。
[Kobayashi 原文 Appendix A](https://aclanthology.org/2020.emnlp-main.574.pdf)，
[Zhu 出版原文](https://journals.sagepub.com/doi/pdf/10.3233/FAIA240520)

可替换英文：

> C3 aggregates per-head norm-weighted terms related to Kobayashi et al. and to the norm-weighted
> component used by Zhu et al. It sums magnitudes before aggregating across heads and therefore omits
> cross-head cancellation. Omitting the output projection yields comparable outcomes under our decision rule.

这既保留 prior credit，也准确界定当前 score；不宜因这种区别反过来宣称提出了全新 attribution 方法。

### P1-6. ClickMe 的来源推断需要收窄

位置：Appendix A “very likely ImageNet training images, which all three models have seen”。
Fel 等人描述的整个 ClickMe collection 含 ImageNet train 和 validation 图像；
仓库的 17 个 `val-*.tfrecords` 是 **ClickMe validation split**，不是 ImageNet validation split。
在 4,002 个候选 validation 图像中只检出 2 个匹配，不能确认全部 17,408 张的原始 split 或三个模型都见过它们。
[Fel 原文 §3](https://proceedings.neurips.cc/paper_files/paper/2022/file/3d681cc4487b97c08e5aa67224dd74f2-Paper-Conference.pdf)，
[作者数据加载代码](https://github.com/serre-lab/Harmonization/blob/main/harmonization/common/clickme_dataset.py)

可替换英文：

> We use the 17,408-image ClickMe validation split released with Harmonization; the TFRecords do not
> identify each image's original ImageNet split. In our near-duplicate check, 2 of 4,002 candidate
> ImageNet validation images matched a ClickMe record. Possible overlap with ImageNet training images
> limits claims about generalisation.

Discussion 相应改为 “ClickMe images may overlap model training data.”

### P1-7. 两项书目作者顺序/版本信息待核

- [22] Ramsauer ICLR 2021：官方 PDF 的可检索文本显示 `... Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner ...`；
  当前条目把 Thomas Adler 放到了 Holzleitner 后。但最新 OpenReview PDF 和 note 直接获取受验证限制，
  本轮未确认最终文件是否与索引版本相同。**列为版本核对项，不直接判定当前 BibTeX 必错**；
  最终按所引用版本的作者名单统一。[官方会议 PDF](https://openreview.net/pdf?id=tL89RnzIiCd)
- [27] MiTA：arXiv v6 PDF 首页为 `Wen, Huang, He, Meng, Li`，arXiv 元数据和作者仓库建议引用则为
  `Wen, Huang, Meng, He, Li`，与当前 BibTeX 相同。**此项暂无法唯一确认，不应直接判当前条目错误或擅改**；
  最终按会议版/作者确认统一，并明确引用版本。
  [arXiv 元数据](https://arxiv.org/abs/2602.01219)，[PDF](https://arxiv.org/pdf/2602.01219)，
  [作者仓库](https://github.com/QishuaiWen/MiTA)

## 3. 论证、注册与统计解释

### 主要主张与证据对照

| 主张 | 现有证据 | 判断 |
| --- | --- | --- |
| C3 的 DeiT-S/B enrichment gain 为 0.038/0.065，ClickMe gain 为 0.033/0.058 | `results/correct_tests.csv` 的 C3–R0 | 支持，四舍五入正确 |
| 四个 supervised alignment cells 中三个也胜过 matched control | C3–zero-high-norm；B ClickMe 差 0.017065，comparable | 支持；必须继续保留例外 |
| C3 没有超过 deletion 的 0.05 gain margin | S/B gain 0.017313/0.028741，95% CI 都排除零 | 支持“未过 margin”；不能说“无统计差异”或“无作用” |
| DINO C3 变化小 | 最大绝对差约 0.003507；原 registered verdict 为 comparable | 支持，限当前 checkpoint/metrics/layers |
| 深层 relocation 降低九个 cells | C2–R0 的 3 models × 3 metrics 全为 lower | 支持 |
| R2 partly holds，1/4 | 两个 lower、一个 comparable、一个 higher | 支持；建议把负面分布也直接写清楚 |
| diagnostic intensity 通用地预测 gain | 仅三个 checkpoints 的 post-hoc 对应、image-level rho=0.045–0.158 | 尚未建立，现有保留限制正确 |
| token-content/position mismatch 导致 C2 失败 | 当前 probe 测 index；原有 MLP/locality checks 未隔离因果机制 | 尚未建立，应继续作为解释假设 |
| 阅读侧改动一般优于模型侧改动 | 仅一个 register setting，MiTA 训练/预处理不同 | 不支持一般性结论；现稿已基本避免 |

P2：建议在正文或 Appendix A 区分 **pre-specified comparison** 和 **primary comparison**。
历史 P7 把 `R_all–R0` 和 `correction–matched control` 设为 primary；其余 correction–R0 为 descriptive。
Figure 2 采用同一 registered decision rule 是正确的，但不表示每个点都有同等的 confirmatory 地位。

可补英文：

> The combination-versus-baseline and correction-versus-control contrasts were designated primary;
> the remaining correction-versus-baseline contrasts are descriptive. Diagnostic-to-benefit interpretations are post hoc.

P2：附录已经披露 earlier execution/diagnostic results 曾被看过；历史记录还披露了 32–64 张 smoke runs
和早期 value-norm pilot。可压缩成一句，而无需复制整个 log：

> The protocol was frozen before the formal runs, after implementation smoke tests and earlier exploratory observations documented in the registration log.

P2：bootstrap 只覆盖固定模型、固定 probes、固定数据处理和 pairing 下的 image sampling。
ClickMe 的 `roll(1)` 在 batch 内复用 human maps，因此观测行并非简单独立配对；现有 image-level bootstrap
也没有重新抽取 pairing 或重训 probes。应披露条件性，不据此直接宣称现有所有区间失效。

可补英文：

> Intervals resample image-level scores with checkpoints, probes and within-batch ClickMe pairing held fixed;
> they do not capture uncertainty from model training, probe fitting or alternative image pairings.

继续保留 existing 1,000 resamples、95% percentile intervals、no multiplicity adjustment 和所有 registered verdicts。
另一个极小的实现精度问题：代码使用 `abs(mean) < margin` 判 comparable，因此恰好等于 margin 的值可进入
higher/lower；文字 “exceeds” 可统一为 “reaches the margin”。当前已报告结果不受这个边界影响。

## 4. Writing：主线、结构与可直接采用的英文

### 4.1 结构与段落

有效主线应保持为：**先界定要检验的 reading assumptions，再报告 controlled comparisons，最后解释何种
diagnosis 未能支持何种 correction。** 目前主线可读，§4.4 拆开 contribution/other corrections/model-side
是有效改进。主要问题是：

- Abstract、Figure 1b、Figure 2f、§4.4 和 Conclusion 重复同一组 C3 gain。保留摘要和结果段的数字，
  其余位置减少同义解释；Figure 1b 若保留，明确其 teaser 作用。
- Related Work 以“方法名称+引用”列举为主。把省出的句子用于说明当前研究与 norm-based reading、
  full-block attribution、token identifiability 的具体区别。
- §3.2 第一句没有显式重复 specificity/location/contribution 的名称，依赖 figure headings；
  可用一个简短句把三者映射回来。
- §4.4 两次粗体标题相接使起首略拥挤。可将 4.4 简化为 “Correction outcomes”，保留各段首的具体结论。
- Appendix B “two further tests” 后实际列出 dose–response、shuffle、value-only 三类 post-hoc checks。
  改为 “additional post-hoc checks”。
- “cell” 适合表内计数，但正文第一次出现时应写成 “model–metric comparison”；
  “no R0 gain”“deletion-margin improvement”“diagnostic prediction”可改成读者更容易解读的完整短语。

### 4.2 摘要替换建议（保留所有主要实验结论）

> Deep-layer ViT attention maps can emphasise background patches. We ask whether changing the reading rule
> improves map quality without modifying the model. Using the fast-weight view, we organise three reading
> assumptions—image specificity, spatial correspondence and output contribution—into diagnostic proxies,
> corrections and matched controls. On frozen DeiT-S/B, norm weighting raises object enrichment by
> 0.038/0.065 and image-specific human alignment by 0.033/0.058; it also outperforms its control in three
> of these four comparisons. Gains in deletion score and DINO ViT-S remain below the registered margins.
> Probe relocation lowers all nine deep-layer model–metric scores, while the combined correction improves
> only one of four registered supervised comparisons. These results support a bounded alignment benefit
> from norm weighting and show that declining patch-index decodability is insufficient to justify score relocation.

建议把当前最后一句的两个开放问题移到 Discussion；摘要用最后一句明确“读者获得了什么认识”。
若要把 “registered supervised comparisons” 改为 “primary comparisons”，需与真实 P7 primary 定义一致。

### 4.3 贡献列表替换建议

> (1) A controlled evaluation protocol connecting three attention-reading assumptions to explicit diagnostic proxies, corrections and matched controls.
>
> (2) Evidence that norm weighting improves localisation and human alignment in DeiT-S/B, while its deletion-score gains remain below the registered margin and DINO gains remain limited.
>
> (3) A negative result showing that reduced patch-index decodability does not translate into effective score relocation, and that combining the corrections improves only one of four registered supervised comparisons.

第三条是当前最有区分度的发现；不要把 C3 本身写成新方法。

### 4.4 重要句子的定点替换

| 位置/现有措辞 | 问题 | 建议英文 |
| --- | --- | --- |
| Abstract “three of four gains exceed the matched control” | gain/control 的比较对象略含糊 | “C3 also outperforms its matched control in three of the four alignment comparisons.” |
| Appendix C “relocation is harmless” | comparable 不等于无伤害或等价 | “At layers 1–4, relocation remains comparable to received attention under the registered rule.” |
| Appendix B “The weights must match the tokens” | 一个 shuffle experiment 不证明普遍必要条件 | “Correctly assigned weights outperform shuffled weights in all four supervised alignment comparisons.” |
| §4.4 “R2 partly holds (1/4), without consistent benefit” | 只给成功数可能弱化负结果 | “Under the registered rule, R2 partly holds: one comparison is higher, one comparable and two lower.” |
| §4.5 “The post-hoc correspondence holds for contribution weighting” | holds 有验证一般关系的意味 | “The observed post-hoc pattern is consistent with the contribution account, but does not establish a general diagnosis-to-benefit relationship.” |
| §4.3 “none in DINO” | 所有层并非精确零；layer 2 mean=0.02455% | “The share is negligible in DINO (at most 0.025% across layers, and zero in layers 9–12 on this dataset).” |
| §4.5 “local content may persist after position codes weaken [Raghu]” | 文献未同时检验本稿 index-probe 下降与 provenance | “Prior work reports that deep ViT tokens retain information about their corresponding input locations under a representational-similarity probe. Whether this relates to our relocation failure remains untested.” |
| Appendix D “original v8 main-text examples” | 编辑历史不服务提交稿读者 | “the two selected examples with the smallest ClickMe indices” |

### 4.5 自定义指标与引文归属

Method 3.3 的引用应支持 ClickMe 数据来源，而不要让读者误认 own-minus-other Spearman 差分出自 Linsley/Fel。

> Using ClickMe human maps [Linsley; Fel], we define image-specific alignment as the within-image Spearman
> correlation minus the correlation with another image's human map.

same-ring enrichment 的定义也是本研究 protocol 的一部分，引用 Gao 支持 masks/dataset，而非整个新指标。
“centre-controlled” 是合适名称；保留它比改回 centre-blind 更准确。

## 5. Figures 与 tables：逐项检查

| 项目 | 论证作用与检查结论 | 可执行修改 |
| --- | --- | --- |
| Figure 1a | 首页给出固定图像下 fast-weight 表示，作用明确；vector 清楚。数学下标较小但主体标签约 8 pt。 | 保留；与 affine-bias/scale 定义同步，避免把它解释为完整 block 或最终 class attribution。 |
| Figure 1b | 有效 teaser，但其 12 个 effect/CI 与 Figure 2f 重复。误差条很窄是已有数据结果。 | 可保留重复以服务首页；若需腾正文空间，优先精简这一面板，保留首页示意图。图注补 image-bootstrap、比较方向与完整模型范围。 |
| Figure 2a–c | 同列 diagnosis→correction 的组织清楚；三种线型配合配色，灰带 layers 9–12 一致。 | 改标题为“Mean-map structure / Index decodability / High-norm attention”等直接观测量，避免 “Specificity” 高值被误读为更高 specificity；注明 b 的 probe chance 为 1/196。 |
| Figure 2d–f | 同一 x-axis range、zero line、margin band、circle/diamond、filled/open 设计合理；数据与原结果一致。 | 图注明确定义 difference=correction−comparator，Enr./ClickMe 缩写、三个 matched controls、sample units/95% interval方法、未做multiplicity adjustment；说明 open 不代表equivalence。 |
| Table 1 | 保留全部 models×metrics、controls、GAE/model-side rows；86 个数值、7 个 diamonds、3 个 bold 与原结果一致。 | caption 补 “Unmarked tested cells are comparable; GAE is descriptive”、paired CI 在 B.2，以及判定来自未四舍五入值。用 “Higher metric values are better” 区别整体↑与superscript↑。 |
| Table A.1 | 将 registered outcomes 放在一处，支持复核 R1–R3。 | R2 明示 one higher / one comparable / two lower；如加 historical P-label 映射，可一列短注，不复制整个注册log。 |
| Figure B.1 | execution test 的证据，但与 diagnostic correction 主线联系较弱，放附录正确。实际轴/图例约 5.65 pt，偏小。 | 附录不限四页，优先扩大到约 3–3.3 in 或独占一行，让轴/图例≥7–8 pt；图注补 n=5,000、明确 dotted 与其他曲线 crop ratio 不同。 |
| Table B.1 | 保留全部逐层诊断值和被 Figure 2 替换掉的 ClickMe 曲线数值，信息未丢失。 | 对 DINO rounded 0.0 加 “rounded; deep-layer values are exactly zero on this set”；四组列标题足够清楚，不必移正文。 |
| Table B.2 | 是 primary/descriptive pairwise effects、CI、verdict 的完整依据，必须保留。 | 增大 intervals/增加行高以改善阅读；caption重述margin0.02/0.05、n来源、未调整multiple comparisons，并区分 primary contrasts。 |
| Table C.1 | 列出四个失败解释及已有测试，作用明确。 | “none meets specified criterion” 需要展示criterion或链接到短附录表；补MLP architecture、validation selection、globality定义。把model顺序S/B/D直接写进caption。 |
| Figures D.1/D.2 | fixed-seed selection公开，能显示 C3变化细微与C2变平滑，未夸大视觉效果。 | 图注增加“Colours show within-map min–max-normalised scores; amplitudes are not comparable across panels”；可共享0–1color key，human map不是ground-truth causalimportance。 |
| Appendix E code | 公式与主要tensor操作对应，字体约7pt。 | 把N改为T（all tokens），与正文N=196 patches区分；清理行末长注释使行更短；补all-layer aggregation/affineprojection边界说明。 |
| Table F.1 | 原历史sanity数值完整；centre Gaussian失败对照合理。 | 继续明确是parameter sensitivity而非faithfulness证明；threshold<0.5是本研究注册规则，而非Adebayo提出的通用认证标准。 |

Figure 2 的建议英文图注（按需压缩）：

> Diagnostic proxies and correction effects on frozen DeiT-S/B and DINO ViT-S. Columns pair each proxy
> (a–c) with its tested correction (d–f). Top: layer-wise means and 95% image-bootstrap intervals;
> shading marks layers 9–12. Bottom: mean correction-minus-comparator differences over layers 9–12
> for enrichment (Enr.) and ClickMe alignment. Circles compare with received attention; diamonds compare
> with centred-Gaussian subtraction (d), displacement-matched blur (e), or zeroing high-norm tokens (f).
> Bars are 95% paired image-bootstrap intervals (1,000 resamples, unadjusted). Filled symbols meet the
> registered higher/lower rule; open symbols are comparable. Grey bands mark ±0.02, not an equivalence region.
> The diagnosis–benefit interpretation is post hoc.

增加图注长度前，先删掉其余正文对视觉编码的重复解释；n 的精确值可通过caption显式指向Appendix A，
不必将所有细节挤在主图下。原置信区间很窄，不应人为加宽或改成标准差。

## 6. References：内容支持、遗漏与格式

31 条均找到官方或原始记录。这里的“元数据核实”不表示每篇全文逐页重读：对核心方法和有疑问的断言核对了
对应章节/公式；一般背景引用核对官方摘要或导言。逐条深度和来源见本报告末尾附录。

除 P1 中待核的 [22]/[27] 外，建议处理以下范围问题：

- [7] Brunner 同时报告 token identity 大程度保留和随深度下降，只写 “can fade” 会片面。
  可写 “Input-token identity remains largely decodable in their language transformer, although decodability declines with depth.”
- [19] Kuang 研究 CNN architectural centre bias，并指出此行为在 ViT/MLP 中大程度不存在。
  当前明确写CNN尚可，但后接“our matched controls”应避免让它成为ViT同机制的证据；
  ViT评估中 prior confound 的直接依据用本研究Gaussian–ClickMe 0.41结果。
  [官方PMLR](https://proceedings.mlr.press/v306/kuang26a.html)
- [21] Raghu 的CKA空间信息结果应与本稿index-probe结果分开，见上文改写。
- [30] Zhou 是CNN的CAM；对ViT attention输出权重的直接论断只引Kobayashi即可，或明确是类比。
- [31] Zhu 支持norm-weighted成分，不能替代对完整读出算法差异的解释。
- [14]/[20]支持ClickMe来源和human alignment背景，不支持本研究新定义的own-minus-other指标。

建议补充两项相关工作，用一两句建立边界，不必新增实验：

> Whole-block token-attribution methods such as GlobEnc and ALTI account for components beyond attention
> and aggregate interactions across layers. Our analysis instead isolates a local, class-agnostic reading
> and tests its alignment and deletion behaviour under fixed checkpoints.

[GlobEnc, NAACL 2022](https://aclanthology.org/2022.naacl-main.19/)，
[ALTI, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.595/)。它们主要源于语言Transformer；
引用作概念定位，不应暗示本文已在ViT上完成这些方法的公平比较。

书目格式：

- [15]补TPAMI页码7457–7476和DOI 10.1109/TPAMI.2022.3218275；2022 online/2023 volume不是错误。
- [23]补9355–9366，[25]补10347–10357，[28]补11–20，[31]补451–458。
- 统一会议名、缩写、可用DOI/URL；arXiv条目补版本/链接，而非只给裸编号。
- 用BibTeX花括号保护专有词的大小写，如 `{ViT}`、`{CNN}`、`{ImageNet}`、`{MiTA}`、`{PyTorch}`。
- [17] Nick/Nicholas、[14] Ivan Felipe/索引中的全名是官方来源中的展示差异，不判为虚构作者；选择一致记录即可。
- [29] Zenodo DOI属DataCite，Crossref查不到不能据此判错；引用软件时另外报告实际使用的timm版本。

## 7. Appendix：复现充分性与信息分配

目前已充分交代：checkpoint名称、输入大小、crop差异、主要dataset大小、valid-pixel mask定义、
held-out 2,000/2,000 split、mean/probe/blur用途、norm threshold、ClickMe `roll(1)`与batch64、
paired bootstrap、margins、registered outcomes、post-hoc结果、fixed-seed gallery和historical sanity。
这些是优点，应保留。

需补的细节全部已存在于代码/配置，直接整理即可：

1. **GAE：** predicted top-1 logit为target；使用全部12层，mean-head positive gradient×attention，
   recurrence `R <- R + mean_h[(grad A * A)_+] R`，最后取CLS→patches；说明它与R0的class-agnostic、
   layers9–12设定不同。现有“uses gradients”不足以复现参考行。
2. **Linear probe：** cross-entropy loss、token minibatch4096、seed0、逐feature训练集mean/std、
   std floor1e-6；训练用2000 images而非2000 independent tokens。
3. **MLP check：** hidden1024、GELU、196-way output；1800/200 images train/validation，
   30epochs、Adam lr0.01、best-validation-epoch selection、原2000 test images；
   不能只写“two-layer MLP”却据此排除容量解释。
4. **Deletion：** 完整budget列表、fraction横轴、trapezoidal integration、original top-1 target、
   CLS/register kept、原positional embeddings保留；SRG式子的adaptation按P1说明。
5. **Score handling：** 明确原始逐层分数的求和与对照的特殊aggregation，C1允许负值且ranking-based metrics无需clip；
   不为让图好看而更改归一化或阈值。
6. **Shuffled-weight check：** 说明现有5个shuffle seeds在每个image上先平均后bootstrap，而非把seeds当独立images；
   C3v省略output projection；globality和probe-failure criteria给定义或指向具体表。
7. **Provenance：** 固定代码revision、config路径、checkpointrevision/hash和split文件链接；
   历史执行环境中未知项保留unknown，不把当前重建环境当成原GPU环境。

应留正文的关键信息：假设/三个公式/控制逻辑、R0 queries含CLS+patches、layers9–12、两个中心控制指标、
原margin规则、主要正负结果、post-hoc边界。正文可再加一句primary/descriptive区别，以及GAE比较范围短注。
不建议把Table 1或关键失败结果移附录来腾版面。

可压缩的信息：Appendix A关于早期Small/Tiny结果的长句、Appendix B与主文重复的C3增益、
Appendix D的v8编辑历史。省出的附录空间用于GAE/MLP/预算和criterion说明。

## 8. PDF逐页检查

| 页 | 内容 | 结果与建议 |
| --- | --- | --- |
| 1 | Title、Abstract、Introduction、Figure 1 | 无裁切/重叠。图注紧凑；贡献列表整体推到第2页，降低首页完整性。可通过压缩摘要/重复C3视觉信息改善，保留首页schematic。 |
| 2 | Contributions、Related Work、Method | §3.3完整落页；公式无越界。内容很密，新增bias说明可把长式移附录，避免缩小正文。 |
| 3 | Figure 2、Experiments 4.1–4.4 | 六面板对齐，字体最小6.4pt；正常比例可读但小。4.4标题和首个粗体结论相接略拥挤。 |
| 4 | Table 1、其余结果、Discussion、Conclusion | 正文终止于本页。表数据清楚；legend含义较密，应显式解释未标记格/GAE。 |
| 5 | References [1]–[19] | 完整、无缺字或越界，约9pt；书目专有词大小写需保护。 |
| 6 | References [20]–[31] | 完整，剩余留白正常，无须为了填满而改变结构。作者排序问题在本页。 |
| 7 | Appendix A模型/数据/诊断/方法 | 无裁切；需要修正本报告P1中的定义和来源措辞。 |
| 8 | Appendix A统计/注册、Table A.1 | 无裁切；注册规则清楚，可补primary/descriptive和conditional uncertainty。 |
| 9 | Appendix B、Figure B.1、Table B.1 | Figure B.1约5.65pt轴/图例偏小，最值得改善。表列清楚。 |
| 10 | Table B.2、C3补充测试、Appendix C起首 | 数据完整，CI排得很密；允许在附录用更多空间。 |
| 11 | Table C.1、Appendix D/gallery | 页面上方接C的浮动表，下方D.2较小；未发生内容碰撞。可扩大D.2至单独一行/页，不影响正文限制。 |
| 12 | Appendix E code、Appendix F、Table F.1 | 两节和表都完整；代码约7pt，可缩短注释而非进一步缩小字号。 |

机器检查：12页，main end位于4，references从5开始，appendix从7开始；
内部跳转目标均有效，未发现页面边界外文字、未解析引用或替代字符。所有字体资源内嵌。
本次对PDF的所有页面重新渲染并人工检查；这些版式检查不替代前述科学内容修订。

## 9. 建议的执行顺序与完成标准

1. 修P1技术定义与引用归属；不动已有结果CSV、注册记录或科学实现。
2. 收窄ClickMe来源、Raghu/provenance和C3机制解释；修“harmless/must”等绝对措辞。
3. 用现有代码补GAE/MLP/预算/seeds/primary对照说明。
4. 保留Figure2证据结构，改清图注和Table1 legend；扩大附录FigureB.1/CI文字。
5. 核定[22]/[27]引用版本及作者顺序；统一已确认的页码、链接和大小写，待核项不自动改写。
6. 最后压缩重复C3表述，重编译并验证四页正文、完整references/appendix、全部数值/verdict不变。

五维自审：贡献定位需收窄；writing以局部重排为主；实验主张有支持；evaluation范围需明确；
method定义需按P1修正。当前最值得交付给读者的认识是：**alignment gain、deletion-score gain与diagnostic
signal是不同证据，index decodability的下降不能直接支持score relocation。**

---

以下为31条参考文献逐条核查记录；保留原始来源与核查深度，以区分书目信息确认与全文内容确认。


# References [1]–[10]: official-source audit

Scope: `paper/references.bib` against v10 main text/appendix. Checked 2026-11-12. `Verified` below means author/title/venue/year (and supplied pages) match an official paper or publisher page. `Content read` states the level used to judge citation support.

| # | Status and primary source | Content read and fit to v10 |
| --- | --- | --- |
| [1] Abnar & Zuidema 2020 | Verified: [ACL Anthology](https://aclanthology.org/2020.acl-main.385/). Metadata all match; bibliography could add official DOI `10.18653/v1/2020.acl-main.385` and pages 4190–4197. | Official abstract and paper introduction: attention rollout/flow aggregates across layers and studies token mixing. Supports `Readings incorporate layers` and introductory context. |
| [2] Adebayo et al. 2018 | Verified: [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html). Authors/title/year/venue match. | Official abstract and full-paper introduction: parameter and label-randomisation tests; visual plausibility can be misleading. Supports related-work and Appendix F sanity-test rationale. The v10 median threshold is the author's own registered rule, not Adebayo's. |
| [3] Ayyar et al. 2025 | Verified as arXiv preprint: [arXiv record](https://arxiv.org/abs/2510.06070). Title/authors/year/eprint match; no journal/conference venue appears on the record. | Official abstract: combines attention maps with statistical filtering, also assesses gaze. Supports `statistical filtering` as an alternative reading. The manuscript does not claim its quantitative result. |
| [4] Bau et al. 2017 | Verified: [CVF official page](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html). Full author order/title/CVPR/year/pages 6541–6549 match. | Official abstract and introduction: unit semantics quantified by alignment to visual concepts. Supports the generic proposition that unit meaning requires measurement; it does not validate ViT patch-index probes specifically. |
| [5] Blücher et al. 2024 | Verified: [TMLR paper](https://openreview.net/pdf?id=bIiLXdtUVM). Authors/title/year/journal match. | Read §2.2 and §5.2, equations (2), (4), (5). Original SRG is `LIF−MIF` on pixel-flipping curves, with the underlying value function using a class prediction/probability. v10 Appendix A uses **binary top-1 preservation** at each budget. The algebraic area difference matches but this is an SRG-style variant, not literally the paper's metric. Related-work claim that deletion depends on occlusion is supported. |
| [6] Bondarenko et al. 2023 | Verified: [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/edbcb7583fd8921dad78adecfe06a99b-Abstract-Conference.html). Authors/title/year/venue match. | Official abstract: large activations arise partly when attention heads learn a no-op/partial residual update. Supports **analogy** to register/high-norm phenomena, but does not show the two mechanisms are equivalent. `which resemble` may overstate the connection. |
| [7] Brunner et al. 2020 | Verified: [ICLR paper](https://openreview.net/pdf?id=BJg1f6EFDB). Authors/title/year/venue match. | Official abstract and introduction: input-token identity is largely retained in a language Transformer **and** decreases with depth. The v10 phrase `Language-model token identity can fade` is selectively negative; suggest a more balanced sentence. |
| [8] Caron et al. 2021 | Verified: [CVF paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf). Author order/title/ICCV/year/pages 9650–9660 match. | Official abstract/figure 1: DINO attention maps visualized and self-supervised ViT features studied. Supports introductory map use and model identity. The released linear classifier mentioned in Appendix A is directly documented in the [authors' code](https://github.com/facebookresearch/dino/blob/main/utils.py), beyond the paper citation. |
| [9] Chefer, Gur & Wolf CVPR 2021 | Verified: [CVF official page](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html). Authors/title/year/pages 782–791 match. | Official abstract and paper introduction: propagates relevance across attention and skip connections. Supports `relevance propagation`. Also a closer citation than [10] for ViT map visualization in the first introduction sentence. |
| [10] Chefer, Gur & Wolf ICCV 2021 | Verified: [CVF official page](https://openaccess.thecvf.com/content/ICCV2021/html/Chefer_Generic_Attention-Model_Explainability_for_Interpreting_Bi-Modal_and_Encoder-Decoder_Transformers_ICCV_2021_paper.html). Authors/title/year/pages **397–406** match BibTeX; note DBLP incorrectly shows 387–396, so follow CVF. | Official paper abstract and method excerpt: extends explanation to self-/co-/encoder-decoder attention, and uses attention gradients. Supports `gradients`, GAE reference row, and a broad introductory statement, though [9] is more directly about ViT maps. |

## Recommended edits, ordered by scientific impact

1. Appendix A deletion definition: `We report an SRG-style difference between the areas under least- and most-important-first token-removal curves, using a binary top-1 preservation indicator rather than the class-probability value function of Blücher et al.` This accurately attributes the area-difference idea without suggesting exact replication of their metric. [TMLR equations (2),(5)](https://openreview.net/pdf?id=bIiLXdtUVM).
2. Related work on Brunner: `Brunner et al. found that input-token identity remains largely decodable in a language transformer, although decodability declines with depth.` [ICLR abstract](https://openreview.net/pdf?id=BJg1f6EFDB).
3. Register/no-op comparison: replace `which resemble attention heads that learn to do nothing` with `an effect functionally analogous to no-op attention heads in quantizable Transformers` only if the distinction helps the argument; otherwise omit the analogy. [NeurIPS abstract](https://proceedings.neurips.cc/paper_files/paper/2023/hash/edbcb7583fd8921dad78adecfe06a99b-Abstract-Conference.html).
4. Metadata polish: add DOI/pages for [1] and official paper URLs/DOIs where easily available. Keep venue formatting consistent. [10] needs no page correction despite the conflicting DBLP record.

## Coverage limits

All ten entries have official-source metadata checks. Content checks used official abstracts/introduction or specified method sections; not every cited paper was reread end to end. No `unverifiable` entry in this group. No obvious indispensable citation omitted within this subset; the central norm-weighting work falls outside [1]–[10] and should receive the strongest related-work treatment in the full review.


---

# Reference audit: PDF items [11]–[20]

Checked against the v10 bibliography and the cited locations in `paper/v10/main.tex` and `paper/v10/appendix.tex` on 2026-11-12. Official publisher/proceedings pages and original paper PDFs are distinguished below. No manuscript files were changed.

| No. | Item; primary source | Metadata | Support for use in v10 / action |
| --- | --- | --- | --- |
| 11 | Darcet et al. (2024), [ICLR paper](https://openreview.net/pdf?id=2dnO3LLiJ1) | Verified title, four authors, venue/year in original paper. | Original abstract and introduction explicitly identify high-norm background artifacts, trained register tokens, smoother attention maps; supports cited motivation. |
| 12 | Deng et al. (2009), [IEEE record](https://ieeexplore.ieee.org/abstract/document/5206848/) and [author PDF](https://cs.stanford.edu/groups/vision/documents/ImageNet_CVPR2009.pdf) | Verified six authors, title, venue/year; DOI 10.1109/CVPR.2009.5206848. | Supports ImageNet provenance, but counts of 5,000 sampled images are this manuscript's own data, not the reference. Add pages 248–255/DOI if harmonizing metadata. |
| 13 | Dosovitskiy et al. (2021), [ICLR paper](https://openreview.net/pdf/a4aa24aed16fcb6f23d1067f1a5ecf47d7115f63.pdf) | Verified 12 authors, title, venue/year. | Supports ViT definition. |
| 14 | Fel et al. (2022), [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/3d681cc4487b97c08e5aa67224dd74f2-Paper-Conference.pdf) | Verified title/venue/year. Author PDF says “Ivan Felipe”; proceedings index spells “Ivan F Rodriguez Rodriguez.” Bib follows paper PDF; this is an alias/display discrepancy, not a clear false author. | Original §3 uses ClickMe human maps and Spearman, so supports those. It does **not** define the manuscript's own-image-minus-previous-image metric; move citation to dataset noun and explicitly say “we define.” It says ClickMe source data span ImageNet train **and** validation, and its own evaluation uses ImageNet-validation intersection. |
| 15 | Gao et al. (2023), [IEEE record](https://ieeexplore.ieee.org/document/9933726/footnotes) and [author PDF](https://mftp.mmcheng.net/Papers/22PAMI-USSSeg.pdf) | Verified six authors in present order, title, TPAMI 45(6), 2023; pages 7457–7476, DOI 10.1109/TPAMI.2022.3218275. Online 2022 / volume 2023 is ordinary early access. | IEEE abstract explicitly identifies ImageNet-S dataset and annotations. The specific 12,419 processed masks are manuscript-specific. Add pages and DOI in bibliography for consistency. |
| 16 | Jain & Wallace (2019), [ACL Anthology](https://aclanthology.org/N19-1357/) | Verified two authors, title, NAACL 2019; pages 3543–3556, DOI 10.18653/v1/N19-1357. | Original NLP experiments support “whether attention explains predictions is debated,” particularly when paired with Wiegreffe/Pinter; avoid implying it is a ViT-specific experiment. |
| 17 | Jiang et al. (2025), [official NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/51bae6441380c4c4cab87f430be89ac2-Abstract-Conference.html) and [paper PDF](https://openreview.net/pdf/bdddccfe50b9f9d6dbfcbb6389b315509524cfe6.pdf) | Verified title, NeurIPS 2025, coauthors. **First author discrepancy:** proceedings says Nicholas Jiang; submission PDF says Nick Jiang; Bib uses Nick. Prefer final proceedings “Nicholas Jiang” for consistency with official record. | Original abstract and method support training-free/test-time register mechanism. Eight selected units and numeric results are this manuscript's own experiment, not Jiang's. |
| 18 | Kobayashi et al. (2020), [ACL Anthology](https://aclanthology.org/2020.emnlp-main.574/) and [full paper](https://aclanthology.org/2020.emnlp-main.574.pdf) | Verified four authors, title, EMNLP 2020; pages 7057–7075; DOI 10.18653/v1/2020.emnlp-main.574. | Supports single-head transformed-value norm `||alpha W_O^h v_j^h||`; paper Appendix A defines **whole multi-head** measure as `||sum_h alpha^h W_O^h v_j^h||`. Manuscript C3 uses `sum_h alpha^h ||W_O^h v_j^h||`, which discards cross-head cancellation. Its exact formula is a related head-wise aggregation, not identical to their whole-layer norm. Reword Appendix B's “the correction is the value-norm weighting of Kobayashi et al.” accordingly. |
| 19 | Kuang et al. (2026), [PMLR ICML 2026 proceedings](https://proceedings.mlr.press/v306/kuang26a.html) | Verified three authors, title, ICML 2026, PMLR 306:61236–61266. This is a real published paper by the current date. | It demonstrates CNN architectural centre bias and explicitly says this behaviour is **largely absent in ViTs/MLPs**. The manuscript correctly says “CNN centre priors” but the immediate “our matched controls” transition risks treating this as direct evidence about ViT; pair with own empirical Gaussian–ClickMe 0.41 finding or clarify analogy. |
| 20 | Linsley et al. (2019), [ICLR paper](https://openreview.net/pdf/2420540d74d8fb3505c5d06ca0708a06d29a95d8.pdf) | Verified four authors, title, ICLR 2019. | Supports original ClickMe collection and human maps, not the manuscript's custom own-minus-other Spearman difference. |

## ClickMe split provenance: highest priority wording fix

The manuscript uses the 17-shard ClickMe `val` release. [Harmonization author code](https://github.com/serre-lab/Harmonization/blob/main/harmonization/common/clickme_dataset.py) labels it **ClickMe validation** and exposes 318 separate training shards; it does not identify original ImageNet train/validation membership for these records. [Fel et al., §3](https://proceedings.neurips.cc/paper_files/paper/2022/file/3d681cc4487b97c08e5aa67224dd74f2-Paper-Conference.pdf) state that the broader ClickMe collection includes ImageNet training and validation images. The manuscript's 2 near-duplicates among 4,002 ImageNet-validation candidates indicate little detected overlap with ImageNet validation under its own matching rule. They do not prove every ClickMe image is from ImageNet training or has been seen by each frozen model. Current Appendix A final sentence is too certain.

Suggested replacement: “We use the 17,408-image ClickMe validation split released with Harmonization; the TFRecords do not identify each image's original ImageNet split. In our near-duplicate check, 2 of 4,002 candidate ImageNet validation images matched a ClickMe record. Possible overlap with ImageNet training images limits claims about generalisation.” In Discussion, “ClickMe images may overlap model training data.”

## Suggested norm-reading replacement

Replace Appendix B's final sentence with: “The value-only variant is comparable in every cell (differences −0.010 to 0.000). Our C3 sums the per-head norm-weighted terms associated with Kobayashi et al. (2020); unlike their whole-layer norm, this aggregation does not capture cancellation across heads.” If Zhu et al. use the exact same head-wise aggregation, add that citation after checking their formula separately.

## Suggested citation-scope replacement

In Method 3.3: “Using ClickMe human maps [Linsley; Fel], we define image-specific alignment as the within-image Spearman correlation minus the correlation with another image's human map.”


---

# Reference audit: final PDF [21]–[31] (2026-11-12)

主审复核说明：[22]因最新OpenReview PDF直接获取受限，以下PDF索引文本与现条目的顺序差异列为待核；以本报告§2 P1-7为最终判断。

Scope: bibliographic metadata against primary publication, proceedings, arXiv or author-maintained software citation; claim support checked against the article text/abstract where indicated. `metadata` means bibliographic fields matched; `content` means I inspected source claims, not necessarily every page. No manuscript edits.

| No. | Status | Primary source | Metadata and content check |
|---|---|---|---|
| 21 Raghu et al. 2021 | Metadata verified; content qualified | [NeurIPS paper](https://proceedings.neurips.cc/paper/2021/file/652cf38361a209088302ba2b8b7f51e0-Paper.pdf) | Five authors/title/NeurIPS 2021 correct. §7/Fig. 9 finds high-layer CLS-trained ViT token CKA strongest at corresponding input patch, so spatial-information preservation is supported. This is **not** a direct token-content provenance test, nor proof of what happens when the present position-code probe weakens. |
| 22 Ramsauer et al. 2021 | **Check version/author order**; content supported by indexed official text | [Official ICLR/OpenReview PDF](https://openreview.net/pdf?id=tL89RnzIiCd) (web direct access blocked by challenge; official search-index text read); [arXiv record](https://arxiv.org/abs/2008.02217) | Official conference PDF first-page author order: Ramsauer, Schäfl, Lehner, Seidl, Widrich, **Thomas Adler**, Lukas Gruber, Markus Holzleitner, Kreil, Kopp, Klambauer, Brandstetter, Hochreiter. Current `.bib` places Adler after Holzleitner. Current arXiv version has 16 authors; conference PDF lists 13. Title and ICLR 2021 correct. Paper explicitly establishes relation of modern Hopfield update and transformer attention. |
| 23 Schlag et al. 2021 | Metadata/content verified | [PMLR official page](https://proceedings.mlr.press/v139/schlag21a.html) | Authors/order/title/ICML 2021 correct; official PMLR 139:9355–9366 (pages omitted in `.bib`). Abstract establishes formal equivalence of **linearized** self-attention and fast-weight controllers. Manuscript's wording “connects linear attention to fast-weight programming” is accurate. |
| 24 Schmidhuber 1992 | Metadata/content verified at abstract level | [DOI/publisher](https://doi.org/10.1162/NECO.1992.4.1.131); [Crossref record](https://api.crossref.org/works/10.1162/neco.1992.4.1.131) | Crossref resolves title/sole author/Neural Computation 4(1):131–139, Jan. 1992. Citation supports fast-weight memories controlled by another network. DOI absent from `.bib`, optional to add. |
| 25 Touvron et al. 2021 | Metadata/content verified | [PMLR official page](https://proceedings.mlr.press/v139/touvron21a.html) | Authors/order/title/ICML 2021 correct; official PMLR 139:10347–10357 (pages omitted). Supports DeiT models. |
| 26 Walmer et al. 2023 | Metadata/content verified | [CVF official page](https://openaccess.thecvf.com/content/CVPR2023/html/Walmer_Teaching_Matters_Investigating_the_Role_of_Supervision_in_Vision_Transformers_CVPR_2023_paper.html) | Four authors/order/title/CVPR 2023:7486–7496 correct. Abstract directly supports that supervision changes ViT attention behaviour. |
| 27 Wen et al. 2026 MiTA | **Author-order discrepancy between primary sources**; content verified | [arXiv version 6 record](https://arxiv.org/abs/2602.01219), [v6 PDF](https://arxiv.org/pdf/2602.01219); [authors' code/model zoo](https://github.com/QishuaiWen/MiTA) | Five authors, title and arXiv ID correct. **PDF title page order is Wen, Huang, Wei He, Xianghan Meng, Li**, whereas arXiv abstract metadata, authors' GitHub citation and current `.bib` put Meng before He. Resolve against final NeurIPS proceedings or ask authors; do not silently choose. The paper itself describes input-conditioned two-layer fast-weight MLP and MiTA's top-k/landmark mechanism. Authors' repository lists NeurIPS 2026 acceptance (25 Sep 2026) and DeiT-S model checkpoint, supporting “released MiTA top-k checkpoint”. |
| 28 Wiegreffe & Pinter 2019 | Metadata/content verified | [ACL Anthology official](https://aclanthology.org/D19-1002/) | Authors/order/title/EMNLP-IJCNLP 2019 correct; official pages 11–20 (omitted in `.bib`). Supports debate over attention as explanation, and authors explicitly argue against a universal negative claim. |
| 29 Wightman 2019 | Metadata verified; content software source | [Maintainer citation file](https://github.com/rwightman/timm/blob/main/CITATION.cff) | Ross Wightman, PyTorch Image Models, 2019, DOI 10.5281/zenodo.4414861, and current repository URL all match. DOI is a Zenodo/DataCite software DOI, not Crossref; Crossref non-JSON/404 is **not** an error here. Citation supports use of timm models. |
| 30 Zhou et al. 2016 | Metadata verified; citation-support concern | [CVF official paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) | Five authors/order/title/CVPR 2016:2921–2929 correct. Paper is class activation mapping in CNNs with global-average-pooling classifier weights. It provides an **analogy** that activations alone miss output weights, but not direct evidence about ViT attention. Current Method 3.2 cites Zhou jointly with Kobayashi after an attention-output assertion; retain only Kobayashi there or explicitly label the CAM analogy. |
| 31 Zhu et al. 2024 | Metadata/content verified; support needs precise scope | [Publisher-hosted ECAI paper](https://journals.sagepub.com/doi/pdf/10.3233/FAIA240520); [University publication record](https://research.manchester.ac.uk/en/publications/noise-free-explanation-for-driving-action-prediction/) | Crossref DOI resolves correct title, five authors/order, 2024; publisher/Manchester volume 392 correct. ECAI 2024 proceedings frontmatter shows starts p.451, next begins p.459, so pp.451–458 (omitted in `.bib`). Paper's §3 weights attention by norm of **transformed value**, then also applies class gradient, residual handling and smoothing. Supports prior use of the norm component, not identical whole C3 reading. |

Priority checks: resolve the [22] version/author-order issue and [27] inconsistent primary-source author order. Narrow Raghu [21] use in §4.5. Narrow or remove Zhou [30] citation for the ViT-attention-output premise. Clarify that [31] shares a norm-weighted component but its full method differs. Add missing proceedings pages/DOIs to selected entries for consistent bibliography formatting when revising `.bib`; these omissions do not make the entries incorrect.

Suggested sentence for §4.5: “Prior work reports that deep ViT tokens can retain information about their corresponding input locations under a representational-similarity probe [21]. Whether this relates to the failure of our index-based relocation remains untested.”
