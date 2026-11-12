# v13 终稿审稿意见（2026-11-13）

**审阅对象**　提交版《How Should We Read ViT Attention Maps? A Fast-Weight Perspective》，论文提交 `8905a22`，PDF SHA-256 `c158c66c945d3024e1fafe1bf7ca58419e101f023886a92d5e9cf33a1d69d708`。审阅范围包括完整 PDF、实际进入编译的引文、课程 rubric、注册记录与支撑主要结论的结果表。以下为对现稿的评价及替换建议。

## 总体评价和课程结构

**建议小修后作为课程终稿。** 论文的主张有一个明确的实验含义：即使冻结同一个 ViT，用不同规则把内部计算画在空间网格上，也会得到适合不同解释目标的排名。Figure 2 的负结果和 Table 2 的同图双目标比较最能体现作者自己的贡献。在一个 checkpoint、一个窄域上观察到的 CLS 结果，足以作为这个观点的实例。若要宣称普遍的定位与分类忠实性权衡，则还需要独立模型和事前固定的跨域检验。现稿的主要问题通过编辑即可解决。

课程文件第 1 页的格式要求是使用 NeurIPS preprint 模板并有清楚的标题、小标题和编号章节。第 2 页规定最多四页正文，包含图表，评分人可以只读正文。第 2 页 §2.2 对第 2–4 页的章节清单使用原话 “Please refer to the following structured suggestions for the project report.” 第 5 页将 contribution、clarity、depth、literature review 和 formatting 等权。依据这些文字，可以自拟标题，也可把结果分两节或合并 Discussion 与 Conclusion。建议保留现有顺序；若希望评分人扫一眼即可定位，可将 §3 命名为 *Methods for reading a frozen attention head*、§4 命名为 *Experiments on three reading assumptions*，§5 继续标出 exploratory。标题修改属可选项。

另有一项与排版独立的提交要求。第 1 页在与外部工作的关系一节要求学生单独署名，协助仅来自任课教师或 GTA。文件没有明说如何处理生成式 AI 协助。鉴于本项目实际使用了 AI 协助写作和实验，最终提交资格及是否需要披露，以课程组对这条规定的解释为准。

| Rubric 维度 | 审稿判断 |
| --- | --- |
| Contribution | 三种假设、匹配对照和目标明确的干预构成可辨认的贡献。前人已给出 fast-weight 表示、norm weighting 和多目标评估。作者自己的发现最清晰地体现在固定图像、模型和层后改变接收 query 所得到的结果。 |
| Clarity | 首页即交代问题与假设。摘要的 “predicts” 和结论的 “all queries” 超出了实际统计量或对照粒度，优先修订。 |
| Depth | 既有三个冻结 checkpoint，又有可失败的对照、直接干预、配对不确定性和负结果。三项代理测的是不同量，不能把诊断与修正后收益拼成因果链。 |
| Literature review | 核心背景覆盖充分。§2 的枚举可以少一两句，换成近期方法与本文之间的具体比较，尤其是 Ayyar、TokenTM 和 Value Zeroing。 |
| Formatting | 已满足正文四页、两图两表、完整附录及关键代码结尾的要求。Figure 2a 的图注需纠正，Figure 2 的小字与点型是可读性瓶颈。 |

审稿尺度参考 NeurIPS 对技术质量、清晰度、意义和原创性的评价；课程格式和页数按本课程文件判断。[NeurIPS Reviewer Guidelines](https://neurips.cc/Conferences/2025/ReviewerGuidelines)

## 各节如何服务研究定位

| 位置 | 判断与建议 |
| --- | --- |
| 标题与摘要，第 1 页 | 标题准确。摘要讲清三项假设、固定模型和配对对照，保留 C2 的失败是优点。把 “human alignment” 具体化为 “alignment with human ClickMe maps”；“better predicts” 改为图内排名与指定干预的相关性；写明 Oxford-IIIT Pet 上的 DeiT-S。 |
| 引言与贡献，第 1 页 | 背景亮点是合适的动机，但首次提到 attention map 可以早点说明本文主要研究 *received attention*。第一项 “give an observable diagnostic” 没有说清 fast-weight 为何必要。可以用下文建议的贡献句替换，无需再加一段。 |
| Related work，第 2 页 | 三段分别谈表示、空间位置与解释目标，顺序合理。区分语言 token 身份和视觉位置信息，避免让引用 [7,25] 看起来证明同一对象。[20] 的 CNN 结果应加对象限定。Ayyar 和 TokenTM 都已分别考虑空间/人类与扰动评价；本文的差异在于同层、同图与特定 CLS 路径。 |
| 方法，第 2 页 | 式 (1) 正确区分了 attention branch、输出偏置和残差。可在式中首次定义 d 为每头 query/key 的维数。C1 的方差比、C2 的位置解码、C3 的 high-norm share 都只是操作化。C3 的诊断用 block-input norm，权重用 projected-value norm，两者宜连续说明。 |
| 原始研究，第 3 页 | 当前先读效果，最后才知道哪些对比事前固定。把 primary/descriptive 的两句移到 Design 段末，让读者在看 R0 与对照差异前先知道它们的证据地位。Figure 2 的列对应关系有价值，应保留。 |
| 深入案例，第 4 页 | §5 已把范数放在一个具体子问题下，随后追问接收端。建议将动词 “coincide” 改为操作性描述。控制结果支持自身空间分配在这些读法中有用，未估计 attention 与 norm 的统计交互项。 |
| 讨论与结论，第 4 页 | 首尾呼应，限定一个 checkpoint 和图像域是必要的。原文 “all queries can carry useful spatial information” 容易被理解为每个 query 都携带有用信息，而实验只比较全部聚合和 CLS 单独聚合。建议回到同图、两种聚合、两个目标。 |

已经没有由 value norms 主导全文的结构性问题。残留的旧口径主要在附录的 *corrections/fixes* 和 *map quality*。注册名称宜保留，在介绍它们时可将 *candidate readout changes* 作为统称，并在结果句写清具体指标。

## 主张、证据和裁决

| 主张 | 核对的证据 | 可以推出的结论 |
| --- | --- | --- |
| 图像共享结构、空间解码和输出项可分别检验 | C1 对标准差匹配的 Gaussian，C2 对平均平方位移匹配的 Gaussian blur，C3 对高范数 patch 的 R0 置零。固定模型，原始输入不变。 | 可以比较三种读法及指定替代解释。无法证明三项假设是独立、穷尽的机制。 |
| 高亮可能包含跨图共享结构 | M1 是跨图汇总的空间方差比；深层均图本身大多呈径向结构。C1 相对 R0 唯一达到增益门槛的是 DINO ClickMe，对 Gaussian 对照仍是 comparable。 | 可说共享结构存在，且所测减均图操作未显示同时超越 R0 和对照的一致收益。“均图呈径向”和“单图方差可由均图解释多少”是两个问题。 |
| 深层 C2 不利 | C2−R0 在 3 个模型 × 3 个指标上均为 lower。在 DeiT-S 和 DINO 也不及位移匹配的模糊。初层两项对齐指标仍落在 R1 的 comparable 类。 | 失败的是当前以索引概率重新分配得分的规则。它并未直接检验、否定 token 与原始像素空间对应关系。 |
| Norm weighting 改善监督 DeiT 的对齐 | DeiT-S/B 相对 R0 的 enrichment 分别 +0.038/+0.065，ClickMe +0.033/+0.058；4 个格子里 3 个也通过高范数对照。DeiT-B ClickMe 相对该对照 +0.017。Top-1 deletion 的 +0.017/+0.029 均低于 0.05。 | 可以报告两个监督 checkpoint 的空间对齐改进。优先比较 correction−control，同样大小的正号在不同终点有不同意义。 |
| 注册裁决 | R1 的 9 格全部 comparable；R2 是 1 higher、1 comparable、2 lower，按既定规则 partly holds；R3 的 4 格在两个 size-S 参照中均无 higher。注册文件记载已见 pilot，附录 A 单列既有观察的复核。 | 现稿裁决准确。“Comparable” 包括估计不精确及小于实践门槛，两种情况都不能当作统计等效。MiTA 跨模型结果还混合训练和预处理差异。 |
| 图像自身的长度位置有用 | 383 个有效裁剪；C3 相对 donor 和同环打乱的 enrichment 分别高 0.022、0.021。192 个 donor 对的重抽样也给出 +0.022 [0.019, 0.026]。只有 norm 的读法更差，未投影 norm 略好。 | 属于新增探索。支持空间分配与保留 attention 变化对这几种读法的作用。不能据此认定输出投影是改进的必要条件或像素具有因果重要性。 |
| 查询聚合的作用依赖解释目标 | 在 128 张图、单一 DeiT-S 末层、同一组得分上，CLS-only 减 all-query 的 margin Spearman、enrichment、AUROC 分别为 +0.041、−0.033、−0.091。对应 95% 配对区间为 [0.026, 0.057]、[−0.043, −0.023]、[−0.110, −0.074]。 | 干预直接测量固定最高两类的 logit margin 对这一内部向量删除的响应；readout 与该响应的吻合仍是排序相关。掩膜复核发生在看到干预结果之后，已披露。分类终点与 CLS 读法共享路径，本来就有结构上的匹配，结果不证明通用的分类忠实性。 |

核对依据包括 [注册与时间次序](../../docs/preregistration.md)、[原始比较](../../results/correct_tests.csv)、[负对照](../../results/negative_controls.csv)、[模型侧参照](../../results/variants.csv)、[CLS 比较](../../research/v13/cls/three_endpoint_contrasts.csv) 和 [事后掩膜协议](../../research/v13/cls/addendum_protocol.md)。

## 可以直接采用的英文

**摘要（可整体替换）。**

> How should a vision transformer's attention map be read as image evidence? Using the fast-weight view of attention, we distinguish three assumptions about a patch score: its dependence on the image, its spatial correspondence to the input, and its relation to the attention output. We test each with a matched control on frozen models. Index-based relocation reduced all three deep-layer scores, while value weighting improved foreground and ClickMe alignment in supervised DeiT. In an exploratory test on pet images, the final-layer CLS-only readout ranked patches more consistently with the change in class margin after their contributions to CLS were removed. The all-query readout gave better foreground localisation on the same images.

可在引言首次使用 “attention map” 时明确定义本文分析的 received attention。

**贡献第一项。**

> We use the fast-weight representation to distinguish three choices behind received-attention maps: which structure recurs across images, where a contextualised token's score is displayed, and how its activation is converted to output magnitude.

**§4 先说明实验地位。**

> The primary contrasts compared each modified readout with its matched control and the combined readout with received attention. Individual comparisons with received attention were descriptive, except for the registered R1 controls.

**案例和结论。**

> Holding each head’s value lengths fixed, both donor reassignment and within-ring shuffling reduced foreground enrichment relative to the image’s own assignment.

> On the same pet images, aggregating all queries gave higher foreground scores, whereas the CLS-only ranking agreed more closely with the measured effects on the final CLS pathway.

## 图表、附录和复现

- **Figure 1，第 1 页。** 保留原图和右侧三个问题。图注的 “The map places softmax activations at the source slots” 可具体为 “Received attention sums activations over queries and heads and places each sum at its source slot.” 这样能区分单个 head 的计算和画到整张图的聚合。
- **Figure 2，第 3 页。** 最应修的是统计量名称。图注把 (a–c) 都称为 “per-layer means”，而 (a) 在 `scripts/make_tables.py` 先汇总 residual 与 total，再计算 1 − Σresidual/Σtotal；(b,c) 才是按图取均值。建议替换为 “Panel (a) shows the pooled share of spatial variance explained by the held-out mean map. Panels (b,c) show mean index-decoding accuracy and mean attention share on high-norm patches.” 正文 “mostly radial” 应指向附录 B 的 radial 分析，Figure 2a 没有画径向方差。图中 6.4–6.5 pt 图例与刻度偏小；保持原结构，缩短标签、略放大字和空心点，实心点的判据写成 “Filled markers indicate a paired interval excluding zero and an absolute mean difference of at least 0.02.”
- **Table 1，第 4 页。** 保留七个读法和差值方向。入选 384 张、有效 383 张。图注的 CI 是按图重抽，正文另报 donor-pair bootstrap，两者已交代。正文可直接报告 donor-pair 区间，少重复一次均值。顺序上将 donor 与 ring 控制紧接 C3 可让读者先看到关键对比。
- **Table 2，第 4 页。** 最后一行直接报告同图有符号配对差，是应保留的核心。各列量纲不同，宜只在列内比较。表注的 “Entries are image means” 可细化为 “Rows give mean per-image scores. The last row gives paired differences.” 两张表层数、分母均不同，现表注已写清。
- **附录 A–B，第 7–12 页。** 配置、阈值、数据预处理及统计方法足够解释正文。12,419 是 ImageNet-S 验证集总数；前景富集有效数为两种 DeiT 各 12,150、DINO 12,144，可在正文样本数后简记。随机化结果在 B 中位置合理。Table B.3 保留完整配对区间；正文可在报告九个 C2 格子的全指标结果时指向它。
- **附录 C–F，第 13–18 页。** C 为失败的 C2 追因，D 的示例图有固定选样说明，E 标出探索、面积上限及额外终点，F 以关键实现收尾。正文在解释定位失败时可引 App. C，在提及定性示例时可引 App. D。几何分析适合保留在 E，无需回到首页。
- **公开复现入口。** 第 9 页给出仓库根链接并将新增脚本描述为随附源码。核对远端 heads 可见 v12，未见本地 v13 研究分支。如果课程提交附 [源码补充包](../../dist/v13_source_supplement.zip)，请写明它覆盖到 v12 基线的用法；若只交 PDF，宜先准备能访问的固定 v13 版本链接。此项影响读者复现新增案例。
- **PDF 验证。** 重新渲染并逐页查看全部 18 页。第 1–4 页正文含图表，第 5–6 页为 [1]–[36]，第 7–18 页为 A–F。正文及附录无图表截断或公式遮挡。复核了 83 个内部链接目标、29 个嵌入字体、编译日志及最后一段代码；实际编译文件的引用集合与 36 个书目键完全一致。

## 逐条文献审查

每一条按此次 PDF 的编号列出作者顺序、题名、年份、出版载体、已有卷页或 DOI、实际引用句和可核验来源。BibTeX 所列 10 个 DOI 包括 9 个 Crossref 标识与 1 个 DataCite/Zenodo 项目标识。论文集中无页码的条目、会议版与预印本的差异、代码项目的概念 DOI 分开处理。

- [参考文献 1–12](v13_refs_01_12.md)
- [参考文献 13–24](v13_refs_13_24.md)
- [参考文献 25–36](v13_refs_25_36.md)

**36 条均能确认作品身份。** 书目更新优先看 [3]，引文范围优先看 [4]、[7,25] 和 [20]，版本冲突看 [31]。[3] 的 arXiv 2025 引用真实；ICPR 2026 会议版已于 2026 年 8 月在线，LNCS 16816，591–605。出版社给出的引用年份为 2027，换成正式版时应使用整条出版记录。该文同时评价扰动和人类注视，因此“两个评价目标”本身不宜列作本文首创。[Springer 正式论文](https://link.springer.com/chapter/10.1007/978-3-032-31666-0_39)

[20] 的主要证据是 CNN 的架构性中心偏置，作者在其测试的 ViT/MLP 中观察到该现象大体缺失。本文实测的径向先验可以独立讨论。推荐 “Kuang et al. observed centre-focused saliency in CNNs and much less of it in the ViTs they tested. We therefore measure recurring spatial structure directly in our checkpoints.” [ICML 正式记录](https://proceedings.mlr.press/v306/kuang26a.html)

[31] 的 arXiv v6 摘要页和 PDF 首页给出不同的第三、第四作者顺序，NeurIPS 2026 海报的署名顺序符合摘要页，但题名为 *as Scalable Fast Weights*，arXiv 为 *via Scalable Fast Weights*。当前条目完整对应摘要版本，建议标注 v6；本次未取得足以统一三者的最终 proceedings PDF。[arXiv v6 记录](https://arxiv.org/abs/2602.01219v6) · [同版 PDF](https://arxiv.org/pdf/2602.01219v6) · [NeurIPS 官方海报](https://neurips.cc/virtual/2026/loc/sydney/poster/150679)

**总评。** 稿件满足课程论文的结构与篇幅要求，核心发现有证据支持。最后一轮编辑应先修 Figure 2a、摘要和结论的测量用语，再整理关键引文与源码入口。以上修改不依赖新训练。
