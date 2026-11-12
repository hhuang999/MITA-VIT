# v12 最终审稿：课程 rubric、全文编辑与逐条引文学术审查

日期：2026-11-12。审阅对象：`output/pdf/attention_fast_weights_paper_v12.pdf`，15 页，
SHA-256 `9f8107f18e0bf06e229f6fa9009675218c44453cc6dfe5326b0b7a83b2c5d5b1`。
同时核对 LaTeX、33 条 BibTeX、注册文件、关键结果 CSV 和统计/指标实现。论文与参考文献保持原状。

## 1. 总体判断与课程 rubric

**新定位已经成立，适合继续作为课程最终稿的主线；建议提交前做一轮小幅定点修订，
不需要再重组整篇论文。** 本稿的核心工作是借助已有 fast-weight 表示，明确从 token 间
attention 到 spatial importance 的三个阅读假设，并用代理量、候选修正和匹配对照组织检验。
正结果、负结果和证据边界共同构成分析贡献。数据集构建、fast-weight 表示本身和范数加权
算法均不是当前稿件声称的新贡献，这个区分基本贯穿全文。

课程要求是 workshop-level 四页正文，references/appendix 可延伸，并按 contribution、clarity、
depth、literature review、formatting 五项等权平均。摘要中的数字是示例，不是强制要求。
本次以课程规定为格式依据，以 [NeurIPS 2026 Reviewer Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines)
中质量、清晰度、意义和原创性的标准检查论断与表达。该指南认可对已有方法的新洞察，
同时要求说明它们增加了什么可验证的理解。

以下为编辑性估计，不能替代任课教师实际评分：

| Rubric 项 | 目前判断 | 主要依据与扣分风险 |
| --- | --- | --- |
| Contribution | A−，约 85–89 | 研究问题有价值，三个假设与对照提供清楚的分析组织；与既有 norm/token-mixing 工作相比，新增知识的独立性仍可写得更具体。 |
| Clarity | A−/A，约 88–92 | 短摘要、方法三问和正负结果主线清楚；R1 范围歧义、压缩的 run-in 标题及引言过短影响首次阅读。 |
| Depth | A，约 90–94 | 多模型、空间对照、行为指标、负对照、参数随机化和 post-hoc 检查较充分；机制没有被证实，稿件已基本诚实披露。 |
| Literature review | A−/A，约 87–91 | 三个主题和 33 条来源覆盖关键线索；部分段落仍偏于逐项罗列，Kuang 的相反证据和 Brunner/Raghu 的支持范围值得说得更明确。 |
| Formatting | A，约 90–94 | 正文恰好四页，矢量图、表格、references 和 A–E 附录完整；局部重复、表前后断句和附录留白可优化。 |

五项平均对应约 88–92，即 A−/A 区间；目前不宜仅凭工作量或引用数量评为 A+。
若按 NeurIPS 主会完整研究的贡献门槛，主要风险仍是影响范围和独立新知识，而不是英语是否足够“高级”。

## 2. 新定位是否贯穿全文

| 部分 | 判断 | 具体评价 |
| --- | --- | --- |
| 标题 | 通过，保留 | *Interpreting ViT Attention Maps Through a Fast-Weight Lens* 清楚承诺一种分析视角，未许诺新注意力架构或全面解释忠实性。 |
| Abstract | 基本通过 | 113 词覆盖问题、已有表示、三个假设、测试设计、主要正负结果及意义。结果句应明确 deep-layer received attention 的范围。 |
| Introduction | 方向一致，论证略薄 | “contextualised tokens → image evidence 需要假设”是正确切入点；缺一条直接研究问题，以及 fast-weight 表示为什么引出这三问的桥梁。 |
| Contributions | 基本通过 | framework / controlled tests / bounded insights 三项层级合理；第 1 项可直接列三个假设，第 3 项可强调新增认识而不是只列实验成败。 |
| Related Work | 明显符合新定位 | 依次交代表示来源、读图/信息混合工作、评价的局限。§2.2 仍较拥挤，需要更清楚的逐句归属与比较。 |
| Method | 通过 | §3.1 先给表示，§3.2 分开三个操作性问题，§3.3 定义评价，避免把已有 C3 成分写成新算法。 |
| Experiments | 基本通过 | “diagnostics show / corrections help / tests establish”区分良好。§4.1 R1 的压缩句造成实质范围歧义，应修正。 |
| Discussion | 通过 | 明确区分 correction benefit 与 interpretation validity，列出 checkpoint、pairing、ClickMe 重叠等限制。 |
| Conclusion | 通过，小修 | 与正负结果一致；“index-based relocation fails”应加 deep-layer/this setting 范围，且可再次点明诊断代理不等于解释验证。 |
| Appendix | 基本通过 | B 的参数随机化归位合理，E 的 key code 收尾符合要求；C 的问句标题和 B 的剂量关联小标题仍有旧的机制解释语气。 |

没有发现正文还在把“新算法”或“reading fixes generally outperform model fixes”作为主贡献。
“correction”作为方法简称可以保留；第一次出现时用 **candidate correction** 会更准确，因为 C1/C2/R_all
并未被实验普遍支持为有效修正。

## 3. 提交前应优先处理的具体问题

### P1：§4.1 的 R1 范围歧义

现句：*DINO norm weighting and relocation at layers 1–4 remain comparable.*

它容易被读成“两种方法都只在 DINO 的 1–4 层测试”。注册与结果实际为：
C3 在 DINO 的 9–12 层，三项指标 comparable；C2 在三个模型的 1–4 层，两项对齐指标 comparable。
Appendix A/Table A.1 正确，`results/negative_controls.csv` 的九个 cell 也正确。应仅修正文句：

> **R1:** norm weighting remains comparable to R0 in DINO at layers 9–12 on all three metrics;
> relocation remains comparable at layers 1–4 in all three models on the two alignment metrics.

这是一项范围澄清，所有预测、margin、cell 数和 verdict 保持不变。

### P1：文献 [29] 的作者顺序/版本冲突须保留明确状态

arXiv 元数据与作者 GitHub 推荐 BibTeX 的顺序是 Wen, Huang, **Meng, He**, Li，符合 v12；
新版 HTML/PDF 首页为 Wen, Huang, **He, Meng**, Li。两种都是直接来源，不能通过多数票或
未经依据的替换来解决。当前条目可以解释为引用作者推荐的 arXiv 记录，但不能宣称其作者信息
已与所有官方版本完全一致。提交前如能确认目标版本，统一题名、顺序、年份与版本链接；
暂时无法确认则继续保留现条目并在核查记录中注明冲突，不擅改。详见 [23–33 核查报告](v12_refs_23_33.md)。

### P2：在引言直接提出研究问题，并补表示到三问的桥梁

首页约四行介绍后即进入图与贡献，学术问题尚依赖摘要帮助读者推断。建议用下列两句中的
核心内容替换当前 “We analyse received attention …” 一句，避免再加一大段背景：

> We ask which assumptions connect internal attention to spatial importance, and which corresponding
> map changes improve alignment under matched controls. The fast-weight view makes tokens
> image-dependent units, separating questions about image specificity, spatial correspondence and output magnitude.

该表述把三个维度定位为分析组织，而非声称它们由某个定理唯一导出。

### P2：Abstract 和 Conclusion 补范围，保留短摘要的重点

当前摘要有 *On DeiT and DINO … in supervised DeiT* 的重复，且 *relocation degrades map quality*
未指出深层 setting。没有必要重新加一串数字。可选 104 词替换稿：

> Attention maps are often read as spatial importance, although their weights describe interactions
> between contextualised tokens. We use the established fast-weight representation of an attention
> head as an image-conditioned two-layer network to organise three reading assumptions: image
> specificity, spatial correspondence and output contribution. Each is examined through a diagnostic
> proxy, a candidate correction and a matched control on frozen DeiT and DINO models. For deep-layer
> received attention, norm weighting improves supervised DeiT alignment, whereas index-based
> relocation degrades map quality. Deletion-score gains remain below the registered practical margin.
> The contribution is a structured framework for examining how internal attention computations are
> translated into spatial explanations.

贡献列表第 1 项可写：

> An analytical framework separating image specificity, spatial correspondence and output
> contribution when attention is read as spatial importance.

贡献列表第 3 项可写：

> Bounded empirical insights: alignment gains do not imply deletion gains, and index decodability
> does not by itself validate spatial relocation.

后半句是证据用途的限制，并非断言已证实 C2 的失败原因。结论可相应写：

> Our analysis framework links three reading assumptions to controlled map comparisons.
> In the tested deep-layer setting, norm weighting improves DeiT alignment, while index-based
> relocation and combined corrections lack consistent benefits. These results separate useful
> map changes from validated interpretations and motivate evaluating alignment and deletion separately.

### P2：Related Work 进一步做批判性综合，保持现有三主题结构

不建议新增一个大节或仅为了增加条目数扩文献。三个具体修订更有价值：

1. Brunner [7] 的原始对象是语言 token identity；Raghu [23] 才提供 ViT 空间信息证据。
   可写：*Token-identity information can survive contextual mixing in language transformers [7],
   and ViTs can retain spatial information [23]; neither result establishes token-content provenance.*
2. Kuang [20] 不只是“CNN 中发现中心偏置”，还报告所测 ViT 的对应结构性现象很弱或不存在。
   可写：*Architectural centre bias has been reported for CNN saliency but was largely absent in
   the ViTs studied by Kuang et al.; our controls address spatial priors without assuming a shared mechanism.*
   [PMLR 原论文](https://proceedings.mlr.press/v306/kuang26a.html) 摘要及 §3.3 支持这项限定。
3. “Unit meaning requires measurement [4]”像泛化格言，与 CNN semantic-unit 测量的具体文献脱节。
   可改为：*Where unit-level studies measure semantic alignment [4], we test operational
   assumptions behind spatial attention maps.* 避免说本稿测量了 token 的语义意义。

§2.2 末可用一句定位收束，替换部分并列介绍：
*Our focus is the joint evaluation of reading assumptions, matched controls and distinct map-quality outcomes.*
不使用 “the first” 或 “no previous study” 这类尚未由全面检索证明的优先权断言。

### P2：主文要更明确处理统计区间的范围

95% bootstrap、0.02/0.05 margin、comparable 非 equivalence、未做多重比较校正、固定 checkpoint/probe
及 post-hoc 标注均已披露，属于强项。但 ClickMe 的 other-image 控制复用 batch 内 human maps，
同一张 human map 同时进入两项得分，行得分具有依赖。固定 pairing 并不消除这种依赖；
逐行 image bootstrap 没有直接估计 pairing 引入的协方差。因此不能把图中的窄区间外推为
跨随机配对、跨训练重复的精确覆盖率。Appendix A 已提及依赖，主文可将一条限制写得更直接：

> Intervals condition on fixed checkpoints and probes; image-level resampling does not capture
> training variability or dependence induced by the reused ClickMe pairings.

保留原统计协议与所有 verdict；本轮范围内不重算或重新分类结果。若将来扩展为正式研究，
再将 batch/pairing-aware 不确定性作为独立分析，而不是在终稿前暗改注册统计。

### P2：补一条既有的 MiTA 有效样本细节

Appendix A 列 ImageNet-S enrichment 的 n=12,150（DINO 12,144），但没有列 MiTA 比较的有效交集。
`results/variants.csv` 显示 MiTA 自身 enrichment n=12,144，与 DeiT-S 的 paired contrast n=12,118。
可在模型侧比较段或 Appendix B 对应段补：

> The MiTA enrichment contrast uses the intersection of valid images (n = 12,118).

这说明为什么不能总是直接用全体均值差来复原 paired contrast，也让已承认的非受控比较更透明。

### P3：清理局部压缩感与机制语气

| 当前表述 | 问题 | 建议 |
| --- | --- | --- |
| §4.3 标题后紧接另一条长加粗 topic sentence | 连续粗体像两个标题挤在一行 | 将 §4.3 缩为 *Correction outcomes.*，再另起 topic sentence；或去掉第二处加粗。 |
| *Three comparisons also beat its control* | its 指代和 comparisons 范围不够直接 | *C3 also exceeds its control in three of the four supervised alignment comparisons.* |
| *weak dose–gain associations* | dose 没有在论文中定义，略带因果剂量语气 | *weak associations between high-norm attention and alignment gains*。 |
| Appendix B: *More high-norm attention, more gain* | 标题强于随后微弱相关性 | *Association between high-norm attention and alignment gain (post hoc).* |
| Appendix C: *Why does probe relocation fail?* | 标题承诺原因，正文承认未解决 | *Additional checks on relocation failure.* |
| *R2 partly holds* | 属于历史注册类别，容易与通常“假设获部分证实”混淆 | 保留类别，在紧接的 1 higher / 1 comparable / 2 lower 中说明其精确定义；当前已做到，不必改 outcome。 |

## 4. 科学严谨性与 claim–evidence 对照

| 主要主张 | 核对结果 | 允许的结论范围 |
| --- | --- | --- |
| 固定图像时 attention head 可表示为两层网络 | Eq. 1 的 affine 投影、output bias、缩放与 softmax 耦合正确；Appendix A 补全定义 | 已有表示用于分析，不等于新架构；不自动给出 input provenance。 |
| Received attention 与 C3 的定义 | sum over queries/heads、排除 CLS、9–12 层求和与代码一致；C3 为 per-head norm 后相加 | C3 省略 cancellation、residual 和后续计算，属于局部 magnitude score，非完整因果 attribution。 |
| DeiT-S/B 的 C3 对齐收益 | enrichment 0.038/0.065，ClickMe 0.033/0.058 与现有 CSV 一致 | 对 R0 的四个对齐比较提高；对 control 为 3/4，B ClickMe 仅 0.017，按 margin comparable。 |
| C3 deletion 没有达到实用阈值 | 0.017/0.029，均小于 0.05；区间排除零也不改变注册 category | 可说没有达到注册实用 margin，不能说严格无效或等价。 |
| DINO 无 practically meaningful C3 收益 | 最大绝对变化约 0.00351；三 cell 均 comparable | 固定 DINO S/16 与当前层/指标，不代表所有自监督模型。 |
| 深层 C2 下降 | 三模型 × 三指标九 cell 均 lower；S/DINO 也低于 matched blur | 此 probe-relocation 方案在当前深层 setting 失败；不证明空间对应性问题不存在。 |
| 组合收益有限 | 四个注册 supervised alignment cell：1 higher、1 comparable、2 lower | R2 的历史 outcome “partly holds”准确；不能概括为组合普遍有效。 |
| 模型侧参考没有 higher | register enrichment/ClickMe 增益 0.012/0.013；MiTA 两对齐指标 lower | R3 在该对照范围成立；训练/preprocessing 差异阻止“readout 优于模型改造”的一般结论。 |
| 分配权重比 shuffled weights 好 | Appendix B 的四个 supervised comparisons 支持 | 只说明权重分配在这组比较中有用，不证明高范数 token 是收益的因果来源。 |
| 高范数比例与收益相联系 | Spearman 0.045–0.158，原值保留 | 微弱、post-hoc association，未构成通用诊断预测器。 |
| 参数随机化通过 | Table B.3 与原 sanity.csv 一致，mean/probe 固定，五个 reading 通过 | 必要参数敏感性；不等于 prediction faithfulness。 |

主要 causal/registration 边界已经写对。仍应避免把 C2 的未奏效说成 fast-weight 视角验证了
“token 漂移是注意力图错误的原因”，或把 high-norm share 当作经过验证的诊断器。本稿当前没有这样断言。

## 5. Figures、tables、逐页排版与 appendix

### Figures 和 tables 的论证作用

| 对象 | 评价 | 可执行建议 |
| --- | --- | --- |
| Figure 1a | 保留；代表视角，图与准确公式互补 | 示意图省略 bias 可以接受，caption/§3.1 已说明是表示示意；不必重画。 |
| Figure 1b | 有效 teaser，但完全重复 Figure 2f 的 C3 两对齐比较 | 首页保留为首个结果预览，正文明确 Fig. 2 展开全部三个测试；不要当成第二组独立证据。只有继续优化空间时才考虑缩减重复 panel。 |
| Figure 2a–f | 当前最有解释力的主图；按列连接 proxy 与 correction/control，零线、margin、形状编码清楚 | 顶部 proxy 和底部效应不能视觉上暗示因果；现 caption 的 post-hoc 标签正确。caption 可补 a/c 为 ImageNet-S、b 为 2,000 held-out images 的样本来源；很窄的 CI 不可被描述为“确定无误”。 |
| Table 1 | 核心完整数值表；与 Figure 2 作用互补，并包含 deletion、组合与模型侧参考 | 字号和细上标在 100% 阅读偏紧但可用。caption 的 *bold: both comparisons* 改成 *Bold: higher than both R0 and control.* 清楚说明空白 MiTA deletion 表示未定义/未测，原因保留 Appendix A。 |
| Figure B.1 / Table B.1 | 支撑执行与层变化，不与主文争抢空间，放附录合理 | 保留 caption 已说明的 crop-ratio 差异，不将曲线严格当作同一 preprocessing 下对照。 |
| Table B.2 | 配对区间与 verdict 是科学透明度的重要支持，8 pt 正文可读 | 把引出表的完整说明段放在表前或表后，避免 p10 的 *for the* 与 p11 表后的 *test-time register* 被整表隔开。 |
| Table B.3 | 纳入 Additional Results 的位置合理，判据与必要性边界充分 | 表本身无需改，数值保留；单独半页留白可以排版压缩，但不是课程违例。 |
| Table C.1 | 失败机制的候选检查，有 post-hoc 与判据说明 | 四项未达到标准不代表四种机制被普遍排除；现正文已保留该界限。 |
| Figures D.1–D.2 | 固定规则选例，明确 min–max 归一化，不靠挑图支撑平均改进 | D.2 标签偏小，允许放大；没有必要把图搬回四页正文。人类图与方法图是不同可视化，不比较绝对色幅。 |
| Appendix E | 代码与 received/C3/C2 的公式对应，作为结尾合适 | 给出 torch tensor 维度，prefix 可处理特殊 token；保持最后位置。 |

### 逐页检查记录

| 页 | 内容 | 结果 |
| --- | --- | --- |
| 1 | 新标题、113 词摘要、引言、Figure 1 | 无裁切；示意图与效果点清晰。贡献列表整体移到第 2 页，不违反课程规定，但首页的研究问题可以更直接。 |
| 2 | 贡献、Related Work、§3.1–3.2、Eq. 1 | 公式正确，三问题段落完整留在本页；§2.2 引文密度高，文字比较可加强。 |
| 3 | Figure 2、§3.3、实验设置/诊断/首项结果 | 图文顺序合理，无断在半句的方法段；§4.3 连续加粗信息过密。 |
| 4 | Table 1、其余结果、§4.4、讨论、结论 | 四页正文在这里结束；表与结论完整，无页外溢出。 |
| 5–6 | 33 条 references | 完整、有序，无悬空链接/乱码；字段与版本问题另列逐条核查。 |
| 7 | Appendix A，模型、数据、诊断、修正 | 信息充分，局部长模型名导致较松字间距，但无溢出。 |
| 8 | A，校准、指标、GAE、注册 | 复现细节扎实；末尾注册说明跨至 p9，属正常续文。 |
| 9 | Table A.1、计算/代码、B 开始 | 注册结果完整；执行图转到 p10，p9 下方留白可以接受。 |
| 10 | Figure B.1、诊断、Table B.1 | 图表清楚；B.2 引导段在页末被截开，需要排版小修。 |
| 11 | Table B.2、contribution checks | 表格完整；表下先接前页半句再开新段，建议消除这种阅读中断。 |
| 12 | B.1 sanity / Table B.3 | 随机化说明和表完整，较多留白，位置和结论范围合理。 |
| 13 | C，relocation checks / Table C.1 | 一页完整；建议把问句标题改为 checks，避免许诺已回答机制。 |
| 14 | D，六个固定规则例图 | 无裁切，D.2 字号较小，caption 交代选择与归一化。 |
| 15 | E，key code | 代码完整且确实收尾；留白属于可选压缩项，不影响四页限制。 |

全文核验：4 页正文、2 页 references、9 页 appendix；所有字体嵌入；无未解析引用、坏的内部
page destination、文本越出 PDF 页面或 overfull box。LaTeX 的 underfull 提示对应已观察到的
少量留白/字间距，不构成内容丢失。`paper/v12/verify_build.py` 已重新通过。

## 6. 33 条 reference 的严格审查结果

方法：按 PDF 编号逐条比作者、顺序、完整度、题名、年份、venue、已有卷期页、DOI、版本；
有 DOI 优先查注册元数据，再查原论文/官方会刊。另核每条引用在正文所支撑的具体论断。
Crossref 不登记的 DOI 查 DataCite；HTTP 429 和 OpenReview 反爬不被解释为文献不存在。

全部 33 条均定位到原论文或维护者的官方记录，未发现捏造文献或已证实 DOI 指向其他论文的情况。
主要待处理项为 [29] 的作者顺序/版本冲突；[31] 为软件 concept DOI 的元数据演变，
不等于 v12 采用的维护者推荐引用错误。其余不同姓名展开/分页/online-first 年份均需按目标版本判断。
省略可选 DOI、页码或 PMLR 卷号不自动构成学术引用不合规。

| # | Reference | 结论与引用支持范围 | 原始来源 |
| --- | --- | --- | --- |
| 1 | Abnar & Zuidema 2020 | 已核作者、ACL 页码/DOI；rollout/flow 的来源正确。 | [ACL](https://aclanthology.org/2020.acl-main.385/) |
| 2 | Adebayo et al. 2018 | 已核；参数随机化是必要敏感性检查，稿件未将其当充分忠实性证明。 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) |
| 3 | Ayyar et al. 2025 | 已核，arXiv 预印本身份正确；statistical filtering 表述获支持。 | [arXiv](https://arxiv.org/abs/2510.06070) |
| 4 | Bau et al. 2017 | 已核 CVF 版 6541–6549；IEEE 另有分页。应更明确为语义单元测量的方法借鉴。 | [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html) |
| 5 | Blücher et al. 2024 | 已核 TMLR；SRG-style/top-1 adaptation 的署名和限定正确。 | [TMLR 原文](https://openreview.net/pdf?id=bIiLXdtUVM) |
| 6 | Bondarenko et al. 2023 | 已核；no-op 与 outliers 仅作为功能类比，范围合理。 | [NeurIPS](https://papers.neurips.cc/paper_files/paper/2023/hash/edbcb7583fd8921dad78adecfe06a99b-Abstract-Conference.html) |
| 7 | Brunner et al. 2020 | 已核；支持语言 token identity，ViT spatial information 应单独归属 [23]。 | [ICLR 原文](https://openreview.net/pdf?id=BJg1f6EFDB) |
| 8 | Caron et al. 2021 | 已核 CVF 版 9650–9660；IEEE 分页不同。DINO 与 attention-map 语义现象引用恰当。 | [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html) |
| 9 | Chefer et al. 2021a | 已核 CVPR、782–791；relevance propagation 的来源正确。 | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html) |
| 10 | Chefer et al. 2021b | 已核 ICCV、397–406；GAE attention×gradient 与跨层聚合来源正确，未与 [9] 混淆。 | [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Chefer_Generic_Attention-Model_Explainability_for_Interpreting_Bi-Modal_and_Encoder-Decoder_Transformers_ICCV_2021_paper.html) |
| 11 | Darcet et al. 2024 | 已核；高范数背景 token 与训练式 registers 归属正确。 | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0b408293619f725fd30162af057e531a-Abstract-Conference.html) |
| 12 | Deng et al. 2009 | 已核作者、CVPR、248–255、DOI；ImageNet 数据来源正确。 | [ImageNet 原文](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf) |
| 13 | Dosovitskiy et al. 2021 | 已核 ICLR、12 名作者顺序；ViT 背景引用正确。 | [ICLR 原文](https://openreview.net/pdf?id=YicbFdNTTy) |
| 14 | Fel et al. 2022 | 已核正式 PDF 的 Ivan Felipe；网页显示名较长，不能据此判作者错误。 | [NeurIPS 原文](https://proceedings.neurips.cc/paper_files/paper/2022/file/3d681cc4487b97c08e5aa67224dd74f2-Paper-Conference.pdf) |
| 15 | Ferrando et al. 2022 | 已核 EMNLP、8698–8714、DOI；ALTI 组件及跨层混合表述正确。 | [ACL](https://aclanthology.org/2022.emnlp-main.595/) |
| 16 | Gao et al. 2023 | 已核 TPAMI 45(6):7457–7476；DOI 中 2022 不使卷年 2023 错误。 | [DOI](https://doi.org/10.1109/TPAMI.2022.3218275) |
| 17 | Jain & Wallace 2019 | 已核；attention 解释性争论引用恰当。可选补 3543–3556、DOI。 | [ACL](https://aclanthology.org/N19-1357/) |
| 18 | Jiang et al. 2025 | 已核 PDF 的 Nick Jiang 与其他作者；网页 Nicholas 为姓名展开；test-time register 来源正确。 | [NeurIPS 原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/51bae6441380c4c4cab87f430be89ac2-Paper-Conference.pdf) |
| 19 | Kobayashi et al. 2020 | 已核 EMNLP、7057–7075、DOI；norm component 来源恰当，稿件已区别完整方法。 | [ACL](https://aclanthology.org/2020.emnlp-main.574/) |
| 20 | Kuang et al. 2026 | 已核正式 ICML/PMLR 306:61236–61266；建议明示所测 ViT 中对应偏置很弱的相反证据。 | [PMLR](https://proceedings.mlr.press/v306/kuang26a.html) |
| 21 | Linsley et al. 2019 | 已核 ICLR/原稿作者与年份；ClickMe 数据来源正确。 | [原稿](https://arxiv.org/abs/1805.08819) |
| 22 | Modarressi et al. 2022 | 已核 ACL 官方页、258–271、DOI；Crossref 本轮 429，以发布方核实。 | [ACL](https://aclanthology.org/2022.naacl-main.19/) |
| 23 | Raghu et al. 2021 | 已核；支持 ViT 空间信息保留，不证明内容 provenance。 | [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/652cf38361a209088302ba2b8b7f51e0-Abstract.html) |
| 24 | Ramsauer et al. 2021 | 已核 ICLR 的 13 名作者顺序与当前条目一致；旧 arXiv 作者列表不同不使当前会议版错误。 | [ICLR](https://iclr.cc/virtual/2021/poster/2657) |
| 25 | Schlag et al. 2021 | 已核；linear attention 的 fast-weight 解释来源正确，区别于 softmax 表示。 | [PMLR](https://proceedings.mlr.press/v139/schlag21a.html) |
| 26 | Schmidhuber 1992 | 已核；fast-weight memory 基础归属正确。 | [MIT Press](https://direct.mit.edu/neco/article-abstract/4/1/131/5620/Learning-to-Control-Fast-Weight-Memories-An?redirectedFrom=fulltext) |
| 27 | Touvron et al. 2021 | 已核 ICML、10347–10357、作者顺序；DeiT 来源正确。 | [PMLR](https://proceedings.mlr.press/v139/touvron21a.html) |
| 28 | Walmer et al. 2023 | 已核 CVPR、7486–7496、作者顺序；supervision 改变 ViT 表征/图的表述正确。 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Walmer_Teaching_Matters_Investigating_the_Role_of_Supervision_in_Vision_Transformers_CVPR_2023_paper.html) |
| 29 | Wen et al. 2026, MiTA | 来源存在、fast-weight 论断获原文支持；元数据/推荐引用与新版 PDF 的 Meng/He 顺序冲突待核。 | [arXiv](https://arxiv.org/abs/2602.01219), [作者仓库](https://github.com/QishuaiWen/MiTA) |
| 30 | Wiegreffe & Pinter 2019 | 已核 EMNLP、11–20、DOI；争论的另一立场引用恰当。 | [ACL](https://aclanthology.org/D19-1002/) |
| 31 | Wightman 2019, timm | 维护者推荐的项目引用可追溯；concept DOI 当前 release/contributor 元数据不同，需区别项目级引用与特定 release 引用。Crossref 404 不代表 DOI 无效，DataCite 可解析。 | [官方仓库](https://github.com/huggingface/pytorch-image-models), [DataCite](https://api.datacite.org/dois/10.5281/zenodo.4414861) |
| 32 | Zhou et al. 2016 | 已核 CVPR、2921–2929；CAM 输出加权是类比，不是 token attribution 的证明。 | [CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html) |
| 33 | Zhu et al. 2024 | 已核 ECAI、451–458、DOI；支持 norm-weighted 成分，不等同于本稿 C3 整体算法。 | [DOI](https://doi.org/10.3233/FAIA240520) |

逐字段证据、版本差异、检索限制和可选补全详见三个独立报告：
[1–11](v12_refs_01_11.md)、[12–22](v12_refs_12_22.md)、[23–33](v12_refs_23_33.md)。
以上没有自动改动任何 BibTeX 字段。

## 7. 建议的最终修订包及验收

优先做：修 R1 范围；补引言研究问题/三问桥梁；给摘要与结论加 deep-layer received-attention
范围；对 [7]/[23] 与 [20] 做更精确的 critical synthesis；改善 §4.3 和 Table B.2 的阅读中断；
保留 [29] 的待核状态。可选做：附录 C/B 标题、MiTA 有效交集、图注样本来源和统一可选书目字段。
均可用现有结果完成，保留 4 页正文与现有两张主图，不需要训练、补数据或改注册。

再次导出时检查：正文结束于第 4 页；references 和 A–E 完整；86 个主表测量、7 个 diamond、
3 个 bold 与所有 paired verdict 不变；registered vs post-hoc 标签保留；E 仍为最后内容；
明确所引用版本而非混搭不同页码、作者和年份；逐页重渲染后复核布局。
