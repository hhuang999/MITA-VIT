# v13 终稿审阅

**总体判断**　按 DSAA 2012 的 workshop-level 标准，这篇论文的贡献是用同一套受控比较检验 ViT 的三种读图假设。最有说服力的证据来自同一批图、同一层、同一组空间排名在两个评价目标上的相反变化。对末层 CLS 路径的解释已有直接实验依据，对原始像素来源的解释仍是开放问题。

## 四页的论证顺序

1. 第 1 页从背景上的亮点出发，解释“一个 token 收到多少注意力”到“输入的哪些位置重要”之间需要哪些假设。首页图保留 fast-weight 示意，将三个问题放在图旁，贡献列表交代框架、检验方式和发现。
2. 第 2 页把既有方法按它们追踪的对象和层次组织起来。式 (1) 区分 activation 和 projected value；三段方法各自给出代理指标、读图改动及其对照。三角不等式解释按头取范数与相加后取范数的区别。
3. 第 3 页用原 Figure 2 并列呈现三个诊断和两类比较。数据、指标、阈值、主要比较和例外的注册负对照均在正文。对诊断与收益关系的事后解读单独标出。
4. 第 4 页递进地问“长度放在哪里有关系吗”和“换一个接收 query 后，哪个目标受益”。两张表分别使用 383 张有效掩膜和 128 张配对图。对象面积和替代删除指标保留在附录 E。

## 主张与证据

| 主张 | 对照或直接证据 | 审阅判断 |
| --- | --- | --- |
| 三个假设可以分别检验 | 同一 checkpoint 上比较 C1 与等幅径向先验、C2 与位移匹配模糊、C3 与高范数抑制 | 支持一个分析框架。三个代理量并非潜在机制的独立因果测量。 |
| 定位 token 索引不能直接用来搬移图上的证据 | 尽管位置仍可解码，深层 C2 与 R0 的 9 项比较均下降；与模糊对照相比，监督 DeiT-S 和 DINO 的三项均下降 | 支持对这条重定位规则的负面结论。没有对原始像素来源标注真值。 |
| Value weighting 改善两种监督 DeiT 的空间读出 | 与 R0 相比，4 项对齐指标均提高；与高范数对照相比，3 项通过门槛。DeiT-B ClickMe 的 0.017 低于 0.02 | 限定为所测 checkpoint 和指标。删除实验的两项增益均低于 0.05。 |
| 排列会影响 value weighting 的收益 | Oxford-IIIT Pet 383 张图中，换用 donor 排序或在同半径内打乱，enrichment 下降 0.022 和 0.021；donor-pair bootstrap 保持方向 | 探索性。保留每头的长度分布，对照支持位置分配的作用。未投影的长度也能得到相近或更高的分数。 |
| Query 的选择取决于解释目标 | 同一 128 张图、同一层和同一对 readouts，CLS-only 相对全部 queries 的 margin Spearman、enrichment、AUROC 配对差值为 +0.041、−0.033、−0.091 | 对规定的干预目标及图像成立。干预直接改变末层 CLS 所接收的值；表中的 Spearman 衡量排序一致性。同图掩膜检验是在查看干预结果后追加的。 |

## 按课程五项标准复核

- **Contribution**　问题、方法与发现各有明确位置。fast-weight 表示、按 value norm 加权和 value zeroing 都有前人工作。本文的原创部分落在三个读图假设的组织、匹配对照和同图双目标比较。
- **Clarity**　摘要和方法均使用 image specificity、spatial correspondence、output contribution。读者能从正文区分原始模型的计算、对图的后处理和对 CLS 路径的实验干预。Table 2 新增带符号的配对差值，避免从四舍五入后的均值手算。
- **Depth**　原注册三项结论分别为 holds、partly holds、holds。R1 的深层 DINO 与浅层全模型对照覆盖 9 格；R2 只有 1 格 higher。个别读图法与 R0 的描述性差异、注册主比较和新增探索分别说明。CLS 的直接干预针对该 query，这对齐了测量目标，也限制其外推；另一 checkpoint 上能否复现双目标结果仍待检验。
- **Literature review**　36 条均有文内引用。沿用已有逐条核查记录，本轮另复读三项直接关系到贡献归属的原文。MiTA 支持输入条件化的两层表示，Kobayashi 等支持 transformed-vector norm 的分析，Mohebbi 等的 Value Zeroing 用于测量上下文混合。MiTA 的版本署名顺序疑点仍按原记录保留。[MiTA](https://arxiv.org/abs/2602.01219) · [Norm-based attention](https://aclanthology.org/2020.emnlp-main.574/) · [Value Zeroing](https://aclanthology.org/2023.eacl-main.245/)
- **Formatting**　逐页检查了渲染后的 18 页。第 1–4 页为正文（含 2 图 2 表），第 5–6 页是 references，第 7–18 页是附录 A–F，最后一段代码演示 CLS 干预。36 条文献进入 bibliography，83 个内部链接目标通过检查，字体已嵌入，编译无超宽盒、未定义引用或重复标签。原 v12 PDF 与注册文件的哈希校验通过。自动核验结果见 [verification.json](verification.json)。

就证据而言，摘要和结论的力度适合课程论文。最有价值的后续检验是，在另一个独立 checkpoint 上事先固定查询集合、分类干预和掩膜指标，复核两类读出目标的分歧。
