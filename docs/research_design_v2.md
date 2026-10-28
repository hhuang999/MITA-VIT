# 研究设计书 v2：用 MiTA 的压缩与路由解剖 ViT 的注意力

Oct 28, 2026 · @deng

## 1. 研究问题与主张

**一句话：**借 MiTA 的"压缩"和"路由"两个零件，不训练、不改权重，逐层测量一个普通 ViT 的注意力有多少可以用摘要代替、多少必须读具体 token、多少是每个 query 独有的。

**问题：**一个训练好的 ViT，它每一层的注意力能不能拆成三部分：(a) 可压缩的部分——用 m 份区域摘要就能代替；(b) 可路由的部分——同一区域的 query 读同一组 k 个具体 token 就够；(c) query 独有的部分——每个 query 必须各读各的？这三部分各占多少，随层怎么变？

*Can each attention layer of a trained ViT be decomposed into a compressible part (m region summaries suffice), a routable part (queries in a region share one set of k tokens), and a query-specific remainder? How large is each part, and how does the split change with depth?*

**主张：**MiTA 把高效注意力统一成"压缩 + 路由"两种缩放快权重的方式。我们把这两种方式当成测量工具：从原模型读出摘要和共享读取集合，强制模型只按它们计算，看预测是否改变。用这把尺子，我们给 DeiT-S 的 12 层各画一张"注意力解剖图"，并用 MiTA 自己训练的模型作对照，看"深层聚焦物体"是注意力的通性还是 MiTA 训练的产物。

*MiTA unifies efficient attention as two ways of scaling fast weights: compression and routing. We turn both into measurement tools: we read summaries and shared top-k sets from the original model, force it to compute only through them, and test whether predictions survive. With this ruler we draw a layer-by-layer anatomy of DeiT-S attention, and use MiTA's own trained model as a control to ask whether the object focus seen in MiTA is a property of attention or of MiTA training.*

主张只承诺"提出工具并测量"，不承诺某一部分一定大。三部分无论怎么分，都能写成结论（第 9 节）。

**和 v1 的区别：**v1 问"很短的说明书够不够忠实"，答案已经出来了：不够（k = 128、读 65% 的 token 才到 90% 一致率）。v2 不再问"够不够"，改问"丢掉的部分是什么性质"。v1 的结果全部保留，成为 v2 的 L3 这一条线。

## 2. 研究故事

**第 1 步：背景——ViT 的注意力读不懂。**DeiT-S 把 224×224 的图切成 196 个 patch token，加 1 个 CLS，共 197 个。每层每个 head 都算一个 197×197 的矩阵 A = softmax(QKᵀ/√d)，12 层、6 个 head，一张图约 280 万个数。没有人能直接读懂。

**第 2 步：热图不够用。**现有方法把这些数压成一张"模型在看哪里"的热图。热图丢了"谁在读谁"的结构，也没法检验它是否忠实。我们在 v1 里试过一种有结构、可检验的替代品：每块区域读 k 个 token。结果是，要保住 90% 的预测，k 要到 128。说明书必须很长。为什么？丢掉的那部分到底是什么？

**第 3 步：MiTA 给了两个零件。**注意力可以看成一个临时的两层 MLP，每个 token 是一个隐藏神经元，宽度 197。MiTA 指出，所有高效注意力都在用两种办法缩小这个 MLP：

- **压缩**：把 197 个神经元合并成 m 个"摘要神经元"（landmark），每个 query 只读这 m 个；
- **路由**：把 query 分成 m 组，每组共享 k 个具体神经元。

MiTA 把两者合在一个 softmax 里，目的是加速。我们的观察是：这两个零件正好对应注意力里两种不同性质的信息——"大家都需要的概要"和"这一组需要的具体内容"。剩下的第三种是"每个 query 自己才需要的"，谁也代替不了。

**第 4 步：我们的做法——一次换一个零件，看预测变不变。**在原模型的前向里读出摘要和共享集合（不改模型），然后强制模型只用摘要、只用共享集合、两者都用、或者每个 query 自选 k 个，分别测预测一致率。四个数字一减，就得到这一层的三个成分。

举个例子。假设某一层里：只用摘要能保住 60% 的预测，只用共享集合 70%，两者都用 85%，每个 query 自选 95%。那么这一层大约有 10 个百分点是"必须各读各的"，其余可以用 m 份摘要加 m 组共享 token 概括。换一层，比例可能完全不同。

**第 5 步：预期发现。**三张图：每层的解剖图（RQ1）；说明书长度对忠实性的前沿（RQ2）；同一套工具用在 DeiT-S 和 MiTA-DeiT-S 上的对照（RQ3）。具体结论等实验完成后填，运行前的预测见第 8 节。

**英文版（写论文引言用）**

1. Attention maps are widely used to explain where a ViT looks, yet one image yields about 2.8 million attention values in DeiT-S, and a heatmap neither preserves who-reads-whom structure nor admits a direct faithfulness test.
2. Viewing attention as a two-layer fast-weight MLP with one neuron per token, MiTA shows that efficient attention shrinks this MLP in two ways: compression (m landmark summaries) and routing (m groups sharing k neurons).
3. We turn both into training-free probes on a pretrained ViT: we read summaries and shared top-k sets from the original forward pass, force the model to compute only through them, and measure whether its predictions survive.
4. Subtracting these measurements gives, for every layer, how much of its attention is compressible, how much is routable, and how much is query-specific.
5. With this ruler we draw a layer-wise anatomy of DeiT-S, map the trade-off between summary length and faithfulness, and show with MiTA's own trained model whether the object focus of top-k sets is learned under MiTA or intrinsic to attention.

## 3. 相关工作与空白

和我们相关的工作分五组。前三组提供零件，后两组提供检验方法和对照。我们的新意在于把零件当尺子，并用执行检验它们。

| 组 | 文献 | 和我们的关系 | 边界 |
| --- | --- | --- | --- |
| 压缩（摘要通道） | [Nyströmformer](https://arxiv.org/abs/2102.03902)（Xiong 等，AAAI 2021） | 最早用"相邻 query 的平均"当 landmark。我们的 landmark 定义和它一样，只是换成 2D 区域 | 它用 landmark 做矩阵近似；我们用 landmark 做可读的区域，并用执行检验 |
| 压缩（摘要通道） | [Agent Attention](https://arxiv.org/abs/2312.08874)（Han 等，ECCV 2024） | 就是 MiTA 的压缩通道：agent 从 query 池化而来，先汇总 K/V 再分发。论文报告它可以不训练直接插进 Stable Diffusion | 这是"摘要通道不训练也能用"的先例，支持 L2；我们用它测量而不是加速 |
| 路由（共享 top-k） | [Top-k Attention](https://arxiv.org/abs/2106.06899)（Gupta 等，SustaiNLP 2021） | 每个 query 只留分数最高的 k 个 key。他们的零训练实验在 UnifiedQA 的 FFN 层上 | 预训练 ViT 注意力上每 query top-k 够不够，仍是我们要测的问题（L1）；引用时不写"已证明注意力上可行" |
| 路由（共享 top-k） | [Clustered Attention](https://arxiv.org/abs/2007.04825)（Vyas 等，NeurIPS 2020）；MoBA、NSA（2025） | 一组 query 共享一组 key。MiTA 代码直接改自 MoBA | 它们按特征或按块分组，目的是加速；我们按图像区域分组，目的是可读 |
| 两者合一 | [MiTA](https://arxiv.org/abs/2602.01219)（Wen 等，NeurIPS 2026）；[代码](https://github.com/QishuaiWen/MiTA) | 快权重视角；压缩 + 路由合成一个 softmax；官方默认 m = 25、k = 25；公开 MiTA-DeiT-S 权重 | 它是新结构，需要训练；我们不训练，用它的分解去读一个普通 ViT，并用它的模型做对照 |
| 汇聚点 / 高范数 token | [Massive Activations](https://arxiv.org/abs/2402.17762)（Sun 等，COLM 2024）；[Registers](https://arxiv.org/abs/2309.16588)（Darcet 等，ICLR 2024）；[Test-time Registers](https://arxiv.org/abs/2506.08010)（Jiang 等，NeurIPS 2025） | 少数 token 范数极大、吸引注意力，其值几乎不随输入变，"像一个偏置项"。Jiang 等不训练就能把它们搬到新 token 上 | 我们的选做实验检验"摘要通道的工作就是送这个偏置"；对照改用 Jiang 等的方法（同一套权重），比 v1 的 DINOv2 有/无寄存器干净 |
| "删掉注意力也没事" | [How Much Does Attention Actually Attend?](https://arxiv.org/abs/2211.03495)（Hassid 等，2022）；[Sixteen Heads](https://arxiv.org/abs/1905.10650)（Michel 等，NeurIPS 2019）；[Not All Attention is Needed](https://arxiv.org/abs/2406.15786)（He 等，2024） | 预训练模型里很多 head 可以换成常数矩阵或直接删掉，性能几乎不变 | 解释为什么单层替换一致率普遍很高；它们是我们的下限对照（L2 且 m = 1 即 Hassid 的常数注意力；删整层即 Michel） |
| 忠实性检验 | [ERASER](https://arxiv.org/abs/1911.03429)（DeYoung 等，2020）；[Attention Knockout](https://arxiv.org/abs/2304.14767)（Geva 等，2023）；[Jain & Wallace 2019](https://arxiv.org/abs/1902.10186)；[Wiegreffe & Pinter 2019](https://arxiv.org/abs/1908.04626) | "只保留解释选出的部分，看预测是否保持"；注意力能否当解释的争论 | 我们的一致率就是 ERASER 的充分性；覆盖率对一致率的对比（第 6 节）直接回应这场争论 |
| ViT 注意力的描述性分析 | [Do ViTs See Like CNNs?](https://arxiv.org/abs/2108.08810)（Raghu 等，NeurIPS 2021）；[Intriguing Properties of ViTs](https://arxiv.org/abs/2105.10497)（Naseer 等，NeurIPS 2021）；[Rollout](https://arxiv.org/abs/2005.00928)；[Chefer 等 2021](https://arxiv.org/abs/2012.09838) | 平均注意力距离区分局部/全局 head；ViT 对丢 patch 稳健；热图方法 | 我们的"神经元并集"和"可共享指数"在距离指标之外加了一个可执行检验的维度；我们不和热图比"像不像物体" |

**快权重视角的来源：**注意力等价于快权重 MLP 是已有的代数事实（[Schlag 等 2021](https://arxiv.org/abs/2102.11174)，Schmidhuber 1992），MiTA 用它统一了高效注意力。我们不主张这一点是新的。

**三条边界（写进 Related Work 结尾）：**

1. 我们不提出新的高效注意力，也不做 token 剪枝；Nyströmformer、Agent Attention、MiTA 的机制在这里只是测量工具。
2. 我们不是另一种热图方法，不和 Grad-CAM、rollout 比"像不像物体"；我们测的是"简化后的计算是否保持预测"。
3. 我们不解释 MiTA 本身；MiTA-DeiT-S 只用作"同结构、不同训练"的对照。

**尚未核实、暂不引用的说法：**"标准注意力模型推理时直接换成 MiTA 能保留 95% 精度"。MiTA 仓库 README 的表全是从头训练的结果；拿到论文 PDF 核对前不写。若原文确有此说法，RQ2 的 L4（m = 25、k = 25）就是对它的直接复现。

## 4. 必备基础知识

**4.1 回顾：注意力是一个临时的两层 MLP。**在一个 head 里，query q 和每个 token 的 key 做点积，过 exp 再归一化，用 value 加权求和。把 key 看成神经元的输入权重、value 看成输出权重，这就是一个隐藏宽度为 N = 197 的两层 MLP。权重由当前图片临时算出，所以叫"快权重"。完整推导和手算例子见 v1 设计书 §4。

**4.2 MiTA 的两条通道。**设有 m 个 landmark q̃₁…q̃ₘ（区域内 query 的平均）。每个 landmark 产生两样东西：

- **摘要**（压缩通道）：用 landmark 对全部 token 做一次标准注意力，得到一个向量。它是"这块区域平均来说读到了什么"：

```latex
\tilde v_r = \sum_j \text{softmax}_j\!\left(\tilde q_r \cdot k_j / \sqrt d\right) v_j
```

- **读取集合**（路由通道）：landmark 打分最高的 k 个 token，S\_r = TopK\_j(q̃\_r · k\_j, k)。

每个 query 被分配给一个 landmark e(i)，然后在两类位置上做**一次**softmax：m 个摘要槽（key 是 landmark，value 是摘要）和 k 个具体槽（key、value 就是那些 token 自己的）：

```latex
\text{out}_i = \frac{\sum_{r=1}^{m} e^{\,q_i\cdot \tilde q_r / d}\,\tilde v_r \;+\; \sum_{j \in S_{e(i)}} e^{\,q_i\cdot k_j/\sqrt d}\, v_j}{\sum_{r=1}^{m} e^{\,q_i\cdot \tilde q_r / d} \;+\; \sum_{j \in S_{e(i)}} e^{\,q_i\cdot k_j/\sqrt d}}
```

注意摘要槽的指数里是 d 不是 √d。这不是笔误：MiTA 官方代码把 landmark 先乘了一次 1/√d 再送进带缩放的 flash-attn，注释写的是"smooth the attention weights on router"。摘要槽因此被故意调淡。复现完整 MiTA 时必须照做；第 5 节有一个缩放消融。

**4.3 手算一个例子。**沿用 v1 §5.2 的 4 个 token、两块区域 A = {1, 2}、B = {3, 4}、k = 2，省略所有缩放。

- queries：q₁ = (3, 1)，q₂ = (−1, 1)，q₃ = q₄ = (1, −1)
- keys：k₁ = (1, 0)，k₂ = (0, 2)，k₃ = (2, 2)，k₄ = (0, −1)
- values：v₁ = (1, 0)，v₂ = (0, 1)，v₃ = (1, 1)，v₄ = (0, −1)

landmark：A = (1, 1)，B = (1, −1)。A 的打分 (1, 2, 4, −1)，S\_A = {3, 2}；B 的打分 (1, −2, 0, 1)，S\_B = {1, 4}。

摘要：ṽ\_A = softmax(1, 2, 4, −1)·V ≈ (0.042, 0.114, 0.839, 0.006)·V = (0.881, 0.947)；ṽ\_B = softmax(1, −2, 0, 1)·V ≈ (0.414, 0.021, 0.152, 0.414)·V = (0.566, −0.241)。

q₁ 分给 A（q₁·A = 4 > q₁·B = 2）。四种计算方式下 q₁ 的输出：

| 计算方式 | q₁ 能读的槽和 logits | 输出 |
| --- | --- | --- |
| L0 全注意力 | k₁…k₄：(3, 2, 8, −1) | (0.997, 0.993) |
| L2 只摘要 | A, B：(4, 2) | (0.843, 0.805) |
| L3 只共享集合 | k₃, k₂：(8, 2) | (0.998, 1.000) |
| L4 完整 MiTA | k₃, k₂, A, B：(8, 2, 4, 2) | (0.994, 0.996) |

对 q₁ 来说，共享集合就够了，摘要单独用会差很多。再看 q₂ = (−1, 1)，也分给 A：

| 计算方式 | logits | 输出 |
| --- | --- | --- |
| L0 全注意力 | (−1, 2, 0, −1) | (0.150, 0.880) |
| L2 只摘要 | (0, −2) | (0.843, 0.805) |
| L3 只共享集合 | (0, 2) | (0.119, 1.000) |
| L4 完整 MiTA | (0, 2, 0, −2) | (0.206, 0.977) |

q₂ 真正想读的是 k₂，共享集合里正好有，所以 L3 和 L4 都还行；只摘要的 L2 给 q₁ 和 q₂ 几乎相同的输出，因为摘要把两者的差别抹平了。这就是"可压缩"和"可路由"的区别：压缩保留的是区域的共性，路由保留的是区域内各 query 对具体 token 的不同权重。

**4.4 这个视角对我们有什么用。**v1 只测了 L3。它在 k = 16 时丢掉 30% 的预测，但分不清是因为缺摘要（这层有很多"只要概要"的 query），还是因为被迫共享（这层有很多"各读各的"query）。把 L1、L2、L4 都测了，四个数相减，就能分清。

## 5. 方法：读取、干预阶梯、四个因子

方法仍分两步：**读取**从原模型里读出摘要和读取集合，不改模型；**执行**强制模型只按某一种简化方式计算。下面的计算都在某一层、某一个 head 里进行。

**5.1 读取（沿用 v1，增加摘要和 CLS 集合）**

1. 划分区域：把 14×14 的 patch 网格用 adaptive average pooling 分成 m 块。m = 16 时相邻窗口有重叠（单轴窗口 \[0,4)、\[3,7)、\[7,11)、\[10,14)）；m ∈ {1, 4, 49, 196} 时是不重叠的正方形。CLS 不参与池化。
2. landmark：q̃\_r = 区域内 patch query 的平均。
3. 读取集合：S\_r = TopK\_j(q̃\_r·k\_j/√d, k)，j 遍历全部 197 个 token（CLS 可被选中）。
4. 摘要：ṽ\_r = Σ\_j softmax\_j(q̃\_r·k\_j/√d) v\_j。
5. CLS 集合（新）：S\_CLS = TopK\_j(q\_CLS·k\_j/√d, k)。CLS 是最后层唯一影响预测的 query，不再被迫借用某块区域的集合。说明书因此有 (m + 1)×k 个编号。
6. 分配：每个 patch query 被分给一个 landmark e(i)，两种方式见 5.3。

读取发生在原模型的一次完整前向里，所有层的摘要、集合、分配都冻结（v1 的 N01）。执行时的 Q、K、V 仍由受干预的前向计算。V03 已经证明现场重算更差（k = 8 时低 13 pp），v2 不再做现场版本。

**5.2 干预阶梯：每个 query 能读什么**

每一级都是在 softmax 前限制"能读的槽"。打分仍用 query 自己的 q\_i，模型权重一点不变。

| 编号 | query i 能读的槽 | 回答什么 | 对应文献 / MiTA 代码 |
| --- | --- | --- | --- |
| L0 | 全部 197 个 token | 恒等检验（必须 100%） | — |
| **L1 每 query top-k** | 自己打分最高的 k 个 token：TopK\_j(q\_i·k\_j, k) | 稀疏的代价；共享的上限 | Gupta 2021（注意力版） |
| **L2 只摘要** | m 个摘要槽 | 压缩够不够 | Agent Attention；MiTA 的 shared expert |
| **L3 只共享集合** | S\_e(i)（CLS 读 S\_CLS） | 路由够不够 | `MiTA_Attention_Route`；v1 的严格版 |
| **L4 完整 MiTA** | S\_e(i) ∪ m 个摘要槽 | 两者加起来够不够 | `MiTA_Attention` |
| rand | 随机 k 个 token（分配不变） | 下限：同样长度、没有信息 | v1 已有 |
| rand+C | 随机 k 个 token ∪ m 个摘要槽 | Shapley 分解需要的第四个角 | 新 |
| const | L2 且 m = 1：所有 query 读同一份全局摘要 | "常数注意力"对照 | Hassid 2022 |
| drop | 这一层注意力分支输出置零 | 最硬的下限 | Michel 2019 |

严格版（L3）的公式和 v1 §5.3 相同。L4 的公式见 §4.2。L1 只是把 S\_e(i) 换成 query 自己的 top-k。

**嵌套关系：**L3 ⊂ L4，L2 ⊂ L4（槽的包含关系）；L1 随 k 增大单调逼近 L0。L2 和 L4 不包含在 L0 里（摘要槽是新加的位置），所以 k = 197 时 L4 不等于原模型；它的正确性靠官方模块对照验证（§11）。

**5.3 四个因子（各做一次消融，主设置加粗）**

| 因子 | 选项 | 为什么重要 |
| --- | --- | --- |
| 路由方式 | **按区域**：e(i) = patch i 所在的区域；点积：e(i) = argmax\_r q\_i·q̃\_r（MiTA 原样，v1 用法） | 按区域的说明书才是真正"区域 r 读 S\_r"；点积路由下第 1 层最大的组有 189 个 query、第 12 层 13.6% 的组是空的。两者差值 = 可读性付出的忠实性代价。m = 196 时按区域路由等于 L1（用作一致性检查）；点积路由则不保证 |
| CLS | **自己的 top-k 集合**；路由到某个 landmark（v1 用法） | v1 第 12 层 k = 8 时 KL = 0.73，是其他层的 5 倍以上；懷疑是 CLS 被迫借用区域集合。消融直接回答 |
| 打分方式 | **query 平均**：s\_{r,j} = q̃\_r·k\_j；注意力平均：s'\_{r,j} = 区域内各 query 对 j 的原始注意力概率的平均 | query 平均等于 logit 平均，容易被范数大的 key（汇聚点）支配；V01 里去掉背景 token 一致率反而从 78% 升到 83%，说明 query 平均选进了有害 token |
| 摘要槽缩放 | **q·q̃/d**（MiTA 代码原样）；q·q̃/√d（标准） | 主设置跟 MiTA，才能说 L4 是 MiTA 的复现；标准缩放告诉我们"调淡"是否关键 |

消融只在主预算（m = 16、k = 16）、全层替换上做，每次只换一个因子。

**5.4 两种替换范围（与 v1 相同）**

- 所有层同时替换：用于预算前沿（RQ2）和消融。
- 每次只替换一层：用于解剖图（RQ1）。单层替换没有前层干预，冻结和现场等价。

**5.5 工程实现（N = 197，全部用密集矩阵，不用 flash-attn）**

摘要槽不能只靠 attn\_mask 注入，需要在上下文管理器里替换 `block.attn.forward`：

```python
# q, k, v: [B, H, N, d]; lm: [B, H, m, d]; summary: [B, H, m, d]; scale = d ** -0.5
k_ext = torch.cat([k, lm * scale], dim=2)          # (q*scale)·(lm*scale) = q·lm / d，即 MiTA 的调淡
v_ext = torch.cat([v, summary], dim=2)
logits = (q * scale) @ k_ext.transpose(-1, -2)      # [B, H, N, N + m]
out = logits.masked_fill(~allowed, -inf).softmax(-1) @ v_ext
```

`allowed` 的形状是 \[B, H, N, N + m\]：L1 只开 query 自己的 top-k；L2 只开后 m 个槽；L3 只开 S\_e(i)；L4 开 S\_e(i) 和后 m 个槽。标准缩放消融把 `lm * scale` 换成 `lm`。读取仍用 v1 的 QKV hook；执行不再用 pre-hook 注入 mask，改为整体替换 forward，退出上下文时恢复。

## 6. 指标与统计

所有指标都先按图片计算，再对图片求均值和区间。图片是唯一的统计单位；层、head、区域、随机种子都不当独立样本。

**6.1 忠实性（主指标）**

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| 预测一致率 Agree | 干预后 top-1 与原模型 top-1 相同的图片比例 | 主指标，所有图的纵轴 |
| KL | KL(p\_原 ‖ p\_干预) | top-1 相同但分布变了的情况（v1 第 12 层） |
| 保留精度 | 干预后准确率 ÷ 原模型准确率（79.62%） | 和 MiTA 论文的"保留 x% 精度"直接对话 |

为什么是一致率不是准确率：我们解释的是模型，原来分错的图应该同样地分错。

**6.2 噪声地板（新）。**读者看到 95% 不知道算高还是低。我们给一条校准线：把中心裁剪窗口向右移 1 像素（在 248 像素的缩放图上裁，没有填充伪影），原模型自己的预测变了多少。这个数画成虚线。如果 1 像素就改掉 4% 的预测，96% 一致率就是"和平移一个像素一样无害"。只需 5,000 次前向。

**6.3 覆盖率（新，不需要干预）。**query i 在原模型里的注意力概率，落在允许集合里的那部分之和：

```latex
c_i = \sum_{j \in \text{allowed}(i)} A_{ij}
```

层覆盖率 = 对 patch query 和 head 求平均；CLS 单列。L4 的摘要槽没有原始概率，覆盖率只算 token 槽。覆盖率是"看图"式的指标，一致率是"检验"式的指标；两者在哪些层分道扬镳，本身就是发现：

| 覆盖率 | 一致率 | 含义 |
| --- | --- | --- |
| 高 | 高 | 集合选对了，模型也真的靠它们 |
| 高 | 低 | 剩下的一点质量很关键（典型：汇聚点或 CLS 被漏掉） |
| 低 | 高 | 这层注意力对预测不重要，残差兑底 |

**6.4 神经元并集（新，不需要干预）。**一层一个 head 里，所有读取集合的并集大小 |∪\_r S\_r ∪ S\_CLS|，除以 197。它是"这个临时 MLP 实际用了多少个神经元"，也是 MiTA token pruning 图的数值版。

**6.5 Shapley 分解（解剖图的定义）。**两个零件：压缩 C、路由 R。价值函数 v(·) = 一致率：v(∅) = rand，v({R}) = L3，v({C}) = rand+C，v({C, R}) = L4。

```latex
\phi_C = \tfrac12\big[(\text{rand+C} - \text{rand}) + (\text{L4} - \text{L3})\big],\qquad
\phi_R = \tfrac12\big[(\text{L3} - \text{rand}) + (\text{L4} - \text{rand+C})\big]
```

两者之和恰好是 L4 − rand，所以堆叠条图有唯一定义，不依赖"先加哪个"。每张图先算出自己的 φ\_C、φ\_R（四个干预的 0/1 一致性相减），再对图片 bootstrap。解剖图每层一根条：底部 rand，其上 φ\_R、φ\_C，到 L4；L1 用一个标记点标在条旁；顶部到 100% 的空白是"这个预算下谁也救不回的部分"。

**6.6 可共享指数（跨层、跨模型可比）。**

```latex
\text{SI} = \frac{\text{Agree}(\text{L4}, m, k) - \text{Agree}(\text{rand}, k)}{\text{Agree}(\text{L1}, k + m) - \text{Agree}(\text{rand}, k)}
```

分母用 L1 在 k + m 个槽上的结果，因为 L4 每个 query 也读 k + m 个槽，这样比较是等预算的。主预筃 m = 16、k = 16 时分母是 L1(32)，它在 k 扫描里本来就有。SI = 1 表示这层的读取完全可以共享，0 表示完全个性化；大于 1 说明摘要加共享比每 query 自选还好，如实报导。同时报告用 L1(k) 做分母的版本。

**6.7 统计口径（全部沿用 v1）。**

- 固定 5,000 张 ImageNet 验证图（seed 0），和 v1 完全相同，所以 v1 结果可以逐图配对地并入 v2。
- 所有区间：按图片 bootstrap 1,000 次、seed 0、95%。任何两个设置的差值都是配对差值，先逐图相减再 bootstrap。
- 随机对照 5 个种子，先逐图平均再算区间；种子间标准差另列，不当置信区间。
- 不做多重比较校正，论文里说明。区间重叠的相邻层不说"名次"。
- 5,000 张图让所有区间都很窄（约 ±1 pp）。真正的不确定性来自设计选择（模型、m、路由），所以第 7 节每个选择都有消融或第二个模型。

## 7. 实验

三个问题串起来：每一层的注意力由什么构成（RQ1）→ 说明书要多长才忠实（RQ2）→ 说明书里的 token 是不是照片的主体，模型是否真的靠它们（RQ3）。RQ3 是可解释性的落脚点：MiTA 论文的 token pruning 图显示深层 top-k 只留下照片主体，我们要回答这是注意力的通性还是 MiTA 训练的产物，并用删除实验证明主体 token 是模型真正依赖的。

**7.1 共同设置**

| 项 | 设置 |
| --- | --- |
| 主模型 | DeiT-S（timm `deit_small_patch16_224.fb_in1k`，固定 revision，与 v1 相同） |
| 对照模型 | MiTA-DeiT-S（官方 Google Drive 权重，m = 25、k = 25，无 DWC），用我们的密集实现加载，全量 50,000 张验证集复现 79.8% 后才使用 |
| 数据 | ImageNet 验证集 5,000 张（seed 0，与 v1 同一个索引文件）；Oxford-IIIT Pet 测试集 3,669 张 |
| 主预算 | m = 16、k = 16（说明书 17×16 = 272 个编号，接近 MiTA 默认 25×25）；稳健性 k = 32 |
| 扫描 | k ∈ {4, 8, 16, 32, 64, 128}；m ∈ {1, 4, 16, 49, 196} |
| 执行 | FP32、batch 32、冻结原模型说明书；全部正式运行在同一种 GPU 上（Colab） |
| 单元测试 | k = 197 恒等；L3(m = 196, 按区域) = L1；L2(m = 1) 所有 query 输出相同；L4 与官方模块单批对照 |

**7.2 RQ1：每一层的注意力由什么构成？**

做法：每次只替换一层，其余 11 层原样，主预算 m = 16、k = 16。每层跑 9 种干预：rand（5 种子）、rand+C（5 种子）、L2、L3、L4、L1(16)、L1(32)、const、drop。共 12×17 = 204 个设置。同时在原模型前向里算每层每 head 的覆盖率和神经元并集。

产出：每层的 φ\_C、φ\_R、L1 上限、可共享指数 SI，和覆盖率、并集大小。

图 2（解剖图）：横轴 12 层；每层一根堆叠条，从 rand 往上叠 φ\_R（路由）和 φ\_C（压缩）到 L4；L1(32) 用标记点；const 和 drop 用短横线；噪声地板用虚线。下方一排小图：覆盖率和并集大小随层变化。

预计的读法：哪些层路由占大头（在读具体 token），哪些层压缩就够（只要概要），哪些层两者都不够（各读各的）。

**7.3 RQ2：说明书要多长才忠实？**

做法：所有层同时替换。

| 干预 | 设置 | 数量 | 来源 |
| --- | --- | --- | --- |
| L1 | 6 个 k | 6 | 新 |
| L2 | 5 个 m（含 const） | 5 | 新 |
| L3 | m = 16 × 6 k × 2 路由；m ∈ {1, 4, 49, 196} × k = 16 × 2 路由 | 12 + 7 | 新（CLS 独立）；v1 的 6 组成为 CLS 消融 |
| L4 | 同 L3 的网格，加 MiTA 默认 m = 25、k = 25 | 12 + 7 + 1 | 新 |
| rand | 6 k × 5 种子 | 30 | v1 已有 |
| rand+C | 主预算 × 5 种子 | 5 | 新 |
| K-means | 6 k | 6 | v1 已有 |
| 消融 | 主预算上：CLS 路由（L3/L4）、注意力平均打分（L3/L4）、标准缩放（L2/L4） | 6 | 新 |
| 噪声地板 | 1 像素平移 | 1 | 新 |

新增约 69 个全层设置。

图 3（预算前沿）：左图横轴 k（对数）、m = 16，线：L1、L3 按区域、L3 点积、L4、K-means、rand；右图横轴 m、k = 16，线：L2、L3、L4。两图都画噪声地板虚线和 90% 参考线。

记录：每条线达到噪声地板所需的最小 k；L4 比 L3 在每个 k 上多多少（摘要的价值随预算怎么变）；按区域路由比点积路由少多少（可读性的代价）。

**7.4 RQ3：说明书里的 token 是不是照片的主体？模型是否真的靠它们？**

三个子问题，同一套读取工具分别用在 DeiT-S 和 MiTA-DeiT-S 上：

| 子问题 | 做法 | DeiT-S | MiTA-DeiT-S |
| --- | --- | --- | --- |
| (a) 读取集合落在哪里 | Pet 测试集，observe 模式，每层读取集合的成分：前景非汇聚点 / 普通背景 / 汇聚点 / CLS（v1 的 E01–E02 定义）；复现 MiTA token pruning 图：每层读取集合的并集叠在原图上，3 张 seed 0 固定示例图 | v1 B1 已有（深层峰值多是汇聚点） | 新，m = 25、k = 25 和 m = 16、k = 16 都跑 |
| (b) 模型是否真的靠主体 token | v1 V01 的匹配删除：从读取集合里删前景 / 背景 / 汇聚点，按排名补满，和删同样数量的随机 token 比 | V01 已有（前景 −54 pp，汇聚点 ≈0） | 新，用它自己的 m = 25、k = 25 |
| (c) 哪条通道在读主体 | 在 MiTA-DeiT-S 上跑 L2、L3、L4（m = 25、k = 25）。L4 必须恒等（模型就是这样训练的）；L3 和 L2 各保留多少，告诉我们 MiTA 训练出的模型把信息放在路由通道还是压缩通道 | RQ1/RQ2 已有 | 新 |

关键对比：如果 MiTA-DeiT-S 深层读取集合的前景比例明显高于 DeiT-S，而 DeiT-S 的集合落在汇聚点上，那么"聚焦主体"是 MiTA 训练出来的；如果两者相似，那是注意力的通性。(b) 证明主体 token 不只是"被选中"，而是"删了预测就变"。

图 4：左图逐层前景比例，两个模型各一条线，配前景面积占比（51.2%）和汇聚点比例；中图 (b) 的删除效应条形图，两个模型并排；右图 3 张示例图 × 2 个模型，第 4、8、12 层的读取并集叠图。

注意：MiTA-DeiT-S 的前向必须照搬官方代码的网格環疵（CLS 算进网格、丢最后一个 patch）和双重缩放，否则复现不了 79.8%。但 (a) 的读取工具对两个模型用同一套干净网格，这样比较才公平；两种网格下的成分都报告。

**7.5 有时间再做**

- 汇聚点走哪条通道：L2 里把汇聚点从摘要的加权平均中排除，看一致率掉多少。掉得多就支持 Sun 等的"偏置项"说法。
- 用 Jiang 等的测试时寄存器替换 v1 的 DINOv2 对照（同一套权重）。
- 按 head 的覆盖率、并集、SI（附录图），和 Raghu 等的平均注意力距离对照。
- 第二个模型：DINO ViT-S/16 加官方线性头，或 DeiT-B，只重复图 2。
- v1 的其他 RQ3 结果（中心高斯、CLS 热图、pointing game、寄存器对照）全部进附录。

## 8. 运行前预测表（冻结）

跑之前把预测写下来，跑完后不改。每一行无论成立还是不成立，都是一句能写进结论的话。预测的依据是 v1 的数据和 MiTA 代码。

| 编号 | 预测 | 依据 | 若成立 | 若不成立 |
| --- | --- | --- | --- | --- |
| P1 | 浅层（1–3）的可共享指数 SI 最低，神经元并集最大 | v1 RQ2：k = 8 时第 2、3 层单层一致率最低；Raghu 等：浅层多局部 head | 浅层注意力各读各的，不可按区域概括，但代价小 | 浅层比想象中更模板化 |
| P2 | 深层（9–12）的 φ\_C（压缩）大于 φ\_R（路由） | v1 RQ3：深层读取集合被汇聚点支配；Sun 等：汇聚点像偏置项，摘要就能送到 | 深层主要靠"概要"，具体 token 次要 | 深层确实在读少数具体 token |
| P3 | 按区域路由比点积路由低不超过 3 pp（主预算，全层） | v1：网格和 K-means 分组相差不到 3 pp，说明分组方式对结果不敏感 | 可读的说明书几乎不付忠实性代价 | 点积路由的跨区域分配是必要的 |
| P4 | CLS 独立集合把第 12 层单层 KL 从 0.73 降到 0.15 以下 | v1：第 12 层 KL 是其他层 5 倍，而一致率并不最低 | 末层的异常是 CLS 被迫借用区域集合造成的 | 末层对任何限制都敏感 |
| P5 | L4 比 L3 在 k = 16 时高至少 10 pp，在 k = 128 时差距小于 2 pp | 摘要在短预算下补的是"漏掉的质量"，长预算下漏得少 | 摘要是便宜的补丁：16 个向量换来 10 pp | 摘要和具体 token 信息重叠 |
| P6 | L1(16) 全层一致率高于 90% | Gupta 等：top-k 在预训练模型上几乎无损；L1 没有共享的代价 | v1 丢掉的 30% 主要是共享的代价，不是稀疏的代价 | 稀疏本身就有代价，注意力的长尾很重要 |
| P7 | 覆盖率和一致率在第 1 层分道扬镳：覆盖率低、一致率高 | v1：第 1 层单层一致率最高，但局部注意力很难被 16 个共享 token 覆盖 | 第 1 层注意力对预测不重要，残差兑底；"看图"指标会误判 | 覆盖率是一致率的好预测量 |
| P8 | MiTA-DeiT-S 深层读取集合的前景比例比 DeiT-S 高至少 15 pp，汇聚点比例接近 0 | MiTA 的 token pruning 图来自 MiTA 训练的模型；v1 在 DeiT-S 上看到的是汇聚点 | "聚焦主体"是 MiTA 训练的产物；标准注意力把路由通道用来访问汇聚点 | 聚焦主体是注意力的通性，只是被热图里的汇聚点遮住了 |
| P9 | MiTA-DeiT-S 上 L3 保留的一致率高于 L2 | MiTA 把具体 token 当主要通道训练 | MiTA 训练出的模型把主体信息放在路由通道，这是它聚焦主体的机制 | MiTA 模型也主要靠摘要 |
| P10 | 注意力平均打分比 query 平均打分高 2 pp 以上 | V01：删背景 token 一致率从 78% 升到 83% | query 平均被大范数 key 带偏，选进了有害 token | 两种打分等价，landmark 的"平均性质"没有代价 |

预测阈值（3 pp、10 pp、15 pp 等）是事先定的整数，不按结果调。跟 v1 一样，配置文件在看到任何新结果之前提交并记录 commit。

## 9. 贡献与结果预案

**9.1 三条贡献**

- **C1 一把解剖注意力的尺子。**不训练、不改权重，把 MiTA 的压缩和路由当作两个可单独开关的零件，按图像区域组织，读出来的说明书人能直接读。
- **C2 一种把忠实性拆成成分的检验。**强制模型只按某一种简化计算，用 Shapley 分解把每层的忠实性拆成"可压缩、可路由、query 独有"三部分，并用噪声地板和覆盖率给每个数字定标尺。
- **C3 用这把尺子得到的发现。**DeiT-S 12 层的解剖图（RQ1），可读性和忠实性的权衡（RQ2），以及"聚焦主体"是注意力的通性还是 MiTA 训练的产物（RQ3）。

C1、C2 不依赖结果好坏；C3 由数据决定。"压缩 + 路由"的分解来自 MiTA，我们的新意在于把它变成测量工具。

**论文贡献列表（英文）**

1. A training-free probe that reads, from a pretrained ViT, two separately switchable summaries of each attention head: m region summaries (compression) and m shared top-k token sets (routing), organized by image regions so that a person can read them.
2. An execution test that forces the model to compute only through these summaries and, via a Shapley decomposition over the two switches, splits each layer's faithfulness into compressible, routable and query-specific parts, calibrated against a one-pixel-shift noise floor and against attention-mass coverage.
3. Findings: a layer-wise anatomy of DeiT-S attention, the price of readability on the length–faithfulness frontier, and a controlled comparison with MiTA's own trained model showing whether top-k object focus is learned under MiTA or intrinsic to attention.

**9.2 结果预案：解剖图的三种形状**

| 形状 | 论文的故事 | 主角 |
| --- | --- | --- |
| A：压缩占大头，路由和独有部分都小 | ViT 的注意力大部分是"概要"，m 个摘要向量就能概括一层；具体 token 的选择没那么重要，热图式解释的根基被动摇 | 图 2，配图 3 的 L2 线 |
| B：路由占大头 | 注意力确实在读少数具体 token，而且同一区域的 query 读的是同一组；说明书可以很短，但必须是具体 token | 图 2，配图 4 的主体对比 |
| C：随层变化，浅层独有、深层压缩或路由 | 注意力的"可读性"是层的性质：浅层各读各的但对预测不重要，深层可概括；我们给出分界在哪一层 | 图 2 + 图 4 |

三种情况下 C1、C2 都成立。从 v1 的数据（浅层单层一致率最低、深层被汇聚点支配）看，C 最可能，且深层偏向压缩。

**9.3 RQ3 的两种结局，都能写**

- MiTA-DeiT-S 聚焦主体、DeiT-S 聚焦汇聚点：结论是"MiTA 的训练把主体信息迁移到了路由通道；标准注意力里这条通道被汇聚点占用"。这解释了 MiTA 论文的图，也解释了 v1 为什么没看到同样的现象。
- 两者都聚焦主体：结论是"聚焦主体是注意力的通性，标准 ViT 也学到了照片的内容；v1 的热图被少数汇聚点的峰值遮住了"。删除实验 (b) 在两种结局下都证明主体 token 是模型真正依赖的。

## 10. 四页版面规划

课程要求 NeurIPS 模板、正文最多 4 页（含图表）。三张主图和核心公式必须在正文。正文只讲一个模型（DeiT-S）、一个主预算（m = 16、k = 16），其他全进附录。

| 页 | 章节 | 内容 | 图表 |
| --- | --- | --- | --- |
| 1 | 摘要、1 引言 | 五句故事展开成三段，末尾三条贡献 | 图 1：方法示意（原注意力 → 读出摘要和共享集合 → 四种执行方式 → Shapley 分解） |
| 2 | 2 相关工作、3 方法 | 五组文献各一句，三条边界；快权重 MLP（一个公式）；读取（landmark、摘要、集合、分配）；干预阶梯（L4 的合成 softmax 公式）；Shapley 和 SI 的定义 | — |
| 3 | 4 实验：设置、RQ1、RQ2 | 共同设置一段（含噪声地板）；RQ1 解剖图一段；RQ2 前沿一段 | 图 2、图 3 |
| 4 | 4 实验：RQ3；5 讨论与局限；6 结论 | RQ3 一段；局限（深层 token 的信息混合、只测一个主模型、冻结说明书的状态失配）；结论回到引言的空白 | 图 4 |

**附录：**k 和 m 扫描的完整表；四个消融；v1 的 RQ3 热图比较、V00–V03；按 head 的图；第二个模型；实现细节和单元测试；预测表和对照结果。

**如果只完成了 RQ1：**第 3 页放大图 2，加入 k = 32 的解剖图作稳健性；第 4 页用 v1 的全层曲线加 L1、L4 两条新线作图 3，图 4 换成 v1 的 V01 删除实验。论文仍是完整的"问题 → 方法 → 检验 → 结论"。

## 11. 执行计划

**11.1 优先级**

| 级别 | 内容 | 没有它论文怎样 |
| --- | --- | --- |
| P0 | L1、L2、L4、rand+C、const、drop；按区域路由；CLS 独立；噪声地板；单层解剖（图 2）和全层前沿（图 3） | 没有就没有新主线 |
| P1 | 覆盖率、神经元并集、SI；四个消融 | 发现偏浅 |
| P2 | MiTA-DeiT-S：密集实现、复现 79.8%、Pet 成分、token pruning 图、L2/L3/L4、匹配删除（图 4） | 少了可解释性的落脚点 |
| P3 | 汇聚点通道、测试时寄存器、按 head、第二个模型 | 局限性里多一句话 |

**11.2 算力**

| 部分 | 前向次数 |
| --- | --- |
| 单层解剖：204 设置 × 5,000 | 1,020,000 |
| 全层前沿：69 设置 × 5,000 | 345,000 |
| 覆盖率、并集：随原模型前向顺手算 | 0 |
| MiTA-DeiT-S 全量验证集复现 | 50,000 |
| MiTA-DeiT-S L2/L3/L4 × 5,000；Pet observe × 2 模型 × 2 预算 | 30,000 |
| MiTA-DeiT-S 匹配删除（只做 all12、k = 25：3 类别 + 5 种子随机 + 基准）× 3,669 | 33,000 |
| 合计 | 约 150 万，按 v1 在 L4 上的吞吐（约 2.5 万/分钟）约 60 分钟 |

全部在同一张 Colab GPU 上完成；不使用本地 GPU，不启动 HPC。

**11.3 复用 v1 的结果（同一个 5,000 张索引，可逐图配对）**

| v1 目录 | 在 v2 里的身份 |
| --- | --- |
| `2026-10-27_rq1_colab_combined` | L3 点积路由、CLS 路由的 6 个 k；rand 6 k × 5 种子；K-means 6 k |
| `2026-10-26_rq2_imagenet`、`2026-10-27_v02_colab` | L3 单层 k = 8/32（CLS 路由），作为 k = 32 稳健性和 CLS 消融 |
| `2026-10-27_v03_colab` | 现场重算的负结果，进附录 |
| `2026-10-26_rq3_pet`、`rq3_diagnosis`、`2026-10-27_v01_colab` | RQ3 (a)(b) 的 DeiT-S 一半 |
| `2026-10-26_rq3_register`、`rq3_dino` | 附录的寄存器对照 |

**11.4 代码改动（在现有 `mita_probe` 上加，不改已有函数的行为）**

| 模块 | 新增 |
| --- | --- |
| `probe.py` | `per_query_readsets(q, k, k_top)`；`spatial_routes(grid, m_side)`；`cls_readset`；`attention_mean_scores`（注意力平均打分） |
| `summary.py`（新） | `region_summaries(landmarks, k, v)`；`dense_mixed_attention(q, k, v, lm, summary, allowed, summary_scale)` |
| `models.py` | `execute_mode(model, mode, patterns, summaries)`：替换 `block.attn.forward` 的上下文管理器，mode ∈ {L1, L2, L3, L4, randC, const, drop}；`coverage_and_union`（读取时顺手算） |
| `mita_model.py`（新） | 照搬官方网格環疵和双重缩放的密集 MiTA attention；从官方 checkpoint 加载到 timm DeiT-S 骨架 |
| `data.py` | 噪声地板变换：裁剪窗口右移 1 像素 |
| `analysis/anatomy.py`（新） | 逐图 Shapley、SI、配对 bootstrap，输出 CSV |
| `scripts/run_anatomy.py`、`run_frontier.py`、`run_mita_contrast.py` | 三个 RQ 的入口，只读配置，支持断点续跑 |
| `scripts/plot_fig2/3/4.py` | 只读 CSV |

**11.5 单元测试（全部通过才开始正式运行）**

1. §4.3 的手算例子：L0–L4 四种输出逐个对上。
2. k = 197 时 L1、L3 与原模型逐位一致（5,000 张）。
3. L3(m = 196，按区域路由) 与 L1 逐位一致。
4. L2(m = 1)：同一层内所有 query 的注意力输出相同。
5. drop 模式：该层块输出等于输入加 MLP 分支。
6. `allowed` 全开且无摘要槽时，密集实现与 timm 原 forward 逐位一致。
7. L4 与官方 `MiTA_Attention` 在同一批输入上误差 < 1e-3（Colab 装 flash-attn 2.6.3，fp16，一次性）。
8. MiTA-DeiT-S 在 50,000 张验证集上 top-1 与 79.8% 相差不超过 0.2 pp。
9. 覆盖率：k = 197 时恒为 1；rand 的覆盖率期望约为 k/197。
10. Shapley：φ\_C + φ\_R = L4 − rand 逐图成立。
11. 随机对照：生成器按 (k, seed) 独立、跨 batch 连续，与 v1 相同。

**11.6 流程**

1. 在 GitHub 项目开新分支 `feature/anatomy-v2`，配置文件和预测表先提交。
2. 本地 CPU 跑全部单元测试（1–6、9–11）。
3. Colab：恢复资产（模型、ImageNet 5,000 张、Pet），核对 SHA-256；跑测试 7、8；先 200 张 smoke，再正式。
4. 按 P0 → P1 → P2 顺序运行，每部分归档（config、indices、metadata、per-image CSV、checks.json）并 push。
5. 分析和画图在本地 CPU 完成，只读 CSV。
6. 结束时停止 Colab 实例，记录余额。

## 12. 常见 QA

**Q1：和 MiTA 有什么不同？**MiTA 是一种新的注意力结构，要用它训练模型，目的是加速。我们不训练，把它的两个零件当尺子去量一个普通 ViT；再用它训练出的模型做对照。

**Q2：为什么要拆成压缩和路由，而不是像 v1 那样只测一条曲线？**v1 的曲线只说"丢了多少"，说不出"丢的是什么"。拆成两个零件后，每层得到三个成分，结果从一个数字变成一张解剖图。

**Q3：摘要槽是新加的位置，执行结果还能叫"原模型的计算"吗？**摘要是原模型的 K、V 用原模型的 landmark 加权得到的，没有任何新参数。L2 问的是"如果这层只能看到区域平均读到的内容，预测还在吗"，和 Hassid 等的常数注意力是同一类问题。论文里要说清 L2、L4 不嵌套在 L0 里。

**Q4：Shapley 为什么只有两个玩家？**因为只有两个可开关的零件。两个玩家的 Shapley 值就是两个边际贡献的平均，不需要任何近似。"query 独有"的部分不是玩家，它是 L1 上限和 L4 的差。

**Q5：为什么 SI 的分母用 L1(k + m)？**L4 每个 query 读 k 个 token 加 m 个摘要，共 k + m 个槽。和读 k + m 个 token 的 L1 比，是等预算的比较。用 L1(k) 的版本也报告。

**Q6：按区域路由和点积路由，论文主用哪个？**主用按区域，因为只有它的说明书是"区域 r 读 S\_r"。点积路由是 MiTA 原样，作为消融报告差值。如果差值很大，说明跨区域分配是必要的，这本身也是发现。

**Q7：为什么用 MiTA-DeiT-S 而不是 DINO 做对照？**MiTA-DeiT-S 和 DeiT-S 结构完全相同、训练数据相同，只差注意力形式。它直接回答"MiTA 论文里的聚焦现象是不是训练出来的"。DINO 差的是监督方式，回答的是另一个问题，放选做。

**Q8：深层的 token 还对应原图位置吗？**不完全对应。读取集合说的是"这一层中被读取的 token 位置"。局限性里说明，和 v1 相同。

**Q9：覆盖率和一致率哪个是主指标？**一致率。覆盖率不需要干预就能算，它的作用是显示"看图"式指标在哪些层会误判。

**Q10：噪声地板为什么是 1 像素平移，不是加噪声？**平移 1 像素不改变图片内容，任何人都会同意"模型应该给出同样的答案"。加噪声要选强度，多一个可以被质疑的参数。

## 附录

**A. 模型**

| 用途 | 模型 | 来源 | 说明 |
| --- | --- | --- | --- |
| 主实验 | DeiT-S/16 | timm `deit_small_patch16_224.fb_in1k`，固定 revision 同 v1 | 22.1M 参数，197 token，原模型 5,000 张准确率 79.62% |
| 对照 | MiTA-DeiT-S | [MiTA 仓库 Model Zoo](https://github.com/QishuaiWen/MiTA)，Google Drive，报告 79.8% | m = 25、k = 25，无 DWC、无 gate；权重键名与 timm DeiT-S 相同（qkv、proj），池化无参数 |
| 选做 | DINO ViT-S/16 | timm `vit_small_patch16_224.dino` + 官方线性头 | 同结构、自监督 |
| 选做 | DeiT-B/16 | timm `deit_base_patch16_224.fb_in1k` | 同训练、更大 |

**B. 数据**

| 用途 | 数据集 | 规模 | 获取 |
| --- | --- | --- | --- |
| RQ1、RQ2、MiTA 复现 | ImageNet-1k 验证集 | 5,000 张（seed 0，索引文件同 v1）；复现用全部 50,000 张 | v1 已下载的固定版本，上传 Colab 后核对 SHA-256 |
| RQ3 | Oxford-IIIT Pet 测试集 | 3,669 张，带 trimap | v1 已有，前景定义沿用 E02 |

**C. 和 v1 的差异清单**

| 项 | v1 | v2 |
| --- | --- | --- |
| 研究问题 | 短说明书够不够忠实 | 注意力有多少可压缩、可路由、query 独有 |
| 干预 | 只有严格版（L3）+ 随机 + K-means | L1、L2、L3、L4、rand、rand+C、const、drop |
| 路由 | 点积 argmax | 按区域（主）+ 点积（消融） |
| CLS | 路由到某个 landmark | 自己的 top-k 集合 |
| 主预算 | k ∈ {8, 32}（RQ2） | m = 16、k = 16；k = 32 稳健性 |
| 指标 | 一致率、KL | + 保留精度、噪声地板、覆盖率、并集、Shapley、SI |
| RQ3 | Pet 热图对比中心高斯、CLS 热图；寄存器对照 | DeiT-S 对 MiTA-DeiT-S；热图对比进附录 |
| 主图 | k 曲线、逐层曲线、前景曲线 | 解剖图、预算前沿、两模型对照 |
| 现场重算 | V03 做了，更差 | 不再做，引用 V03 |

**D. 参考文献**

- [MiTA: Mixture-of-Top-k Attention（Wen 等，NeurIPS 2026）](https://arxiv.org/abs/2602.01219)，[代码](https://github.com/QishuaiWen/MiTA)
- [Nyströmformer（Xiong 等，AAAI 2021）](https://arxiv.org/abs/2102.03902)
- [Agent Attention（Han 等，ECCV 2024）](https://arxiv.org/abs/2312.08874)
- [Memory-efficient Transformers via Top-k Attention（Gupta 等，SustaiNLP 2021）](https://arxiv.org/abs/2106.06899)
- [Fast Transformers with Clustered Attention（Vyas 等，NeurIPS 2020）](https://arxiv.org/abs/2007.04825)
- [Linear Transformers Are Secretly Fast Weight Programmers（Schlag 等，ICML 2021）](https://arxiv.org/abs/2102.11174)
- [Massive Activations in LLMs（Sun 等，COLM 2024）](https://arxiv.org/abs/2402.17762)
- [Vision Transformers Need Registers（Darcet 等，ICLR 2024）](https://arxiv.org/abs/2309.16588)
- [Vision Transformers Don't Need Trained Registers（Jiang 等，NeurIPS 2025）](https://arxiv.org/abs/2506.08010)
- [How Much Does Attention Actually Attend?（Hassid 等，2022）](https://arxiv.org/abs/2211.03495)
- [Are Sixteen Heads Really Better than One?（Michel 等，NeurIPS 2019）](https://arxiv.org/abs/1905.10650)
- [What Matters in Transformers? Not All Attention is Needed（He 等，2024）](https://arxiv.org/abs/2406.15786)
- [ERASER（DeYoung 等，ACL 2020）](https://arxiv.org/abs/1911.03429)
- [Dissecting Recall of Factual Associations（Geva 等，2023）](https://arxiv.org/abs/2304.14767)
- [Attention is not Explanation（Jain & Wallace，NAACL 2019）](https://arxiv.org/abs/1902.10186)；[Attention is not not Explanation（Wiegreffe & Pinter，EMNLP 2019）](https://arxiv.org/abs/1908.04626)
- [Do Vision Transformers See Like CNNs?（Raghu 等，NeurIPS 2021）](https://arxiv.org/abs/2108.08810)
- [Intriguing Properties of Vision Transformers（Naseer 等，NeurIPS 2021）](https://arxiv.org/abs/2105.10497)
- [Quantifying Attention Flow（Abnar & Zuidema，ACL 2020）](https://arxiv.org/abs/2005.00928)
- [Transformer Interpretability Beyond Attention Visualization（Chefer 等，CVPR 2021）](https://arxiv.org/abs/2012.09838)
- [RISE（Petsiuk 等，BMVC 2018）](https://arxiv.org/abs/1806.07421)
- [DeiT（Touvron 等，ICML 2021）](https://arxiv.org/abs/2012.12877)；[DINO（Caron 等，ICCV 2021）](https://arxiv.org/abs/2104.14294)

本文档中 MiTA 的机制细节（m = 25、k = 25、双重缩放、网格環疵、合成 softmax）全部来自官方代码 commit 95460f7（2026-10-07），不是论文正文。论文 PDF 到手后核对并更新。
