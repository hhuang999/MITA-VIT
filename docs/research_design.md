# 研究设计书：用读取模式解释 ViT 的注意力

Oct 24, 2026 · @Kevin

## 1. 研究问题与主张

**问题**：一个训练好的 ViT，它每一层的注意力，能不能用“m 块区域、每块在读哪 k 个 token”这样一份很短的说明概括出来？这份说明是否忠实于模型真实的计算？

*Can each attention layer of a trained ViT be summarized as “m regions, each reading k tokens”, and is this summary faithful to what the model actually computes?*

**主张**：借用 MiTA 的 landmark + top-k 机制，我们可以在不改动模型的前提下，从原模型中读出这份说明；再强制模型只按这份说明计算，看预测是否改变，就能直接检验它是否忠实。用这把“尺子”，我们测量 ViT 的注意力在多大程度上、在哪些层能被这样概括。

*Borrowing MiTA’s landmark + top-k mechanism, we read this summary out of the original model without changing it, and test its faithfulness directly by forcing the model to compute only through it. With this ruler we measure how far, and in which layers, ViT attention can be summarized this way.*

主张只承诺“提出工具并用它去测量”，不承诺“一定能概括得好”，所以实验结果无论好坏都能写成结论（见第 7 节）。

## 2. 研究故事

**第 1 步：背景——ViT 的注意力为什么“读不懂”**

以 DeiT-S 为例：输入 224×224 的图片，切成 16×16 的小块，得到 14×14 = 196 个 patch token，再加 1 个 CLS token（分类时只看它的最终输出），一共 197 个 token。每一层、每一个 head 里，注意力做的事情是：

```latex
A = \text{softmax}(QK^\top/\sqrt{d}),\quad \text{out} = AV
```

A 是一个 197×197 的矩阵，第 i 行表示“第 i 个 token 在读谁、读多少”。DeiT-S 有 12 层、每层 6 个 head，所以一张图片对应约 280 万个注意力数值。没有人能直接读懂这么多数。

**第 2 步：现有解释的两个问题**

现有方法通常把这些数压成一张“模型在看哪里”的热图，比如只看最后一层 CLS 那一行，或者用 rollout 把各层乘起来。这样做有两个问题：

1. 丢失了结构：热图只说明“哪里重要”，不说明“哪个区域在读哪个区域”；
2. 忠实性难以检验：没法直接证明这张热图真的反映了模型的计算过程。“注意力能不能当解释”至今仍有争论（Jain & Wallace 2019；Wiegreffe & Pinter 2019）。

**第 3 步：关键洞察——注意力是一个临时的 MLP，而且神经元被大量共享**

注意力可以改写成一个两层 MLP：每个 token 贡献一个隐藏神经元，它的 key 是这个神经元的输入权重，value 是输出权重（第 4 节用手算例子展开）。这些权重是由当前这张图片临时算出来的，所以叫“快权重”。

MiTA 的实验说明，许多 query 激活的神经元高度重叠：只让每个 query 读一小组共享的 top-k token，模型仍能保持大部分性能；把用标准注意力训练的模型在推理时直接换成 MiTA，精度能保留 95% 以上。这暗示一层注意力的结构，可能远比 197×197 个数简单。

**第 4 步：我们的方法——先读出“说明书”，再强制模型按说明书计算**

- **读取**：把 14×14 的网格分成 m 块区域，每块区域的 query 取平均，得到一个 landmark；找出 landmark 打分最高的 k 个 token，作为这块区域的读取集合。每个 query 再被分配给与它最相似的那个 landmark。这一步不改动原模型。
- **检验**：让每个 query 只能读它被分配到的那个读取集合，其余 token 全部屏蔽，看模型的预测是否和原来一致。

一份说明只有 m×k 个 token 编号。比如 m = k = 16 时是 256 个编号，而原来每层每个 head 有约 3.9 万个注意力数值。

**第 5 步：预期发现**

实验会给出三类结果：一致率随 k 变化的曲线；哪些层容易被概括、哪些不容易；读取集合是否集中在物体上。具体结论等实验完成后填写，三种可能的版本见第 7 节。

**英文版（写论文引言用）**

1. ViTs pass information between image patches through attention, and attention maps are widely used to explain where a ViT “looks”.
2. Yet one image produces about 2.8 million attention values in DeiT-S; existing methods compress them into a single heatmap and cannot directly verify that the heatmap reflects the model’s computation.
3. MiTA suggests two things: attention is a temporary two-layer network in which each token acts as a neuron, and many positions reuse the same few groups of neurons, so a handful of reading patterns may describe a layer.
4. We split the image into m regions, take each region’s average query to find the k tokens it attends to most, and use these as a summary of the layer; we then force the model to compute only through this summary and check whether its prediction changes.
5. With this ruler, we measure how far, and in which layers, ViT attention can be summarized this way, and whether the reading patterns concentrate on objects.

## 3. 相关工作与空白

和我们相关的工作有三类。我们的视角是把第二类的机制和第三类的检验思路结合起来，用来做第一类的事。

**第一类：用注意力解释 ViT。** 最常见的做法是画最后一层 CLS 那一行的注意力（DINO 论文中的注意力图就是这样画的）；[Attention Rollout](https://arxiv.org/abs/2005.00928)（Abnar & Zuidema 2020）把各层注意力矩阵连乘，得到对输入 patch 的整体权重；[Chefer 等](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)（CVPR 2021）进一步结合梯度传播相关性。这些方法的输出都是一张热图，忠实性只能间接评估。“注意力能不能当解释”至今仍有争论（Jain & Wallace 2019；[Wiegreffe & Pinter 2019](https://arxiv.org/abs/1908.04626)）。

**第二类：“分组 + top-k”的高效注意力。** [Clustered Attention](https://arxiv.org/abs/2007.04825)（Vyas 等，NeurIPS 2020）先用 K-means 把 query 分组，再为每组找注意力最高的 top-k 个 key，只用 25 个组就近似了预训练的 BERT。[MiTA](https://arxiv.org/abs/2602.01219)（2026）用平均池化得到 landmark，再取 top-k，并发现标准注意力训练的模型推理时换成 MiTA，精度能保留 95% 以上。这两项工作的目的都是加速。

**第三类：用“只保留解释”来检验忠实性。** NLP 中的 [ERASER](https://arxiv.org/abs/1911.03429)（DeYoung 等 2020）提出“充分性”：只把解释选出的那部分输入给模型，看预测是否保持。[Attention Knockout](https://arxiv.org/abs/2304.14767)（Geva 等 2023）则屏蔽注意力中的某些连接，看预测如何变化。

**我们的视角。** 第二类工作说明，注意力的结构可以被少数分组和 top-k 集合概括；但它们只把这一点用于加速。我们把同样的机制当作解释工具：按图像区域分组，使每一组都对应一块看得见的区域，概括出来的说明书人可以直接读；再借用第三类的思路，强制模型只按说明书计算，直接检验它是否充分。这同时补上了第一类方法的两个不足：读不出结构，以及忠实性难以检验。

## 4. 必备基础知识

**4.1 ViT 一层在算什么**

DeiT-S 的输入是 197 个 token，每个 token 是一个 384 维向量，排成矩阵 X（197×384）。每一层先做注意力，再做 MLP，两步都带残差：

```latex
x \leftarrow x + \text{Attn}(\text{LN}(x)),\qquad x \leftarrow x + \text{MLP}(\text{LN}(x))
```

注意力分成 6 个 head，每个 head 用 64 维。在一个 head 里，先用三个可学习的矩阵把 X 变成 query、key、value：Q = X W\_Q，K = X W\_K，V = X W\_V，形状都是 197×64。然后：

```latex
A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right),\qquad \text{out} = AV,\qquad d = 64
```

A\[i, j\] 表示第 i 个 token 从第 j 个 token 那里拿多少信息，每一行加起来等于 1。12 层算完后，只取 CLS token 的输出，接一个线性层得到 1000 类的分数。

**4.2 手算一个最小的例子**

只有 3 个 token、每个向量 2 维；为了好算，这里省略除以 √d。

- keys：k1 = (1, 0)，k2 = (0, 1)，k3 = (1, 1)
- values：v1 = (1, 0)，v2 = (0, 1)，v3 = (2, 2)
- 某个 query：q = (2, 0)

第一步，打分：q·k1 = 2，q·k2 = 0，q·k3 = 2。

第二步，取指数：e² ≈ 7.39，e⁰ = 1，e² ≈ 7.39，总和 15.78。

第三步，归一化（这就是 softmax）：权重 ≈ (0.468, 0.063, 0.468)。

第四步，加权求和：out = 0.468·(1, 0) + 0.063·(0, 1) + 0.468·(2, 2) ≈ (1.405, 1.000)。

这个 query 主要在读 token 1 和 token 3，几乎不读 token 2。

**4.3 同一个计算，换成 MLP 的读法**

普通的两层 MLP（就是 Transformer 里的 FFN）是：

```latex
y = W_2\,\sigma(W_1^\top x)
```

W₁ 的每一列是一个隐藏神经元的输入权重，σ 是激活函数，W₂ 的每一列是这个神经元的输出权重。

把 4.2 的计算重新读一遍：输入是 q；3 个隐藏神经元的输入权重分别是 k1、k2、k3；激活函数是 exp（再除以总和做归一化）；输出权重分别是 v1、v2、v3。隐藏层的激活值就是 (7.39, 1, 7.39)，输出就是 (1.405, 1.000)，和上面完全一样。

| 普通 MLP | 注意力 |
| --- | --- |
| 输入 x | query q |
| 第一层权重 W₁ 的第 j 列 | 第 j 个 token 的 key |
| 激活函数 σ | exp，再归一化 |
| 第二层权重 W₂ 的第 j 列 | 第 j 个 token 的 value |
| 隐藏层宽度 | token 数 N（DeiT-S 中是 197） |

所以注意力就是一个两层 MLP，**每个 token 贡献一个隐藏神经元**。区别在于：FFN 的权重训练完就固定了，叫慢权重；注意力的“权重” K 和 V 是由当前这张图片算出来的，换一张图就变，所以叫快权重。这个等价关系是已有的代数事实（Schlag 等 2021 已经讨论过），MiTA 用它统一了各种高效注意力方法。

**4.4 这个视角对我们有什么用**

在 MLP 的读法下，“第 i 个 token 在读谁”就变成了“第 i 个输入激活了哪些神经元”。4.2 的例子里，q 激活了神经元 1 和 3，几乎没有激活神经元 2。

MiTA 的观察是：很多 query 激活的是同一小组神经元。如果真是这样，就不需要为 197 个 query 各写一份说明，只要写少数几份“这一组 query 都在激活哪几个神经元”就够了。这就是第 5 节方法的出发点。

## 5. 方法：读取模式与执行模式

方法分两步：**读取模式**从原模型里读出一份说明，不改动模型；**执行模式**强制模型只按这份说明计算，用来检验说明是否忠实。下面的计算都在某一层、某一个 head 里进行，每个 head 各算各的。

**5.1 读取模式：四步算出“说明书”**

第一步，划分区域。把 14×14 的 patch 网格均匀分成 m 块区域（m = 16 时是 4×4 块，每块约 3.5×3.5 个 patch）。记第 r 块区域包含的 patch 集合为 R\_r。CLS token 不属于任何区域。

第二步，算 landmark。把区域内所有 patch 的 query 取平均：

```latex
\tilde q_r = \frac{1}{|R_r|}\sum_{i \in R_r} q_i
```

第三步，找读取集合。用 landmark 给全部 197 个 token 打分，取分数最高的 k 个：

```latex
s_{r,j} = \frac{\tilde q_r \cdot k_j}{\sqrt d},\qquad S_r = \text{TopK}_j\,(s_{r,j},\ k)
```

因为点积对 query 是线性的，landmark 的打分恰好等于区域内各 query 打分的平均：

```latex
\tilde q_r \cdot k_j = \frac{1}{|R_r|}\sum_{i \in R_r} q_i \cdot k_j
```

所以 S\_r 的含义很直接：区域 r 作为一个整体，最想读的 k 个 token。

第四步，分配 query。每个 query（包括 CLS）被分配给和它最相似的 landmark：

```latex
e(i) = \arg\max_r\ q_i \cdot \tilde q_r
```

注意 query 不一定被分到自己所在的区域。比如背景里的某个 patch，可能和“狗头区域”的 landmark 最相似，就被分到那里。

**说明书**就是这两样东西：m 个读取集合 S₁…S\_m（共 m×k 个 token 编号），以及每个 query 的分配结果 e(i)。

**5.2 手算一个读取的例子**

4 个 patch token 排成一行，分成两块区域：A = {1, 2}，B = {3, 4}，k = 2。省略 √d。

- queries：q1 = (3, 1)，q2 = (−1, 1)，q3 = (1, −1)，q4 = (1, −1)
- keys：k1 = (1, 0)，k2 = (0, 2)，k3 = (2, 2)，k4 = (0, −1)

算 landmark：A = (q1 + q2)/2 = (1, 1)，B = (q3 + q4)/2 = (1, −1)。

算打分：A 对 4 个 key 的打分是 (1, 2, 4, −1)，取最高的 2 个，S\_A = {3, 2}；B 的打分是 (1, −2, 0, 1)，S\_B = {1, 4}。

验证“平均”性质：q1 的打分是 (3, 2, 8, −1)，q2 的打分是 (−1, 2, 0, −1)，两者平均正好是 (1, 2, 4, −1)。

分配 query：q1·A = 4，q1·B = 2，所以 q1 分给 A；同理 q2 分给 A，q3、q4 分给 B。

说明书读出来就是：“区域 A 主要在读 token 3 和 2；区域 B 主要在读 token 1 和 4。”原来的注意力是 4×4 = 16 个数，说明书只有 2×2 = 4 个编号。

**5.3 执行模式：强制模型按说明书计算**

**严格版本**：query i 只能读它所分配的读取集合 S\_e(i) 里的 token，其余 token 的分数设为 −∞，再在剩下的 token 上做 softmax：

```latex
\tilde A_{ij} = \frac{\exp(q_i\cdot k_j/\sqrt d)\;\mathbb{1}[\,j \in S_{e(i)}\,]}{\sum_{j' \in S_{e(i)}} \exp(q_i\cdot k_{j'}/\sqrt d)},\qquad \widetilde{\text{out}}_i = \sum_j \tilde A_{ij}\, v_j
```

注意，打分仍然用 query 自己的 q\_i，模型权重完全不变，只是“能读谁”被限制了。

接着 5.2 的例子：q1 分给 A，只能读 token 2 和 3。原来 q1 对 4 个 token 的权重是 (0.007, 0.002, 0.991, 0.0001)；执行模式下变成 (0, 0.002, 0.998, 0)。几乎没变，说明对 q1 来说这份说明书是够用的。

**宽松版本（就是完整的 MiTA）**：除了读取集合，每个 query 还能读 m 个“区域摘要”。区域 r 的摘要用 landmark 当 key，value 是 landmark 对全部 token 的加权平均：

```latex
\tilde v_r = \sum_j \text{softmax}_j(s_{r,j})\, v_j
```

严格版本只检验“精确读取”够不够；宽松版本再加上“全局摘要”。两者的差距说明模型有多依赖全局摘要。

**两种替换范围**：

- **所有层同时替换**：检验整份说明书合起来够不够，结果是一个数字，作为 RQ1 的主结果；
- **每次只替换一层**：其余层保持原样，用来找出哪一层最难被概括（RQ2）。

**单元测试**：当 k = 197 时，每个 query 都能读全部 token，执行模式的输出必须与原模型逐位一致。

**5.4 怎么衡量**

**说明规模**（衡量可读性）：一层一个 head 的说明书有 m × k 个 token 编号。原注意力是 197 × 197 ≈ 3.9 万个数。

**预测一致率**（衡量忠实性，主指标）：在 N 张图片上，执行模式的预测类别与原模型相同的比例：

```latex
\text{Agree} = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\big[\arg\max f(x_n) = \arg\max \tilde f(x_n)\big]
```

这里比较的是原模型自己的预测，不是真实标签，因为我们要解释的是模型。

**辅助指标**：替换前后预测分布的 KL 散度；对于没有分类头的模型（如 DINO），比较替换前后 CLS 最终特征的余弦相似度。

**前景比例**（RQ3 用）：在带前景标注的图片上，被选入读取集合的 token 中落在前景的比例：

```latex
\text{FG} = \frac{\#\{\text{被选中且在前景的 token}\}}{\#\{\text{被选中的 token}\}}
```

比较时要和“随机选 token 时的期望”（即前景面积占比）对照。

## 6. 实验与评估

实验由三个问题串起来：说明书够忠实吗（RQ1）→ 哪些层能被概括（RQ2）→ 说明书看到了什么（RQ3）。只做完 RQ1 就是一个完整的闭环，RQ2、RQ3 依次加深。整个实验部分只有三张主图。

**6.1 共同设置**

- 模型：DeiT-S（timm 名称 `deit_small_patch16_224.fb_in1k`）。
- 数据：调试用 Imagenette；RQ1、RQ2 用 ImageNet 验证集中随机抽取的 5,000 张图片（固定随机种子 0）；RQ3 用 Oxford-IIIT Pet 测试集（自带猫狗的前景标注）。
- 说明书设置：m = 16（4×4 块区域），k ∈ {4, 8, 16, 32, 64}，执行用严格版本。
- 单元测试：k = 197 时一致率必须是 100%。

**6.2 RQ1：说明书够忠实吗？**

做法：所有层同时替换成说明书，在 5,000 张图上计算预测一致率，k 从 4 增加到 64。

同时画两条参照线：

- **随机读取集合**：每块区域随机选 k 个 token，其余步骤不变，跑 5 个随机种子。它告诉我们，“选对 token”比“随便选同样多的 token”多带来多少信息。
- **K-means 分组**：把分组方式从“按图像区域”换成“按 query 的相似度”（即 Clustered Attention 的分组方式），其余步骤不变。它告诉我们，用人能读懂的区域来分组，忠实性和按特征分组相比如何。

期待看到：我们的曲线明显高于随机参照，并随 k 增大快速上升。同时记录一致率达到 90% 时所需的最小 k，作为“一层注意力需要多长的说明书”的答案。

**图 2**：横轴是 k（对数刻度），纵轴是一致率，三条线分别是我们的方法、随机参照、K-means 分组。

**6.3 RQ2：哪些层能被概括？**

做法：每次只把一层替换成说明书，其余 11 层保持原样，得到 12 个一致率。取 k = 8 和 k = 32 两组。

期待看到：不同层的一致率有明显差异。方向不预设：MiTA 图 4 显示深层的读取集中在少数 token 上，深层可能更容易概括；但浅层以局部注意力为主，也可能很容易概括。无论是哪种，都直接告诉我们“模型的哪一部分是可读的”。

**图 3**：横轴是层号 1–12，纵轴是只替换该层时的一致率，两条线对应两组 k。

**6.4 RQ3：说明书看到了什么？**

做法：在 Pet 测试集上，逐层计算读取集合的前景比例（5.4 节的 FG）。再把最后几层的选中次数图（每个 token 被多少个读取集合选中，对 16 块区域和 6 个 head 求和）上采样到图片大小，当作一张热图。

同时画两条参照：

- **CLS 注意力热图**：最常用的注意力解释，看我们的热图和它有什么不同；
- **中心高斯图**：一张固定的、中间亮四周暗的图。猫狗通常在图片中央，这条参照告诉我们“看中间”能得多少分。

评估：前景比例，以及 pointing game（热图最亮的点是否落在前景上）。前景标注中的边界像素不参与计算。

期待看到：深层读取集合的前景比例明显高于前景面积占比，也高于中心参照。

**图 4**：左边是逐层前景比例曲线；右边是 3 张例子，每张依次是原图、选中次数图、CLS 注意力图，以及一块背景区域在读哪些 token。

**6.5 有时间再做**

以下内容不进入核心设计，有时间时再选做：

- 更多热图方法作参照：Attention Rollout、Grad-CAM、Chefer 等；
- 宽松版本（加区域摘要）与严格版本的对比；
- 其他 m 值的扫描（结果放附录热图）；
- 删除/插入实验、参数随机化检验；
- 在自监督的 DINO ViT-S/16 上重复图 2、图 3（与 DeiT-S 架构相同，只差训练方式）；
- Waterbirds 应用：用读取集合的背景比例判断模型是否在“看背景作弊”。

## 7. 核心贡献与结果预案

**7.1 三条贡献**

- **C1 一个读注意力的工具。** 不需要训练，就能把 ViT 每层每个 head 的注意力概括成“m 块区域、各读哪 k 个 token”的说明书。m = k = 16 时只有 256 个编号，原来是约 3.9 万个注意力数值。
- **C2 一种检验解释的方法。** 强制模型只按说明书计算，看预测是否改变。忠实性从“看图判断”变成了一个可以直接测量的数字。
- **C3 用这把尺子得到的发现。** 一层注意力需要多长的说明书（RQ1），哪些层可读（RQ2），说明书看到了什么（RQ3）。

C1、C2 是方法本身，不依赖实验结果的好坏；C3 的具体内容由实验决定。“分组 + top-k”的机制来自 Clustered Attention 和 MiTA，我们的新意在于把它变成按图像区域组织的解释，并用执行的方式检验它。

**论文贡献列表（英文）**

1. A training-free tool that summarizes each attention head of a pretrained ViT as m image regions, each reading k tokens.
2. An execution test that measures the faithfulness of the summary by forcing the model to compute only through it.
3. Using this ruler, we find how long a faithful summary needs to be, which layers are readable, and what the summaries look at.

**7.2 三种可能的结果，对应三种故事**

| 结果 | 论文的故事 | 主角是哪张图 |
| --- | --- | --- |
| A：k 很小时一致率就很高 | ViT 的注意力可以被很短的、人能读懂的说明书概括，并且我们能验证这一点 | 图 2（RQ1） |
| B：一致率普遍很低 | ViT 的注意力比 MiTA 图 4 暗示的更“个性化”：每个位置都有自己的读法，难以用少数共享模式概括；我们的检验方法第一次把这一点量化出来 | 图 2 与图 3 |
| C：有的层高、有的层低 | 注意力在某些层可读、在另一些层不可读，我们给出了它们的分布和可读层的内容 | 图 3（RQ2），再接图 4 |

三种情况下，C1 和 C2 都成立，C3 写成对应的发现。从 MiTA 图 4 的现象看，情况 C 最有可能出现。

## 8. 常见 QA

**Q1：和 MiTA 有什么不同？**

MiTA 是一种新的注意力结构，需要用它来训练模型，目的是加速。我们不训练任何模型，而是借用它的机制，去读一个已经训练好的普通 ViT。

**Q2：读取模式会改动原模型吗？**

不会。读取模式只是在正常的前向计算中把每层的 Q、K 取出来，额外算出说明书，模型的输出不受影响。

**Q3：执行模式把注意力换掉了，测的还是原模型吗？**

说明书是从原模型读出来的；执行模式是对它的一次检验：如果模型只看说明书里的 token，仍然给出和原来相同的答案，就说明说明书包含了模型需要的信息。只替换一层时（RQ2），模型的其余部分完全不变。这和 NLP 中检验解释“充分性”的思路相同。

**Q4：为什么看一致率，而不是准确率？**

我们要解释的是模型本身，而不是判断它对不对。模型原来就分错的图片，说明书也应该让它“同样地分错”。所以比较的是替换前后的预测是否相同。

**Q5：为什么用平均池化来求 landmark？**

两个原因。第一，按图像区域取平均，每个 landmark 就对应一块看得见的区域，说明书才读得懂。第二，5.1 节证明了 landmark 的打分正好是区域内各 query 打分的平均，含义清楚。MiTA 的消融实验中，2D 平均池化的效果也优于可学习参数和卷积等取法。

**Q6：选中次数图和 CLS 注意力图有什么区别？**

CLS 注意力图只是一个 query（CLS）那一行的权重；选中次数图汇总了一层中所有区域在读什么，并且来自一份经过执行检验的说明书。

**Q7：深层的 token 还对应原来图片上的位置吗？**

不完全对应。经过多层注意力后，每个 token 都混入了其他位置的信息（Abnar & Zuidema 2020 讨论过这个问题）。我们把读取集合理解为“这一层中被读取的 token 位置”，这是注意力可视化的通常做法，论文中会在局限性里说明这一点。

## 9. 论文 4 页版面规划

课程要求 NeurIPS 模板、正文最多 4 页（含图表），参考文献和附录不计页数，但附录不一定会被阅读。所以三张主图和核心公式都必须放进正文。

| 页 | 章节 | 内容 | 图表 |
| --- | --- | --- | --- |
| 1 | 摘要、1 引言 | 五句话故事展开成三段，末尾列三条贡献 | 图 1：方法示意（原注意力 → 读取模式 → 说明书 → 执行检验） |
| 2 | 2 相关工作、3 方法 | 三类相关工作各一段；注意力即快权重 MLP（一个公式）；读取模式（landmark、读取集合、分配三个公式）；执行模式和一致率 |  |
| 3 | 4 实验：设置、RQ1、RQ2 | 共同设置一段；RQ1、RQ2 各一段结论 | 图 2、图 3 |
| 4 | 4 实验：RQ3；5 讨论与局限；6 结论 | RQ3 一段；局限（深层 token 的信息混合、只测了一个模型）；结论回到引言中的空白 | 图 4 |

**附录**：其他 m 值的结果热图、实现细节、有时间做的选做实验。

**如果只完成了 RQ1**：第 3 页放大图 2，并加入不同 k 下一致率的分析；第 4 页放读取模式的可视化例子和讨论。论文依然是“问题 → 方法 → 检验 → 结论”的完整结构。

## 附录

### 附录 A：模型

| 用途 | 模型 | timm 名称 | 说明 |
| --- | --- | --- | --- |
| 核心实验 | [DeiT-S/16](https://huggingface.co/timm/deit_small_patch16_224.fb_in1k) | `deit_small_patch16_224.fb_in1k` | ImageNet-1k 有监督训练；22.1M 参数，4.6 GMACs，224×224 输入，197 个 token；自带 1000 类分类头 |
| 选做 | [DINO ViT-S/16](https://huggingface.co/timm/vit_small_patch16_224.dino) | `vit_small_patch16_224.dino` | 与 DeiT-S 架构相同，自监督训练；timm 版没有分类头，可用 CLS 特征余弦相似度，或加载 [DINO 官方线性头](https://github.com/facebookresearch/dino)（线性评估 77.0%） |
| 选做 | [DINOv2 ViT-S/14 + 4 个寄存器](https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m) | `vit_small_patch14_reg4_dinov2.lvd142m` | 用于检查背景伪影 token；默认 518×518 输入，可改为 224 |

以上模型的 Hugging Face 卡片均标注 Apache-2.0 许可。加载方式：`timm.create_model(name, pretrained=True)`。

### 附录 B：数据集

| 用途 | 数据集 | 规模与标注 | 获取方式 |
| --- | --- | --- | --- |
| 调试 | [Imagenette](https://github.com/fastai/imagenette) | ImageNet 的 10 个类别；160px 版约 102 MiB | GitHub 直接下载；文件夹名是 WordNet ID，需映射到 ImageNet 类别编号 |
| RQ1、RQ2 | [ImageNet-1k 验证集](https://huggingface.co/datasets/ILSVRC/imagenet-1k) | 50,000 张、1000 类；本项目随机抽 5,000 张 | Hugging Face 门控数据集，登录并同意非商业研究与教育用途条款 |
| RQ3 | [Oxford-IIIT Pet](https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html) | 37 类猫狗，带分割 trimap（前景、背景、边界） | `torchvision.datasets.OxfordIIITPet(split="test", target_types="segmentation", download=True)` |
| 选做 | Waterbirds | 训练/验证/测试 4,795 / 1,199 / 5,794 张 | 斯坦福提供的 tar 包；需要微调模型 |

### 附录 C：评估指标与参照方法的来源

| 内容 | 来源 / 实现 |
| --- | --- |
| 预测一致率 | 自写；思路对应 [ERASER](https://arxiv.org/abs/1911.03429) 的“充分性” |
| 随机读取集合、K-means 分组 | 自写：只改变读取集合或分组方式，其余代码共用；K-means 对每个 head 的 196 个 patch query 做，迭代 10 次即可 |
| CLS 注意力热图 | 用前向 hook 取最后一层 CLS 那一行，对 6 个 head 求平均 |
| 中心高斯图 | 几行代码：以图片中心为均值的二维高斯 |
| pointing game | 定义见 [RISE](https://arxiv.org/abs/1806.07421)（Petsiuk 等 2018） |
| 选做的热图参照 | [Attention Rollout](https://arxiv.org/abs/2005.00928) 自写；Grad-CAM 用 [pytorch-grad-cam](https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html)；Chefer 等用其 [官方代码](https://github.com/hila-chefer/Transformer-Explainability) |

### 附录 D：算力与工程要点

**算力**：核心实验约 30 万次 DeiT-S 前向（RQ1 约 18 万次，RQ2 约 12 万次，RQ3 约 1 万次）。DeiT-S 每张图 4.6 GMACs，学校 GPU 集群上不到一小时。N = 197 很小，直接在完整注意力矩阵上加掩码即可，不需要 flash-attn。

**接入 timm**：timm 的 Attention 模块默认可能调用融合的 `scaled_dot_product_attention`，拿不到中间的 Q、K。需要替换该模块的 forward，或关闭融合路径，以所装 timm 版本的源码为准。注意 timm 中缩放因子乘在哪里。

**一层的核心逻辑**（示意，需对照 timm 实现接入）：

```python
import torch, torch.nn.functional as F

def probe_and_execute(q, k, v, m_side, k_top):
    # q, k, v: [B, H, N, d]；token 0 是 CLS，其余是 g×g 个 patch
    B, H, N, d = q.shape
    g = int((N - 1) ** 0.5)
    # landmark：对 patch query 做 2D 平均池化
    qg = q[:, :, 1:].reshape(B * H, g, g, d).permute(0, 3, 1, 2)
    lm = F.adaptive_avg_pool2d(qg, m_side).flatten(2).transpose(1, 2)
    lm = lm.reshape(B, H, -1, d)                          # [B, H, m, d]
    # 读取集合：每个 landmark 的 top-k token（这就是说明书）
    s = lm @ k.transpose(-1, -2) / d ** 0.5               # [B, H, m, N]
    read = s.topk(k_top, dim=-1).indices                  # [B, H, m, k]
    # 分配：每个 query 分给最相似的 landmark
    route = (q @ lm.transpose(-1, -2)).argmax(-1)         # [B, H, N]
    # 执行（严格版）：query 只能读所分配的读取集合
    allowed = torch.zeros(B, H, lm.shape[2], N, dtype=torch.bool, device=q.device)
    allowed.scatter_(-1, read, True)
    mask = torch.gather(allowed, 2, route.unsqueeze(-1).expand(-1, -1, -1, N))
    logits = (q @ k.transpose(-1, -2)) / d ** 0.5
    out = logits.masked_fill(~mask, float('-inf')).softmax(-1) @ v
    return out, read, route
```

**几个容易出错的地方**：

- k = 197 时执行结果必须与原模型逐位一致，先通过这个单元测试再做其他实验；
- CLS 不参与区域池化，但要作为 query 被分配，也可以作为 key 被选中；
- 每个 head 单独计算说明书，汇总时再求和，不要一开始就把 head 平均掉；
- 每组实验的结果写进 CSV（模型、层、m、k、方法、随机种子、一致率），画图脚本只读 CSV。

### 附录 E：参考文献

- [MiTA：Mixture-of-Top-k Attention（2026）](https://arxiv.org/abs/2602.01219)，[代码](https://github.com/QishuaiWen/MiTA)
- [Fast Transformers with Clustered Attention（Vyas 等，NeurIPS 2020）](https://arxiv.org/abs/2007.04825)
- [Quantifying Attention Flow in Transformers（Abnar & Zuidema，ACL 2020）](https://arxiv.org/abs/2005.00928)
- [Transformer Interpretability Beyond Attention Visualization（Chefer 等，CVPR 2021）](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)
- [Attention is not not Explanation（Wiegreffe & Pinter，EMNLP 2019）](https://arxiv.org/abs/1908.04626)；Jain & Wallace，Attention is not Explanation（NAACL 2019）
- [ERASER（DeYoung 等，ACL 2020）](https://arxiv.org/abs/1911.03429)
- [Dissecting Recall of Factual Associations（Geva 等，2023）](https://arxiv.org/abs/2304.14767)
- [RISE（Petsiuk 等，BMVC 2018）](https://arxiv.org/abs/1806.07421)
- [Emerging Properties in Self-Supervised Vision Transformers（DINO，2021）](https://arxiv.org/abs/2104.14294)
- [Vision Transformers Need Registers（Darcet 等，ICLR 2024）](https://arxiv.org/abs/2309.16588)
- [Training data-efficient image transformers（DeiT，ICML 2021）](https://arxiv.org/abs/2012.12877)
