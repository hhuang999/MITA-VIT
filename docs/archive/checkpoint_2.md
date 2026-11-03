# 检查点 2：核心探针如何读取与执行

P1 已完成，代码在 `feature/core-probe` 分支。22 项测试通过；在 Imagenette 全部 3,925 张验证图片上，读取模式和 k=197 的全层执行均与原模型输出逐位一致。本阶段验证实现正确性，尚未测量小 k 时说明书的忠实性。

## 从四个 token 看懂两步计算

沿用研究设计 §5.2 的例子。这里 token 从 1 编号，省略除以 √d；代码索引从 0 开始。四个 query 是 q₁=(3,1)、q₂=(−1,1)、q₃=q₄=(1,−1)，分为 A={1,2}、B={3,4}，每组读取 k=2 个 token。

先取各区域的平均 query，得到 landmark（区域代表）：

| 区域 | landmark | 对四个 key 的打分 | 读取集合 |
| --- | --- | --- | --- |
| A | (1,1) | (1,2,4,−1) | {3,2} |
| B | (1,−1) | (1,−2,0,1) | {1,4} |

然后每个 query 与两个 landmark 做点积，选择得分较高的区域。q₁ 的两个分数为 4 和 2，所以分给 A；最终分配为 **A、A、B、B**。这次分配由 query 内容决定，不要求它回到自己的原始区域。

到这里，`read_patterns` 已产出说明书：每组读取哪些 token（`readsets`），以及每个 query 使用哪一组（`routes`）。这一步没有改变原模型的注意力。

执行时，`execution_mask` 把说明书转为允许读取的位置。q₁ 分到 A，只允许读 token 2、3。仍然使用 q₁ 自己对它们的分数 2、8，另外两个位置屏蔽，再做 softmax：

| q₁ 的注意力权重 | token 1 | token 2 | token 3 | token 4 |
| --- | --- | --- | --- | --- |
| 原注意力（约） | 0.0067 | 0.0025 | 0.9907 | 0.0001 |
| 严格执行（约） | 0 | 0.002473 | 0.997527 | 0 |

执行保留了“每个 query 自己打分”的过程，只限制它能读哪些 key。q₁ 原来就主要读取 token 3，因此这个例子的变化很小；其他图片和层是否也如此，要由 P2/P4 的一致率曲线回答。

## 接入真实 DeiT-S

真实模型每层有 197 个 token、6 个 head，共 12 层。每张图片、每层、每个 head 独立生成说明书。

1. `read_attention` 在 QKV 投影的输出处挂 hook。取出原模型的 Q/K，用 patch query 计算 16 个 landmark，再生成读取集合和路由。CLS 不参与平均，但要被分配，也能作为 key 被读取。
2. 离开读取上下文后，说明书保存在 `patterns` 中。它只包含整数索引，不保留计算图。
3. `execute_attention` 在选定层注入布尔掩码，交给 timm 现有 Attention 计算。模型参数和原有归一化、缩放、softmax、输出投影都保留。
4. 离开执行上下文后移除 hook，模型恢复正常前向。

```python
with torch.inference_mode():
    with read_attention(model, k_top=16) as patterns:
        original = model(images)
    with execute_attention(model, patterns):
        executed = model(images)
```

这里先从原模型读一次，执行时固定同一份说明书。执行过程中后层的 Q/K/V 会受到前层干预影响，但其允许读取的位置不重新选择。这样检验的始终是原模型刚才给出的说明书。

默认读取和限制所有层；`layers=(3,)` 只处理第 3 层，供之后 RQ2 使用。对外层号从 1 开始。

14×14 的网格不能等分成 4×4 个整数 patch 区域。按交接文档使用自适应平均池化时，单轴窗口是 `[0,4)`、`[3,7)`、`[7,11)`、`[10,14)`，相邻窗口有重叠。图上应画这些实际窗口，不画成互不重叠的四等分。

## 对照与指标

K-means 只改变 landmark 的获得方式：用网格 landmark 初始化，对 patch query 做 10 次欧氏距离聚类；得到中心后，读取集合和点积路由规则与网格版相同。遇到空簇保留上一轮中心。

随机对照只替换读取集合，在全部 197 个 token 中均匀无放回抽取 k 个，保留原路由。调用方维护带种子的独立 `torch.Generator`，跨 batch 连续抽样；不会消耗模型的全局随机状态。

主指标比较两个模型的预测类别是否相同，包括原模型分错的图片。辅助指标为 KL(p_original || p_executed)。bootstrap 把同一张图片在两个方法中的结果一起重采样，避免打乱配对关系；随机对照先对 5 个种子求平均，种子间波动另行报告。

## 验收结果

| 检查 | 结果 |
| --- | --- |
| 设计书四 token 手算 | landmark、集合、路由、执行权重均符合手算 |
| landmark 平均性质 | 逐区域、逐 head 检验通过，绝对误差容限 1e−4 |
| k=197 同一数学路径恒等 | 张量输出误差为 0，覆盖 grid 与 K-means |
| k=1…197 | 每个 query 恰好允许 k 个 key，输出均有限，屏蔽位置权重为 0 |
| timm 接入 | CPU 融合/非融合路径、CUDA 控制组、单层作用域与 hook 清理均通过 |
| 全部测试 | 22 项通过，无跳过；见 `experiments/2026-10-24_p1_unit/results.xml` |
| 预训练模型读取模式 | 3,925/3,925 预测一致；最大 logit 误差 0 |
| 预训练模型 k=197 全层执行 | 3,925/3,925 预测一致；最大 logit 误差 0，KL=0 |
| 原模型准确率 | 3,151/3,925，80.28%；执行后相同 |

准确率按完整 ImageNet 1,000 类输出计算，使用 timm 默认 eval 预处理和 Imagenette 160px 图片。10 个 WordNet ID 与 timm 的 ImageNet 类别表逐项核对一致。常见错误包括 church→monastery、cassette player→tape player；没有把候选类别限制在 Imagenette 的 10 类。这个结果不能直接当作完整 ImageNet 验证集的准确率。

真实数据验收耗时约 39.0 秒，不含模型加载；代码提交为 `2b17f53`。配置、日志、逐图预测、抽样顺序和运行元数据保存在 `experiments/2026-10-24_p1_validation/`。

ImageNet 的 5,000 张恒等验收仍待数据授权。当前结果已覆盖 P1 本地验收，但不替代那项正式检查。

## 代码与复现入口

- `src/mita_probe/probe.py`：说明书与执行掩码。
- `src/mita_probe/grouping.py`：网格平均与 K-means。
- `src/mita_probe/models.py`：读取、全层/单层执行与恢复。
- `src/mita_probe/baselines.py`、`metrics.py`、`data.py`：随机对照、指标和标签映射。

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_p1.py --config configs/validate_p1.yaml
```

复跑验收前，将配置复制一份并修改 `output` 为新的运行目录，提交后再运行。脚本要求已提交的代码，且不覆盖已有验收记录。

本次停在检查点 2。确认理解后再进入 P2：Imagenette 约 500 张，比较 k={4,8,16,32,64} 下的网格读取、随机读取和 K-means 曲线。
