# 论文图表清单（paper/figures/）

所有图表都由 `make_paper_figures.py` 一个脚本生成：

```bash
python paper/figures/make_paper_figures.py --results <release/v1 的 results/ 文件夹>
```

- **格式和字号：** 每个图都有矢量 PDF 和 300 dpi PNG。按下表尺寸直接排进正文时，最小字号为 7 pt。
- **自动检查：** 有字小于 7 pt，或有内容超出画布，脚本就会停下来报错。
- **图里没有的东西：** 没有图号、面板字母、图注，也没有内部编号（R0、C1–C3、M1–M3 等）。名字一律用论文正文的名字。
- **模型颜色：** 所有数据图统一用 DeiT-S 蓝、DeiT-B 橙、DINO ViT-S 绿。示意图改用紫、深红、灰和 inferno 色阶，不和模型颜色冲突。

## 图

| 文件 | 内容 | 尺寸（英寸） | 数据来源 |
| --- | --- | --- | --- |
| `schematic_fast_weight_network` | 示意图：一个注意力头就是每张图的两层网络 | 2.60 × 1.90 | 无数据 |
| `schematic_reading` | 示意图：received attention 把单元 j 的激活画在第 j 个 patch 上 | 1.45 × 1.90 | 无数据 |
| `schematic_premises` | 示意图（表格形式）：三个前提 → 怎样测量 → 修正方法 → 它必须胜过的替身 | 5.50 × 1.50 | 无数据 |
| `execution_k_sweep` | 每个区域读的单元数 k（4–128），与原模型的 top-1 一致率（%）。线型：实线 = top-k reading sets，虚线 = random sets of the same size，点线 = attention removed；三个模型 | 2.60 × 1.90 | `results/k_sweep.csv`（5,000 张 ImageNet 验证图） |
| `gallery_layers9_12` | 两张 ClickMe 图（索引 47、287，按种子随机抽取，不是挑选的）；行依次为 image、human（ClickMe）、received attention、contribution weighting、probe relocation；DeiT-S，第 9–12 层合并 | 2.80 × 1.90 | `data/gallery_clickme_deit.npz`，取自 `experiments/2026-11-05_correct_ablation_clickme_deit/gallery_*.npz`（逐图数组，`results/` 里没有） |
| `gallery_layer12` | 同上，只用第 12 层（附录） | 2.80 × 1.90 | 同上（`*_l12` 数组） |
| `premise_mean_map_share` | 逐层：share of variance explained by the mean map；阴影为 95% 区间 | 1.35 × 1.50 | `results/diagnose.csv`（panel M1，variant R0） |
| `premise_probe_accuracy` | 逐层：position-probe accuracy | 1.35 × 1.50 | `results/diagnose.csv`（panel M2，variant probe） |
| `premise_high_norm_share` | 逐层：share of received attention on high-norm tokens | 1.35 × 1.50 | `results/diagnose.csv`（panel M3，variant R0） |
| `received_attention_vs_clickme` | 逐层：received attention 与 ClickMe 的图片特异 Spearman | 1.35 × 1.50 | `results/diagnose.csv`（panel quality，variant R0） |
| `c2_analysis_distance_error` | 第 6–12 层：位置探针（实线）和 matched blur（虚线）到真实 patch 的期望距离（patch） | 2.60 × 1.90 | `results/c2_analysis/errors.csv`（metric distance） |
| `c2_analysis_far_error_share` | 第 6–12 层：误差质量中距离超过 3 个 patch 的比例；灰线为预注册阈值 0.5 | 2.60 × 1.90 | `results/c2_analysis/errors.csv`（metric far_share） |
| `c2_analysis_shift_by_distance` | 按到图像中心的距离（patch）：probe relocation（实线）和 matched blur（虚线）使得分份额变化多少；第 9–12 层，ImageNet-S | 2.60 × 1.90 | `results/c2_analysis/shift_bins.csv` |
| `c2_analysis_mean_prediction_deit_s` | DeiT-S 第 12 层：探针在全部图上的平均预测，以均匀分布为 1 | 1.80 × 1.30 | `data/mean_prediction_layer12.npz`，取自 `experiments/2026-11-07_explore_c2_deit/mean_prediction.npz` |
| `c2_analysis_mean_prediction_dino` | 同上，DINO ViT-S | 1.80 × 1.30 | 同上（`..._dino/`） |

## 表

| 文件 | 内容 | 数据来源 |
| --- | --- | --- |
| `table1.tex` | 正文表 1：每格为“enrichment / ClickMe”，第 9–12 层；列为 DeiT-S、DeiT-B、DINO ViT-S；每个修正行下面紧跟它的替身行（缩进） | `results/table1_long.csv`、`results/variants.csv` |
| `table1_full.tex` | 附录完整表 1：三个指标（SRG、enrichment、ClickMe），每格为均值和 95% 区间，包含全部参照和两种合用顺序（order A = 先 contribution weighting，再 probe relocation，再 mean-map subtraction，这是预注册的顺序；order B = 先 contribution weighting，再 mean-map subtraction，再 probe relocation） | `results/table1_long.csv`、`results/order.csv`、`results/variants.csv` |

两个 `.tex` 文件只含 `tabular`，需要 `booktabs`，图注和标签由论文自己加。`table1_full.tex` 自带 `\footnotesize` 和 `\tabcolsep`，这样才能放进 5.5 英寸正文宽，所以要放在 `table` 环境里使用。

## 和要求不一样的地方（原因）

1. **示例图改成横排。** `gallery_*` 每行是一种图，每列是一张图片，不是“列为 image / human / …”。原因：在 2.8 英寸宽、7 pt 字号下，五个列标题（例如 “contribution weighting”）放不进约 0.5 英寸宽的列。
2. **探针平均预测图没有做最小–最大归一化。** `c2_analysis_mean_prediction_*` 用的是原始比值（均匀分布 = 1）。原因：这张图要表达的就是“接近均匀”（峰值只有均匀值的 1.1–1.3 倍）；归一化以后，这个信息就看不出来了。示例图仍然按要求对每张图做最小–最大归一化，并配统一色标。
3. **示意图拆成三个文件。** `schematic_*` 按“每个面板一个文件”的规则拆开，同时去掉了面板字母和内部编号。

## 其他

- `data/` 里放的是两个小数组：示例图用的数组，以及探针平均预测。加 `--experiments experiments` 参数运行，会从研究的运行文件夹重新提取。
- 本文件夹里的 `execution.*` 和 `layers.*` 属于旧初稿（`paper/main.tex`），不是这套图表的一部分。
