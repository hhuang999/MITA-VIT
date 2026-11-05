# v5 盘点：现有实验在新闭环中的位置

2026-11-05 · 依据：[研究调整说明](research_adjustment.md) 第 5 节、[研究骨架](skeleton.md) 第 3–4 节 · 决策 B33

**一句话：** 现有实验全部保留；按“读 → 诊断 → 修正”的闭环分成核心、扩展、只放附录、移出主线四类。新实验只补闭环缺的部分。

## 怎么整理的

- **不删除任何结果。** v5 开始前的完整状态保存在本地存档分支 `archive/pre-v5-2026-11-05`（提交 db46c53）。
- **“移出主线”的运行，文件暂时留在原处。** 原因是现有笔记本 `notebooks/mita_probe_research.ipynb` 还在读这些文件，现在移走会让它无法运行。等笔记本按 v5 重写、不再读这些文件后，再把它们从主线移除。存档分支里已经有完整副本。
- **每项都标出服务于哪张图表。** 对应不上任何图表的，不进论文。

## 1. 核心（正文主体）

| 实验 / 结果 | 运行文件夹 | 闭环位置 | 图表 |
| --- | --- | --- | --- |
| 执行检验，k = 16，12 层同时限制（读取名单 72/62/83%，随机名单 9/4/30%） | `2026-10-28_v2_rq2`（DeiT-S）、`2026-10-28_v2_dino_rq2`、`2026-10-29_v3_deit_b_rq2`；汇总 `2026-10-28_v2_analysis/rq2_anatomy_all_layers.csv`、`2026-10-28_v2_dino_analysis/rq2_main_budget.csv`、`2026-10-29_v3_deit_b_analysis/rq2_main_budget.csv` | 读（RQ1） | 正文 5.1；图 2(a) 中 k = 16 的点（新运行复现） |
| 完全没有注意力（0.1–0.2%） | `2026-11-03_ab_t7_{deit,dino,deit_b}` | 读（RQ1） | 图 2(a) 的虚线 |
| 主设置 k 扫描（DeiT-S） | `2026-10-28_v2_rq2`（L3_spatial 与 rand，k = 4–128，5,000 张） | 读（RQ1） | 图 2(a) 的复核（新运行 `read_ksweep` 为正式来源） |
| 中心对照与中心参照 | `2026-10-28_center_control`；E4、E7 中的 `centre` | 读；M1 的证据 | 表 1 参照行 |
| 完整注意力（R0）在 ImageNet-S、ClickMe 上的结果 | `2026-10-31_s_e4_imagenet_s_*`、`2026-10-31_s_e7_clickme_*`；`2026-10-31_s_analysis/` | 读；修正的基线 | 表 1 的 R0 行（新运行复现对比） |
| 逐层 ClickMe 图片特异 Spearman（S2） | `2026-10-31_s_e7_clickme_*`；`2026-10-31_s_analysis/s2_layers.csv` | 诊断 | 图 3(d) |
| GAE 参照 | `2026-10-29_v3_e4_imagenet_s_*`、`2026-10-29_v3_e7_clickme_*` | 修正的参照上限 | 表 1 参照行（新运行重算） |
| E3 位置探针 | `2026-10-29_v3_e3_identity_*`、`2026-10-29_v3_e3_identity.csv` | 诊断 M2 | 图 3(b)（新运行 `diagnose_probe` 复核） |
| S3 读取落在高范数 token 上的比例 | `2026-10-31_s_layers_*`；`2026-10-31_s_analysis/s3_layers.csv`（`attention` 一族即 R0） | 诊断 M3 | 图 3(c) |
| 值向量长度与均值替换（M3 的“激活 ≠ 贡献”） | `2026-11-03_ab_t3_*`（值向量长度比）、`2026-11-03_ab_t4_{deit,deit_b}`（均值替换）；`2026-11-03_ab_analysis_formal/` | 诊断 M3 | 正文 5.2 一句话；附录 A5 |
| 随机化健全性检验（v3） | `2026-10-29_v3_sanity_*` | 新读法也要做 | 附录（新运行 `correct_sanity`） |

## 2. 扩展（正文一句话，细节在附录）

| 扩展 | 运行文件夹 | 处理 | 图表 |
| --- | --- | --- | --- |
| A 每个区域只用很少的神经元 | 旧 k 扫描 `2026-10-27_rq1_colab*`（第一代设置）；S1 逐图相关 `2026-10-31_s_analysis/s1_similarity.csv` | 正文用新运行 `read_ksweep`；旧扫描只放附录并注明第一代设置 | 图 2(a)；附录 A1 |
| B 单层可替换，全部去掉就崩溃 | 单层限制：`2026-10-28_v2_analysis/rq1_anatomy.csv`、`2026-10-28_v2_dino_analysis/rq1_anatomy.csv`、`2026-10-29_v3_deit_b_analysis/rq1_anatomy.csv`；单层删注意力：`2026-11-03_ab_t7_*` | 两类运行用同一批 5,000 张图、同样的预处理；三个模型都有数字（见下方“数据核对”） | 附录 A2 |
| C 读的是头 | `2026-10-29_v3_e8_parts_*`、`2026-10-31_s_e8_parts_*` | 正文一句话；定性图中可选一例 | 附录 A3 |
| D 跟着物体走 | `2026-10-29_v3_e5_shifted_*` | 正文一句话；引用并区分 Kuang 等 | 附录 A4 |
| E 改模型 vs 改读法 | `2026-11-03_ab_t2_deit`（register 神经元）、`2026-11-03_ab_t6_deit`（R0–R5）、`2026-11-03_ab_t3_mita*`、`2026-11-03_ab_analysis_formal/`；`2026-10-28_v2_mita_*` | 表 1 加两行（新运行按表 1 协议补算）；附录 A5 | 表 1；附录 A5 |

**扩展 B 的数据核对。** 单层限制和单层删注意力用的都是固定的 5,000 张 ImageNet 子集（种子 0）和各模型自己的预处理，图片一致。调整说明里的概括数字只来自 DeiT-S，另外两个模型不同：

| 模型 | 单层限制为读取名单（第 1 层 / 其余层） | 单层删掉注意力（第 1 层 / 其余层） |
| --- | --- | --- |
| DeiT-S | 94.5% / 91.7–96.0% | 68.8% / 92.8–97.2% |
| DINO ViT-S | 76.7% / 80.9–94.9% | **0.2%** / 85.7–95.0% |
| DeiT-B | 97.1% / 95.1–96.6% | 87.8% / 93.7–97.8% |

所以，“每层单独可替换（第 1 层除外）”对 DeiT-S、DeiT-B 成立；DINO 第 1 层的注意力不能删（只剩 0.2%），但可以限制为读取名单（保留 76.7%）。DINO 的后几层也明显更难替换（第 12 层 80.9%）。这一项不需要重跑，只需要按模型分别报告。

## 3. 只放附录

| 实验 | 运行文件夹 | 理由 | 图表 |
| --- | --- | --- | --- |
| 与 Grad-CAM、IG、Rollout、CLS 注意力的完整比较 | `2026-10-29_v3_e4_imagenet_s_*`、`2026-10-29_v3_e7_clickme_*` | 不回答 RQ；正文只留 GAE | A8 |
| E6 距离匹配删除 | `2026-10-29_v3_e6_deletion_imagenet_s` | 第 1 步的补充证据 | A8 |
| 预注册对账总表 | `docs/decisions.md` | 体现严谨性 | A9 |
| 推理成本 | v3 E4 的 `cost.csv` | 一行带过 | 正文一句话 |
| 随机抽取的定性图（ImageNet-S、ClickMe） | `2026-10-29_v3_gallery_*`；新运行 `correct_ablation` 的 `gallery_*.npz` | 不挑图 | A10 |

## 4. 移出主线（论文不出现；文件暂留原处，见开头说明）

| 实验 | 运行文件夹 | 理由 |
| --- | --- | --- |
| Pet、Hard ImageNet、ImageNet 子集上的 E4 | `2026-10-29_v3_e4_{pet,hard_imagenet,imagenet}_*`、`2026-10-31_s_e4_{pet,hard_imagenet,imagenet}_*`、`2026-10-29_v3_e6_deletion_hard_imagenet` | 重复主数据集的结论 |
| top-k 名单作为解释图的各项比较（`probe_mass` 等） | v3 E4/E7/E8 中的 `probe_*` 方法；S1 比较 | 已由扩展 A 的一句话覆盖；B26 已改用完整注意力画图 |
| IG 未通过随机化检验 | `2026-10-29_v3_sanity_*` 中的 IG | 与快慢权重无关 |
| 噪声底、Pet 上的早期高范数诊断 | `2026-10-28_v2_noise`、`2026-10-28_v2_dino_noise`、`2026-10-26_rq3_diagnosis`、`2026-10-28_v2_pet*`、`2026-10-28_v2_dino_pet`、`2026-10-29_center_matched_deletion*` | 已被后续实验取代 |
| 其他数据集的定性图库 | `2026-10-29_v3_gallery_*` 中的 Hard ImageNet、PartImageNet 部分 | 只留 ImageNet-S、ClickMe |
| 第一代 k 扫描以外的旧设置（V2 解剖的 L1–L4 分解、m 扫描、消融） | `2026-10-28_v2_rq1`、`2026-10-28_v2_rq2` 中除 L3_spatial 与 rand 以外的设置 | 不进入 v5 的闭环 |
| 注意力预算研究中与 v5 无关的部分（H1 跨 7 个模型的比较、DeiT-T、DeiT-III-S、sink 分型） | `2026-11-03_ab_*` | 保留在 `docs/budget/report.md`，v5 只用 register 与 MiTA 两行和 M3 证据 |

## 5. 新实验（对应骨架第 4.1 节）

| 运行 | 回答什么 | 图表 |
| --- | --- | --- |
| `read_ksweep_{deit,dino,deit_b}` | 每个区域读多少个神经元就够 | 图 2(a) |
| `diagnose_probe_{model}` | 每层 token 现在代表哪里；C2 的探针和匹配的模糊宽度 | 图 3(b) 复核；C2 |
| `diagnose_m1_{model}` | 读取中有多少与图片无关；C1 的平均图 | 图 3(a)；C1 |
| `correct_ablation_{imagenet_s,clickme}_{model}` | 修正能改进多少（含对照、参照、第 12 层、1–4 层阴性对照） | 表 1；附录 A6；P4 |
| `correct_order_{imagenet_s,clickme}_{model}` | 修正顺序会不会改变结论 | 附录 A7 |
| `correct_sanity_{model}` | 新读法是否通过随机化健全性检验 | 附录 |
| `correct_{register,mita}_{imagenet_s,clickme}` | 改模型的两行，按表 1 的协议 | 表 1 改模型两行 |

## 6. 文档

| 文档 | 处理 |
| --- | --- |
| `docs/decisions.md` | 新增 B33（定位，替代 B21）；B34–B40 为 v5 的提议 |
| `docs/experiment_plan_v5.md` | 新写：v5 执行规格 |
| `docs/analysis/v3_report.md`、`docs/analysis/supplement_s1_s3.md` | 内容不变，开头加一行“已被新定位取代” |
| `docs/research_story.md`、`docs/budget/*` | 内容不变，开头注明 v5 的位置 |
| `README.md`、`docs/progress.md`、`AGENTS.md` | 更新一句话和主线说明 |
| `paper/`、`slides/`、`notebooks/` | 暂不改。论文正文由作者写；幻灯片和笔记本等结果出来后再定 |
