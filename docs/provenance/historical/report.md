# 第二阶段汇报：整理成可开源的代码包（release/v1）

**一句话总结：** `release/v1` 已整理完：只保留论文用到的实验，一个论文要素对应一个配置、一条命令。三层验证全部通过：从已保存的输出重建的表格逐位一致；全新克隆从头重跑后，31 张表全部在容差内（绝大多数逐位相同），所有判定相同；新环境的测试和冒烟测试通过。重跑发现了我整理时引入的 3 个问题，都已修正。没有推送。

- 分支：`release/v1`，从 `study/v5`（81bad1c）拉出，保留完整历史；没有删除任何分支，没有改写历史。
- 提交按步骤分开：追溯表 → 改包名（机械替换）→ 结构调整与精简 → D2 与 reproduce/ → 文档 → 验证结果。

---

## 1. 保留和删除了什么

### 1.1 前后对比

| | study/v5（81bad1c） | release/v1 |
| --- | --- | --- |
| 受版本管理的文件 | 2,083 | 154 |
| `src/`（.py） | 57 个文件，6,227 行 | 42 个文件，4,080 行 |
| `scripts/` | 38 个文件，4,539 行 | 6 个文件，1,376 行 |
| `tests/` | 14 个文件，2,242 行 | 10 个文件，1,428 行 |
| `configs/` | 57 个，638 行（多层 `base:` 继承） | 20 个，925 行（每个都已展开，自成一体） |
| 运行文件夹 `experiments/` | 1,735 个文件（约 1 GB） | 不进入；汇总表在 `results/`（372 KB） |

### 1.2 删除（仍保留在 `study/v5`）

- 目录：`slides/`、`paper/`、`notebooks/`、`colab/`（只新写 `reproduce/colab.ipynb`）、`references/`、`docs/` 下除新文档以外的全部、`experiments/`、旧图、`AGENTS.md`。
- 实验：匹配删除、中心对照、解释方法全面比较（Grad-CAM、IG、Rollout、CLS 注意力、SaCo）、物体移位、E3 小探针、gallery 独立实验、模型准确率检查、注意力预算的 budget / sink_use / register formal 阶段、anatomy 的 noise / repro / MiTA / Pet 阶段、parts 的 map 部分、human_alignment 的运行部分。
- 库代码：`sinks.py`、`results.py`、Pet 和 Hard ImageNet 数据、DeiT-T / DeiT-III-S / MiTA-DeiT-T 模型、执行模式 L1 / L2 / L4 / randC 与 k-means 地标、coverage / union 统计、`budget.py` 中的编辑与分类工具。

### 1.3 保留并重新组织

`src/fastweight_vit/`：`readings`（R0、C1–C3、组合、替身 `stand_in_*`、D2）、`references`（GAE、centre、random、完整注意力、top-k 图）、`probes`（线性与 MLP 位置探针）、`execution`（top-k 读取名单与执行）、`instrument`（变体模型的 R0）、`registers`、`metrics`（KL、Spearman、ClickMe）、`enrichment`、`faithfulness`（SRG）、`tokens`（高范数、距离环）、`models`、`data`（含 `build.py`）、`experiments/`（9 个实验）。

### 1.4 用新代码重新实现、需要重跑的旧结果

| 论文要素 | 原来 | 现在 |
| --- | --- | --- |
| AB1 单层限制 / 删除 | anatomy rq1（17 种设定）+ T7 | `execution` 的 `single_layer`（只算 L3）和 `attention_off` |
| AB2 头部 vs 身体 | parts（map + reliance） | `part_reliance`（只算 reliance） |
| AB3 top-k 图相似度 | explanations（12 种方法）+ 事后分析 | `map_similarity` |
| 图 3(c)、3(d) | layer_reading、human_alignment（旧流程） | `diagnose` 的 `high_norm`；`correct` 的 R0_l1–R0_l12（`clickme_layers`） |
| 值向量比例 0.29 / 0.23 | 预算研究 T3 | `diagnose` 的 `high_norm`（`value_norm.yaml` 用原设置） |
| register 的 8 个神经元 | T2 搜索 + gate.json | `register_search`（最小代码）；索引写进 `configs/table1_register.yaml` |
| 复现检查（AC3） | 新旧两套代码比较 | 只保留结果 `results/history/reproduction_check.csv`，在 `docs/preregistration.md` 说明 |

### 1.5 待定表

| 事项 | 现在的做法 | 需要作者决定吗 |
| --- | --- | --- |
| 逐图输出约 0.5 GB，超过 100 MB | 不进 git，由 `reproduce/full.sh` 重新生成；`results/` 只放汇总表 | 是否另外托管（默认不托管） |
| MiTA-DeiT-S 权重 | 不是本仓库训练的；`scripts/download_models.py` 用 gdown 从 MiTA 作者的 Google Drive 文件夹下载，SHA-256 校验 | 开源时放在哪里（默认：继续指向作者的文件夹） |
| 图 3(c)、3(d) 的来源 | 改用 R0 逐层读法（与旧的注意力质量只差正的常数倍） | 无 |
| `CITATION.cff` 的作者 | 暂写 "The fastweight-vit authors" | 定稿后填真实作者 |
| README 摘要 | 按 04 号清单第 1 节和第一阶段结果写；第 4.5 节用作者给的写法 | 作者定稿后同步 |

---

## 2. 最终目录结构

```
README.md  LICENSE  CITATION.cff  pyproject.toml  requirements.txt  requirements-torch.txt
src/fastweight_vit/     readings, references, probes, execution/, instrument, registers, metrics, enrichment,
                        faithfulness, tokens, models/, data/, experiments/
configs/                20 个，一个论文要素一个
scripts/                run.py, make_tables.py, make_figures.py, prepare_data.py, download_models.py, check_environment.py
reproduce/              from_outputs.sh, full.sh, smoke.sh, colab.ipynb
data/splits/            imagenet_val_5000.json, imagenet_s_srg_2000.json, heldout_4000.json（只有 ID）
results/                全部汇总表（含 c2_analysis/、supplementary/、d2/、history/）
figures/                图 2(a)、2(b)、3，附录图 A_2b_layer12、A_c2
docs/                   preregistration.md, reproduce.md, release/（本汇报、追溯表、第一层验证脚本）
tests/                  10 个文件
```

---

## 3. 第 8 节需要作者决定的事项（先用默认值）

| 事项 | 默认值 | 现状 |
| --- | --- | --- |
| 许可证 | MIT | 已加 `LICENSE` |
| 包名 | `fastweight_vit` | 已改 |
| 逐图输出超过 100 MB 是否另外托管 | 不托管，由脚本重新生成 | 约 0.5 GB |
| MiTA-DeiT-S 权重放在哪里 | 待定 | 现在指向 MiTA 作者的 Google Drive 文件夹 |
| 是否公开仓库、推送到哪里 | 不推送 | 没有推送 |

---

## 4. 三层验证

### 4.1 第一层：从已保存的输出重建

做法：把 `study/v5` 的运行输出按新布局导入（`docs/release/import_study_outputs.py`，只改名字，不改数值），用 `make_tables.py`、`make_figures.py` 重建，再与 `study/v5` 的表逐行比较（`docs/release/compare_layer1.py`）。

| 结果 | 表数 |
| --- | --- |
| 行完全对应、最大绝对差 0 | 28 张（图 2(a)、P1、图 3、表 1 长表与 markdown、全部修正检验、最后一层、两种顺序、改模型、健全性、预测对账、top-k 相似度、探针正确率、C2 分析 6 张、补充分析 10 张） |
| 均值一致，区间最大差 0.0012 | 1 张：附录 B 单层表。原因：旧表来自更早的分析脚本，bootstrap 时图片的行顺序不同；同一批每图数值，均值完全相同 |

图：图 2(a)、2(b)、附录 2(b)、图 3 逐像素相同；C2 附录图数据相同、目视一致（字体渲染差异）。在全新克隆、全新虚拟环境中运行 `reproduce/from_outputs.sh`，结果与 `results/`、`figures/` 逐字节相同。

### 4.2 第二层：全新克隆、从头重跑

**一句话：** 31 张表全部在容差内，没有一个判定改变；26 张逐位相同（GPU 上确定性计算的结果完全复现）。

- 环境：2026-11-08，Colab 一块 NVIDIA L4。全新克隆 `release/v1`（代码 `9bbee6f`；图 3(d) 重跑时 `0d47c04`），全新虚拟环境，按 README 安装；数据用同一台机器上已下载、校验过的副本（`FASTWEIGHT_DATA_ROOT`），模型由 `download_models.py` 重新下载并校验 SHA-256。
- 用时：全部运行 11.5 GPU 小时（原来的运行合计约 18 小时，因为旧运行还算了论文不用的量）。
- 完整对比表：`docs/release/layer2.csv`（`docs/release/compare_layer2.py`）。

| 对象 | 容差 | 结果 |
| --- | --- | --- |
| 各读法、各指标的均值（表 1、全部检验、健全性、两种顺序、补充分析、D2 等） | 1e-4 | 逐位相同（差 0） |
| 重新实现的要素：图 3(c) / 3(d)、头部 vs 身体、top-k 相似度、值向量比例；P2 的 Spearman | 1e-4 | 最大差 1.2e-6 / 1.5e-7 / 7.1e-6 / 2.5e-5 / 1.5e-8；2.9e-10 |
| 判定（更高 / 相当 / 更低）、P1–P7、各分支 | 完全相同 | 完全相同（0 处不同） |
| 执行检验一致率（k 扫描、单层、删除注意力） | 0.1 个百分点 | 逐位相同 |
| 重新训练的探针正确率 | 0.5 个百分点 | 逐位相同 |
| 依赖探针的读法（C2、R_all、组合、D2 等） | 0.002 | 逐位相同 |
| 图 | 目视一致 | 数据相同；PNG 因 VM 系统字体不同有像素级差异，目视一致 |

**超出容差、已查明并修正的一处：** 第一次重跑时，图 3(d) 的 DeiT-S、DINO 最大差 3.0e-4。原因是我写新配置时把这一项的 batch 统一成 64，而原来的运行是 128 / 64 / 128；ClickMe 的“图片特异”对照在 batch 内与前一张图配对，batch 大小会改变配对。改回原 batch 后重跑这两个模型，差降到 1.5e-7。

### 4.3 第三层：新环境

| 检查 | 结果 |
| --- | --- |
| 全新克隆 + 全新虚拟环境（只装 `requirements-torch.txt`、`requirements.txt`），`pytest`（CPU） | 78 个全部通过 |
| GPU（L4）上全新环境的 `pytest` | 78 个全部通过 |
| `reproduce/smoke.sh`（16 张图，覆盖每种读法、每个指标和执行检验） | 通过，233 秒（要求 10 分钟内） |

---

## 5. 整理中发现的原代码问题（没有改动行为，由作者决定）

1. **“无注意力”运行的预处理与说明不符。** 原 T7 实验的说明写“每个模型用自己的预处理（与论文执行图一致）”，但配置里有 `preprocessing: uniform`，实际对 DeiT-S、DeiT-B 用了 0.875 裁剪，而 k 扫描用的是它们自己的 0.9 裁剪（DINO 两者相同）。影响：图 2(a) 的“无注意力”水平线和附录 B 的“删除一层注意力”一列。数值很小（全部层删除时一致率 0.1%–0.2%），但两条曲线不是同一预处理。整理后保留原设置以复现原数字，并在配置里写明。
2. **ClickMe 对照的文字说明写反了方向。** 代码用 `roll(1)`，把每张图与 batch 中**前一张**图的人类图配对；原说明写“下一张”。只改了说明文字。
3. **两处未使用的代码**（C2 分析里一个算出后没用的变量；diagnose 里一个没用的导入）。删除，不影响数值。

---

### 5.1 第二层重跑发现的、我整理时引入的问题（都已修正）

1. **`gdown` 安装缺依赖：** README 和 notebook 里 `gdown` 和本包一起用 `--no-deps` 安装，缺了 `requests`，MiTA-DeiT-S 权重下载失败。改为单独安装 `gdown`（`9bbee6f`）。
2. **出图脚本的后端：** Colab 设置了 `MPLBACKEND=module://matplotlib_inline...`，全新环境里没有这个模块，`make_figures.py` 失败（表已经生成）。改为始终用 Agg 后端写文件（`db0c515`）。
3. **图 3(d) 的 batch 大小：** 见 4.2 节（`0d47c04`）。
4. **使用上的一点：** `full.sh` 最后会重写受版本管理的 `results/`、`figures/`；`run.py` 要求源代码没有未提交的改动，所以之后再跑实验要先提交或还原这两个文件夹。已写进 README。

## 6. 与论文草稿不一致的地方

1. **DINO 在 k = 16 时保留的预测比例：** 结果文件是 0.6148，四舍五入为 61%；草稿第 57、149 行写 62%。
2. **ClickMe 指标的定义：** 草稿 4.1 节写“扣掉每张图的数据集平均”；代码是“本图人类图的相关 − 另一张图人类图的相关”（另一张 = batch 中前一张）。
3. **最后一层找回比例 31%–62%：** 只包括富集和 ClickMe；SRG 的找回比例是 0.078 [0.004, 0.151]（DeiT-S）和 0.290（DeiT-B）。
4. **表 1 图注的区间宽度：** 草稿写约 ±0.002 / ±0.003；有些格达到 ±0.003（富集）和 ±0.0047（ClickMe）。
5. **值向量比例 0.2–0.3 倍：** 第 9–12 层逐层是 0.214–0.387（DeiT-S）和 0.192–0.268（DeiT-B）；0.29 / 0.23 是第 9–12 层的平均。
6. 第一阶段 X8 列出的 4 处（表 1 第一行混用 centre 与 mean_map、“M1 在第 12 层最小”只对有监督模型成立、“past layer 6”只对 DeiT-S 成立、ClickMe 定义）草稿里仍未改，由作者改。
