# What Do Vision Transformers Read?

**Executable attention reading sets under centre-bias control** — DSAA 2012 Deep Learning 课程项目（HKUST(GZ)，2026 秋）。

**一句话：** 注意力可以看成一个"快权重 MLP"，每个图块（token）是一个神经元。我们按 MiTA 的规则，从**未改动**的预训练 ViT 中读出每块区域读取的 16 个神经元，并让模型只按这些名单计算，用模型自己的输出检验读取是否真实；再用扣除中心偏差的检验，分析 ViT 逐层读了什么。

## 交付物

| 内容 | 文件 |
| --- | --- |
| 论文初稿（NeurIPS 格式，正文 4 页 + 附录） | [`paper/main.pdf`](paper/main.pdf)，源文件 [`paper/main.tex`](paper/main.tex) |
| 报告幻灯片初稿（13 页，含英文讲稿） | [`slides/mita_probe_talk.pptx`](slides/mita_probe_talk.pptx) |
| 研究笔记本（方法演示、指标、全部核心结果、预测对账） | [`notebooks/mita_probe_research.ipynb`](notebooks/mita_probe_research.ipynb) |
| 完整结果报告 | [v3 报告](docs/analysis/v3_report.md)、[补充实验 S1–S3](docs/analysis/supplement_s1_s3.md)、[注意力预算研究](docs/budget/report.md) |

## 主要结果

在 DeiT-S、DINO ViT-S、DeiT-B 三个模型和 ImageNet-S、Hard ImageNet、Pet、ClickMe、PartImageNet 五个数据集上：

- **读取名单可以执行**：12 层同时替换（k = 16）时 DeiT-S 保留 72% 的预测，随机名单只有 9%，完全没有注意力只剩 0.1%。只替换一层不能作为证据：直接删掉一层注意力也保留 93%–97%。
- **不只是看中间**：所有注意力读取在扣除中心偏差后仍偏向物体；物体移到角落时跟着走；与人类重要性图的一致超出中心偏差；模型确实依赖物体本身，也更依赖动物的头（删头部的伤害是删身体的 2–3.4 倍）。
- **逐层**：最像人的读取在中后层；有监督模型的最后一层离开物体，DINO 不明显。这一规律在 top-k 探针、完整注意力和 CLS 注意力上都出现。有监督模型中后层的读取大量落在极少数高范数 token 上（DINO 没有），但它们不能解释最后一层的下降。
- **注意力预算**（补充研究，7 个模型）：DeiT-B 和两个 MiTA 模型深层 30%–50% 的注意力落在约 3% 的高范数 token 上。它们带回的内容像一个常数。训练时稀疏化（top-k）和测试时额外 token 都没有把注意力引向物体：腾出的注意力转到新的停靠点。最后一层离开物体也不是它们造成的。
- **如实的负面结果**：作为热图，所有注意力读取的忠实性都明显低于用梯度的 GAE；top-k 截断也没有让解释图比完整注意力更好。

所有预测和判定标准都在运行前登记（[决策日志](docs/decisions.md)）：32 条有方向的预测中，20 条成立，7 条部分成立，5 条不成立。

## 仓库结构

```
paper/        论文：main.tex、references.bib；tables/ 和 figures/ 由 scripts/paper_assets.py 生成
slides/       幻灯片：mita_probe_talk.pptx；slide_data.py 取数，build_deck.js 生成
notebooks/    研究笔记本
src/mita_probe/
  models/       模型注册表（deit, deit_b, dino, mita, deit_t, mita_t, deit3_s），权重版本和校验值固定
  probe/        探针：读取名单（landmark、top-k、路由）、按名单执行注意力
  data/         ImageNet、ImageNet-S、Hard ImageNet、Pet、ClickMe、PartImageNet
  experiments/  每个实验一个模块（由 scripts/run.py 调用）
  explain.py    所有重要性图：探针、完整注意力、CLS 注意力、Rollout、GAE、Grad-CAM、IG 和参照图
  faithfulness.py  删图块的忠实性检验（在所有层删除 token）
  plausibility.py  物体占比、AUROC、同距离环内富集等
configs/      每个实验的配置（预设参数改动须先登记）
scripts/      run.py（运行实验）、download_*.py、analyze_*.py（CPU 分析）、plot_*.py（作图）
colab/        Colab GPU 运行：local/ 上传代码和取回结果，remote/ 在实例上建环境、下数据、排队运行
  budget.py     注意力预算：逐层记录每个 token 收到的预算（MiTA 的压缩槽分回 token）、"不读/置零/平均值"三种处理
  registers.py  制造高范数 token 的神经元搜索、测试时额外 token
tests/        单元测试和端到端测试（假数据、随机权重；101 项）
experiments/  正式运行结果：每次运行一个文件夹，含配置、日志、结果表、git commit（大表在被忽略的 raw/）
figures/      报告中的图（PNG 和 PDF）
docs/         决策日志、实验日志、进度、相关工作、分析报告；docs/archive/ 是早期设计和中间报告
```

### 结果文件夹

| 前缀 | 内容 |
| --- | --- |
| `2026-10-27_rq1_colab*` | RQ1：k 扫描（第一代设置） |
| `2026-10-28_v2_*`、`2026-10-26_rq2_imagenet` | RQ1/RQ2 解剖（DeiT-S、DINO、MiTA-DeiT-S）、Pet 合理性 |
| `2026-10-26_rq3_diagnosis`、`2026-10-28_center_control`、`2026-10-29_center_matched_deletion*` | Pet 上的高范数诊断、中心对照、中心配对删除 |
| `2026-10-29_v3_*` | v3：E3–E8、健全性检验、成本、定性图、DeiT-B 解剖 |
| `2026-10-31_s_*` | 补充实验 S1–S3 及其分析（`2026-10-31_s_analysis`） |
| `2026-11-03_ab_*` | 注意力预算研究：T0 模型核对、T1 预试验、T2 神经元搜索、T3 预算账、T4 用没用上、T6 额外 token、T7 全部删掉注意力，及分析（`analysis_pilot`、`analysis_formal`）；大的逐图表为 `.csv.gz` |

## 运行

环境（Python 3.11–3.12）：

```bash
python -m venv .venv
.venv/bin/pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e .
.venv/bin/python scripts/download_models.py
.venv/bin/python -m pytest
```

数据：`python scripts/download_data.py <imagenet|imagenet_s|hard_imagenet|pet|clickme|partimagenet>`。ImageNet 需要 `HF_TOKEN`（只从 `.env` 或环境变量读取）；其余数据下载后核对固定的 SHA-256 或 MD5。

所有实验用同一个入口；源代码有未提交改动时程序拒绝运行，结果中记录 git commit：

```bash
python scripts/run.py explanations --config configs/explanations.yaml --model deit --output experiments/<日期>_e4_imagenet_s_deit
python scripts/run.py layer_reading --config configs/layer_reading.yaml --model deit_b --output experiments/<日期>_layers_deit_b
```

分析和作图只读取结果文件（CPU）：

```bash
bash scripts/analyze_v3.sh experiments/2026-10-29_v3_     # v3 分析和图 8–14
python scripts/analyze_supplement.py && python scripts/plot_supplement.py   # 补充实验和图 15–17
python scripts/paper_assets.py                             # 论文的表和图
python scripts/analyze_budget.py formal --prefix experiments/2026-11-03_ab_ && python scripts/plot_budget.py experiments/2026-11-03_ab_analysis_formal   # 注意力预算和图 18–20
python slides/slide_data.py                                # 幻灯片的图表数字
```

Colab（GPU）：`colab/local/send_code.sh <会话> <分支> [基准提交]` 上传代码；在实例上依次运行 `colab/remote/setup.sh`、`data.sh`、`jobs.sh`（读取 `/content/jobs.txt` 排队运行）；`colab/local/fetch_runs.sh` 分块取回结果。

论文编译：`bash paper/get_style.sh`，然后 `cd paper && latexmk -pdf main.tex`。

## 文档

- [决策日志](docs/decisions.md)：每项设计、预测、判定标准和结果对账（B01–B26），均在运行前登记。
- [实验日志](docs/experiment_log.md)：每次运行，包括失败的。[进度](docs/progress.md)：当前状态和待办。
- [相关工作](docs/related_work.md)及[调研细节](docs/research/)；[实验计划 v3](docs/experiment_plan_v3.md)。
- 注意力预算研究：[研究设计](docs/budget/research_design.md)、[工程设计](docs/budget/engineering_design.md)、[预试验报告](docs/budget/pilot_report.md)、[正式报告](docs/budget/report.md)（决策 B27–B32）。
- 分析报告：[v3 完整报告](docs/analysis/v3_report.md)、[补充实验 S1–S3](docs/analysis/supplement_s1_s3.md)、[RQ1/RQ2 解剖](docs/analysis/probe_report.md)、[V2 解剖细节](docs/analysis/v2_anatomy.md)、[高范数诊断](docs/analysis/rq3_diagnosis.md)。
- [归档](docs/archive/README.md)：早期设计、交接文档和中间报告；移出主线的运行在分支 `archive/pre-cleanup-2026-11-03`。
- [代理规范](AGENTS.md)。
