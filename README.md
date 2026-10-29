# MITA-VIT：把 MiTA 的 top-k 读取用作 ViT 的解释方法

**一句话：** 注意力可以看成一个"快权重 MLP"，图片的每个图块（token）就是它的一个隐藏神经元；我们借用 MiTA 的做法，只让每层读取少数几个神经元，再检验这些神经元能否复现模型的判断（忠实），以及它们是否落在人认为重要的物体和部件上（常识）。

研究主线、研究问题和每项实验的判定标准见 [决策日志](docs/decisions.md)（B01–B16）和 [实验计划 v3](docs/experiment_plan_v3.md)；相关工作见 [综述](docs/related_work.md)。旧版说明（第一、二代设计、HPC 记录）保存在 [docs/history/](docs/history/README_2026-10-27.md)。

## 研究问题

| RQ | 问题 | 主要实验 |
| --- | --- | --- |
| RQ1 忠实 | 少数共享神经元能否复现一层注意力？ | E1 解剖（k、m 扫描） |
| RQ2 结构 | 各层、各模型如何使用神经元？ | E1、探针逐层图 |
| RQ3 常识 | 模型依赖的神经元是否就是物体和关键部件，而不只是画面中间？ | E2 Pet、E4 合理性、E5 移位裁剪、E6 中心配对删除、E7 人类一致性、E8 部件 |
| RQ4 对比 | 在同一套检验下，探针和现有解释方法相比如何？ | E4 忠实性、健全性检验、成本 |

前提检验 E3：深层的 token 是否仍代表原来的图块。

模型：DeiT-S/16、DINO ViT-S/16（官方线性头）、DeiT-B/16（更大模型）；MiTA-DeiT-S 作对照。

## 代码结构

```
src/mita_probe/
  models/         模型注册表（deit, deit_b, dino, mita），权重版本和校验值固定
  probe/          探针：读取名单（landmark、top-k、路由）、按名单执行注意力
  data/           ImageNet、ImageNet-S、Hard ImageNet、Pet、ClickMe、PartImageNet
  explain.py      所有重要性图：探针和 CLS 注意力、Rollout、GAE、Grad-CAM、IG，以及中心、随机、掩码参照
  faithfulness.py 删图块的忠实性检验（在每层注意力中屏蔽，等价于删 token）
  plausibility.py 物体占比、AUROC、同距离环内富集等合理性指标
  identity.py     token 身份检验
  interventions.py 删除与替换读取名单中的 token
  experiments/    每个实验一个模块（由 scripts/run.py 调用）
  metrics.py stats.py results.py plotting.py
scripts/          run.py（运行实验）、download_*.py、analyze_*.py（CPU 分析）、plot_*.py（作图）
configs/          每个实验的配置；预设参数改动须先登记
colab/            Colab 运行：local/ 上传代码，remote/ 在实例上建环境、下数据、排队运行
tests/            单元测试和端到端测试（假数据、随机权重）
experiments/      每次运行一个文件夹：配置、日志、CSV、git commit
figures/          论文图（PNG 和 PDF）
docs/             决策、进度、实验日志、分析报告、调研
```

## 运行

所有实验用同一个入口，源代码有未提交改动时拒绝运行（结果里记录 git commit）：

```bash
python scripts/run.py explanations --config configs/explanations.yaml --model deit --output experiments/<日期>_e4_imagenet_s_deit
python scripts/run.py human_alignment --config configs/human_alignment.yaml --model dino --output experiments/<日期>_e7_clickme_dino
```

| 实验名 | 内容 | 主要输出 |
| --- | --- | --- |
| `anatomy` | RQ1/RQ2 解剖（需要 `--stage`） | predictions.csv、settings.csv |
| `explanations` | E4：所有方法的合理性和忠实性 | plausibility.csv、faithfulness.csv、saco.csv、cost.csv |
| `shifted_crops` | E5：同一物体放在中间和四角 | crops.csv |
| `matched_deletion` | E6：删除物体 token 与中心配对对照 | deletion.csv |
| `identity` | E3：逐层位置分类器和自相似度 | identity.csv |
| `human_alignment` | E7：与 ClickMe 人类重要性图的 Spearman 相关 | alignment.csv |
| `parts` | E8：各部件的富集，以及删掉某部件对模型的影响 | parts.csv、reliance.csv |
| `sanity` | 逐级重新初始化权重的健全性检验 | sanity.csv |
| `gallery` | 定性图用的随机样例 | gallery_<数据集>.npz |
| `center_control` | 探针读取名单对比"只读中间"（CPU） | — |

常用参数：`--smoke N` 只跑前 N 张；`--resume` 接着中断的运行继续；`--batch-size`、`--workers` 覆盖配置。

分析和作图只读结果文件：

```bash
bash scripts/analyze_v3.sh experiments/2026-10-29_v3_   # 全部 v3 分析和图
python scripts/analyze_explanations.py <run>              # 单个运行
```

每个分析会在运行文件夹里写 `analysis_*.csv`；登记过的主比较在 `analysis_comparisons.csv`（配对 bootstrap 95% 区间）。

## 环境

本地 Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe -m pytest
```

数据：`python scripts/download_data.py <imagenet|imagenet_s|hard_imagenet|pet|clickme|partimagenet>`。ImageNet 需要 `HF_TOKEN`（只从 `.env` 或环境变量读取）；其余数据在下载后核对固定的 SHA-256 或 MD5。默认缓存在 `checkpoints/`，数据在 `data/`（可用 `MITA_DATA_ROOT` 改位置）。

Colab（GPU）：`colab/local/send_code.sh <会话> <分支> [基准提交]` 上传代码；在实例上依次运行 `colab/remote/setup.sh`、`data.sh`、`jobs.sh`（读取 `/content/jobs.txt` 排队运行）。已建好环境后更新代码用 `update.sh`，不要重跑 `setup.sh`（会删除已下载的数据）。

## 文档

- [决策日志](docs/decisions.md)：每项设计、预测和判定标准（运行前登记）。
- [进度](docs/progress.md)、[实验日志](docs/experiment_log.md)（包括失败的运行）。
- [分析报告](docs/analysis/)：[探针综合报告](docs/analysis/probe_report.md)、[V2 解剖](docs/analysis/v2_anatomy.md)。
- [代理规范](AGENTS.md)。
