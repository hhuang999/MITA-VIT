# MITA-VIT

把预训练 ViT 每层每个注意力头概括成“16 块区域各读取 k 个 token”，再通过限制读取集合检验预测是否保持。核心模型是 DeiT-S/16，研究问题依次为概括的忠实性（RQ1）、逐层差异（RQ2）和与物体前景的对应（RQ3）。

当前阶段为 P1 核心探针验收完成，停在检查点 2。读取模式、固定说明书的严格执行、网格/K-means 分组、随机对照及指标已实现；RQ1 曲线试跑在下一阶段进行。实现与验收讲解见 [检查点 2](docs/checkpoint_2.md)。

本地环境已通过验收，项目已同步到 [MITA-VIT 私有仓库](https://github.com/hhuang999/MITA-VIT)。默认分支为 `main`。

## 本地快速开始

在本目录打开终端，VS Code 解释器选择 `.venv/Scripts/python.exe`。环境为 Python 3.12.11、PyTorch 2.7.0 + CUDA 12.8、TorchVision 0.22.0、timm 1.0.30。Python 与邻近课程项目隔离。

从头重建时执行（`python` 须为 Python 3.11 或 3.12）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

CUDA 安装组合来自 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/#v270)。本机 NVIDIA 驱动为 610.62；不需要另装完整 CUDA Toolkit。`requirements.txt` 锁定其余依赖；torch/torchvision 单独安装以免 HPC 装错平台。

更新依赖锁定文件时使用 `uv pip compile pyproject.toml --extra dev --universal --excludes requirements-torch.txt --no-header --no-annotate -o requirements.txt`，然后重新验证环境。`requirements-torch.txt` 只用于声明单独安装的平台依赖；不把整个课程环境的包带入本项目。

```powershell
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe scripts/download_data.py imagenette
.\.venv\Scripts\python.exe scripts/download_data.py pet
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/check_environment.py
```

P0 单图推理记录位于 `experiments/2026-10-24_environment/environment.json`。P1 已在 Imagenette 全部 3,925 张验证图上检查：原模型准确率 80.28%，读取模式和 k=197 全层执行的输出均与原模型完全一致；记录位于 `experiments/2026-10-24_p1_validation/`。RQ1 调试运行待 P2 交付，计划使用约 500 张验证图片。

`.env` 中保留 `HF_TOKEN`，通过 `python-dotenv` 读取。默认缓存为 `checkpoints/huggingface/` 和 `checkpoints/torch/`；可用 `HF_HOME`、`TORCH_HOME` 显式指定其他磁盘。

## 资源

| 资源 | 本地路径 | 用途 |
| --- | --- | --- |
| DeiT-S/16 | `checkpoints/huggingface/` | 核心模型 |
| Imagenette 160px | `data/imagenette2-160/` | 调试；原模型输出是 ImageNet 1000 类，不能直接与 10 类编号比较 |
| Oxford-IIIT Pet | `data/oxford-iiit-pet/` | RQ3 测试集与 trimap |
| ImageNet 验证集 | 尚未下载 | 2026-10-25 已验证项目 token 可读取验证集文件；先查 HPC 共享副本 |
| 14 篇参考文献 | `references/papers/` | PDF 不进 Git；来源与校验和见 `references/manifest.json` |

文献可通过 `python scripts/download_references.py` 重建，BibTeX 见 `references/references.bib`。DINO、DINOv2、Waterbirds 属于选做项目，本阶段不下载。资源实际完成情况以 `docs/progress.md` 为准。

## 目录与文档

`src/mita_probe/` 为可编辑安装的包，`scripts/` 放命令入口，`configs/` 保存研究设置；`tests/` 放测试，`hpc/` 放远程环境脚本。`configs/validate_p1.yaml` 可用于实现验收；RQ1 入口已实现，RQ2/RQ3 随后续阶段实现。

P2 完整试跑使用 Imagenette 500 张，包含 35 组设置及 k=197 恒等检查。先提交代码，再指定新的输出目录；入口拒绝覆盖已有实验：

```powershell
.\.venv\Scripts\python.exe scripts/run_rq1.py --config configs/rq1_pilot.yaml --output experiments/2026-10-25_rq1_pilot
.\.venv\Scripts\python.exe scripts/make_figures.py experiments/2026-10-25_rq1_pilot
```

`results.csv` 每行对应一个方法/k/种子；`predictions.csv` 保留逐图结果；`summary.csv` 汇总方法均值及随机种子标准差；`contrasts.csv` 保存 grid 减随机对照的配对 bootstrap 95% 区间。图只读 CSV。区间按图片重采样，随机对照先对 5 个种子求平均，不把种子视为额外图片。

- [研究设计](docs/research_design.md)：方法公式与实验依据。
- [研究叙事](docs/research_narrative.md)：保留所有【待填】。
- [课程评分标准](docs/course/rubric.pdf)：报告、视频与截止时间。
- [进度](docs/progress.md)、[决策](docs/decisions.md)、[实验日志](docs/experiment_log.md)。
- [交接](HANDOFF.md)：完整上下文与检查点；[代理规范](AGENTS.md)。

实验输出放在 `experiments/`，保留小型配置、CSV、日志和 git commit；`raw/` 大文件不进 Git。最终图片放 `figures/`，正式实验分析放 `docs/analysis/`，论文放 `paper/`。数据、模型、`.env` 和 HPC 账号说明始终忽略。

## 在 HPC 上运行

先阅读本地 `docs/hpc/HPC.txt`。学校平台创建按需 JupyterLab 开发环境，镜像为 `admin/webide-jupyter:v6.2`，基础解释器路径为 `/opt/miniconda3/envs/pytorch/bin/python`。这不是已验证的 Slurm 集群配置；本次未创建远程环境、联网检查或提交作业。

首次使用 GPU 环境和运行作业须用户确认。建立环境后，通过平台当次提供的 SSH 命令或 Web 终端进入，在持久化项目目录 clone 私有仓库；账号和动态连接信息只放本地 HPC 说明。

```bash
git clone https://github.com/hhuang999/MITA-VIT.git
cd MITA-VIT
nvidia-smi
bash hpc/setup.sh
```

`setup.sh` 默认使用镜像解释器创建独立 `.venv`。它要求 Python 3.11/3.12；若镜像版本不同，先提供符合版本的 `PYTHON_BIN`。CUDA 默认 cu128，必须先核对实际 GPU 驱动；可通过 `TORCH_INDEX_URL` 选择 PyTorch 2.7.0 支持的 cu118/cu126/cu128。脚本尚未在 HPC 实测。

在 HPC 独立设置 `.env`，不要通过 Git 传密钥。将 `HF_HOME`、`TORCH_HOME` 指向项目 `checkpoints/` 或持久化数据盘，数据直接在 HPC 下载。ImageNet 账号授权已验证；先确认 HPC 是否有共享副本，没有则执行：

```bash
bash hpc/download_data.sh imagenet
```

该入口只下载 validation（50,000 张），保存 seed=0 的 5,000 张抽样索引，后续 RQ1/RQ2 共用。Hugging Face 的 parquet 分片与解码后的数据都会占空间，下载前核对存储余量。

`bash hpc/run.sh rq1 experiments/YYYY-MM-DD_rq1_formal` 使用正式配置（ImageNet 5,000 张）；RQ2/RQ3 入口尚未实现。原有连通测试保留为 `hpc/train_smoke.py`，未经用户确认不启动首次 HPC 作业。

代码用本地 push / HPC pull 同步；远程生成的结果小文件经检查后 commit / push，本地 pull。若平台不能访问 GitHub，按 HPC 说明使用 scp 或 JupyterLab 上传代码，数据不从本地上传。用完停止开发环境以释放资源。
