# MITA-VIT

把预训练 ViT 每层每个注意力头概括成“16 块区域各读取 k 个 token”，再通过限制读取集合检验预测是否保持。核心模型是 DeiT-S/16，研究问题依次为概括的忠实性（RQ1）、逐层差异（RQ2）和与物体前景的对应（RQ3）。

P2 的完整 RQ1 试跑已完成：Imagenette 500 张、35 组设置，grid 在 k=64 时一致率为 88.2%，随机对照为 51.88%，K-means 为 86.8%。用户已确认进入 P3 HPC 适配，首次 GPU 作业待审阅；见 [首次作业审阅](docs/hpc_first_job_review.md)。试跑曲线与核心实现讲解分别见 [检查点 3](docs/checkpoint_3.md)、[检查点 2](docs/checkpoint_2.md)。

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

P0 单图推理记录位于 `experiments/2026-10-24_environment/environment.json`。P1 已在 Imagenette 全部 3,925 张验证图上检查：原模型准确率 80.28%，读取模式和 k=197 全层执行的输出均与原模型完全一致；记录位于 `experiments/2026-10-24_p1_validation/`。P2 的 500 张完整试跑见 `experiments/2026-10-25_rq1_pilot/`。

`.env` 中保留 `HF_TOKEN`，通过 `python-dotenv` 读取。默认缓存为 `checkpoints/huggingface/` 和 `checkpoints/torch/`；可用 `HF_HOME`、`TORCH_HOME` 显式指定其他磁盘。

## 资源

| 资源 | 本地路径 | 用途 |
| --- | --- | --- |
| DeiT-S/16 | `checkpoints/huggingface/` | 核心模型 |
| Imagenette 160px | `data/imagenette2-160/` | 调试；原模型输出是 ImageNet 1000 类，不能直接与 10 类编号比较 |
| Oxford-IIIT Pet | `data/oxford-iiit-pet/` | RQ3 测试集与 trimap |
| ImageNet 验证集 | `data/imagenet_val/` | 50,000 张已下载并校验；固定 seed=0 的 5,000 张索引，覆盖 994 类 |
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

先阅读本地 `docs/hpc/HPC.txt`。学校平台创建按需 JupyterLab 开发环境，镜像为 `admin/webide-jupyter:v6.2`，集群 `AI-X86_NVIDIA`。已在 CPU 2 核环境通过 SSH 检查网络和存储；这不是 Slurm。镜像解释器 `/opt/miniconda3/envs/pytorch/bin/python` 实测为 Python 3.10.13 / torch 2.3.1，不能直接使用。

首次使用 GPU 环境和运行作业须用户确认。建立环境后，通过平台当次提供的 SSH 命令或 Web 终端进入，在持久化项目目录 clone 私有仓库；账号和动态连接信息只放本地 HPC 说明。

```bash
git clone https://github.com/hhuang999/MITA-VIT.git
cd MITA-VIT
git switch feature/hpc-adaptation
# Use a project-owned persistent directory and leave the platform environment intact.
/opt/miniconda3/bin/conda create --prefix /absolute/persistent/path/mita-python python=3.12 pip -y
PYTHON_BIN=/absolute/persistent/path/mita-python/bin/python bash hpc/setup.sh
```

`setup.sh` 要求显式提供 Python 3.11/3.12 的 `PYTHON_BIN`，在项目内创建 `.venv` 并安装固定依赖；不运行模型推理。CUDA 默认 cu128；GPU 环境中的 preflight 和测试负责检查实际驱动兼容性。若需切换 CUDA wheel，使用 `TORCH_INDEX_URL` 指定受支持的索引，并记录实际环境后重新验收，不更改 torch 2.7.0 与研究参数。

在 HPC 独立设置 `.env`，不要通过 Git 传密钥。将 `HF_HOME`、`TORCH_HOME` 指向项目 `checkpoints/` 或持久化数据盘，用 `MITA_DATA_ROOT` 指定数据盘目录。先确认 HPC 是否有与固定版本/顺序一致的共享副本，没有则直接在 HPC 下载：

```bash
bash hpc/download_data.sh imagenet
.venv/bin/python scripts/verify_imagenet.py --skip-inference --output experiments/YYYY-MM-DD_hpc_environment/data_verification.json
```

该入口只下载 validation（50,000 张），保存 seed=0 的 5,000 张抽样索引，后续 RQ1/RQ2 共用。Hugging Face 的 parquet 分片与解码后的数据都会占空间，下载前核对存储余量。

本地数据已于 2026-10-25 下载并验收。源分片约 6.69 GB，保存的数据集约 6.71 GB，另有缓存。数据版本见 `data/imagenet_manifest.json`，验收记录见 `experiments/2026-10-25_imagenet_download/verification.json`。`scripts/verify_imagenet.py --output <新的 JSON 路径>` 会核对源文件 SHA-256、标签顺序、固定子集解码及 32 张真实图片的模型前向。

数据和模型下载固定为 P2 已接受的 revision。数据已存在时下载入口拒绝覆盖，应直接运行验证命令。`--skip-inference` 仅验证分片哈希、标签和 5,000 张解码，不启动模型实验。

首次 GPU 作业使用 `bash hpc/first_job.sh experiments/YYYY-MM-DD_hpc_first`：离线 preflight → 完整 pytest → ImageNet 固定子集前 50 张 × 35 组 → 曲线。脚本和资源选择见 [审阅文档](docs/hpc_first_job_review.md)，用户确认后才运行。

`bash hpc/run.sh rq1 experiments/YYYY-MM-DD_rq1_formal` 使用正式配置（ImageNet 5,000 张）；首次脚本不自动调用它。RQ2/RQ3 入口尚未实现。原有连通测试保留为 `hpc/train_smoke.py`。

代码在私有 GitHub 分支保存。可通过本地 push / HPC pull 同步；本轮使用同一已提交分支的 Git bundle 经 SSH 传送，远程保留完整 Git 历史与 GitHub origin，不在计算节点保存 GitHub 凭据。远程结果小文件经检查后回传、提交并推送；数据不从本地上传。用完停止开发环境以释放资源。
