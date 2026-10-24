# HANDOFF：MiTA 启发的 ViT 注意力可读概括

> 写给接手本项目的 Codex。你看不到此前的讨论记录，所以这份文档把讨论中形成的关键判断、决策、放弃的方案和尚未确定的问题都写了下来。请完整读完再动手。
>
> - 项目根目录：`D:\HHB\HKUST(GZ)_Library_code\DSAA2012_Deep_learning\MITA_vit`
> - 交接日期：2026-10-24
> - 项目性质：本科课程 DSAA 2012 Deep Learning（HKUST(GZ)，Fall 2026）的期末项目，单人作者。课程允许使用 AI 工具（已与用户确认）。
> - 已确认的条件：
>     - 本地 GPU：NVIDIA RTX 4060（8 GB 显存），用于开发、测试和小实验；
>     - HPC：用于大实验，用法见 `HPC.txt`；
>     - GitHub：你已连接用户的 GitHub，用户同意新建私有仓库 **MITA-VIT**；
>     - Hugging Face token：在根目录的 `.env` 文件中（变量名 `HF_TOKEN`），需要时读取；**绝不能写进任何会提交的文件**。
> - 有任何不确定的地方，随时找用户确认。

---

## 0. 一页速览

**研究问题**：一个训练好的 ViT，每一层的注意力能不能用“m 块图像区域、每块在读哪 k 个 token”这样一份很短的说明书概括出来？这份说明书是否忠实于模型真实的计算？

**方法（两步）**：

1. **读取模式**：不改动模型，从原模型每层每个 head 的 Q、K 中读出说明书。把 14×14 的 patch 网格分成 m 块区域，每块区域的 query 取平均得到 landmark，取 landmark 打分最高的 k 个 token 作为该区域的“读取集合”，再把每个 query 分配给最相似的 landmark。
2. **执行模式**：强制模型只按说明书计算（每个 query 只能读它所分配的读取集合，其余 token 屏蔽），看预测是否和原模型一致。一致率就是忠实性的度量。

**三个研究问题**：

- **RQ1**：说明书够忠实吗？（所有层同时替换，一致率随 k 的变化）
- **RQ2**：哪些层能被概括？（每次只替换一层）
- **RQ3**：说明书看到了什么？（读取集合是否落在物体上）

**三条贡献**：C1 一个读注意力的工具；C2 一种把忠实性变成可测数字的检验方法；C3 用它对 RQ1–RQ3 的回答。C1、C2 不依赖实验结果的好坏。

**核心原则**：这是课程项目，重视“相关工作 → 方法 → 实验 → 评估”的论证闭环，而不是和很多方法比较。只做完 RQ1 就是一个完整闭环，RQ2、RQ3 依次加深。

**截止时间**：报告（NeurIPS 模板，正文最多 4 页）和视频（8–10 分钟）都在 **2026-12-11 23:59** 截止，硬截止。

---

## 1. 你接手后要做的事（按顺序）

1. 读完本文档。
2. 按第 3 节的阅读顺序读文档，理解研究。
3. 执行第 2 节：配置环境、建立目录结构、把已有文档归位、初始化 Git 并推送到私有仓库 MITA-VIT、下载资源。
4. 建立第 9 节的进度管理与结果分析体系（`docs/progress.md`、`docs/experiment_log.md`、`docs/analysis/`）。
5. **停下来向用户汇报（检查点 1）**，用户确认后才进入下一步。
6. 按第 6 节的路线推进。路线中有固定的检查点，每到一个检查点都要停下来汇报，等用户确认后再继续。
7. 和用户的所有沟通遵循第 9.4 节的风格要求。

---

## 2. 目标一：配置项目环境

### 2.1 目录现状

根目录下目前有 4 个文件：

| 文件 | 内容 |
| --- | --- |
| `2026_Fall_DSAA_2012_Deep_Learning_Project_Guideline_and_Rubric_2.pdf`（文件名可能被截断） | 课程项目评分标准 |
| `研究设计书：用读取模式解释 ViT 的注意力.md` | 研究设计书：问题、方法公式、实验设计、贡献、QA、版面、附录 |
| `研究叙事稿：用读取模式解释 ViT 的注意力.md` | 研究叙事稿：按讲故事的顺序写的完整研究故事，结果处用【待填】占位 |
| `HPC.txt` | 用户的 HPC 计算资源使用方法（可能包含账号信息，**不要提交到 Git**） |
| `.env` | 存放 `HF_TOKEN`（Hugging Face token），**不要提交到 Git** |

加上本文档 `HANDOFF.md`。

### 2.2 目标目录结构

```
MITA_vit/
├── HANDOFF.md                 # 本文档（保留在根目录）
├── .env                       # HF_TOKEN 等密钥（不进 Git）
├── README.md                  # 项目简介与快速开始（你来写）
├── AGENTS.md                  # Codex 工作规范（你来写，内容见 2.4）
├── pyproject.toml             # 依赖与包配置（src 布局，可 pip install -e .）
├── requirements.txt           # 由 pyproject 导出或手写的锁定版本，便于 HPC 安装
├── .gitignore
├── configs/                   # 实验配置（YAML），每个 RQ 一个基础配置
│   ├── base.yaml
│   ├── rq1.yaml
│   ├── rq2.yaml
│   └── rq3.yaml
├── src/mita_probe/            # 核心代码（Python 包）
│   ├── __init__.py
│   ├── models.py              # timm 模型加载、注意力模块的替换与 hook
│   ├── probe.py               # 读取模式与执行模式的核心计算
│   ├── grouping.py            # 分组方式：grid（默认）、kmeans
│   ├── baselines.py           # 随机读取集合、CLS 注意力热图、中心高斯图
│   ├── data.py                # Imagenette / ImageNet 子集 / Oxford-IIIT Pet 的加载与预处理
│   ├── metrics.py             # 一致率、KL、前景比例、pointing game、bootstrap 置信区间
│   ├── io.py                  # 结果 CSV 读写、运行元数据（git commit、时间、配置）
│   └── viz.py                 # 画图工具（统一风格）
├── scripts/                   # 命令行入口
│   ├── download_models.py
│   ├── download_data.py
│   ├── run_rq1.py
│   ├── run_rq2.py
│   ├── run_rq3.py
│   └── make_figures.py
├── hpc/                       # HPC 作业脚本与环境脚本（依 HPC.txt 编写）
├── tests/                     # pytest 单元测试
├── data/                      # 数据集（不进 Git）
├── checkpoints/               # 模型权重缓存（不进 Git），设置 HF_HOME / TORCH_HOME 指向这里
├── experiments/               # 实验输出：每次运行一个子目录，含 config、log、results.csv
├── figures/                   # 论文用的最终图（png + pdf）
├── docs/
│   ├── course/rubric.pdf
│   ├── research_design.md     # 研究设计书（重命名后）
│   ├── research_narrative.md  # 研究叙事稿（重命名后）
│   ├── progress.md            # 进度看板（你来维护，见第 9 节）
│   ├── decisions.md           # 决策日志（你来维护，见 2.4）
│   ├── experiment_log.md      # 实验日志（你来维护，见第 9 节）
│   ├── analysis/              # 每个 RQ 的结果分析报告（见第 9 节）
│   └── hpc/HPC.txt            # 不进 Git
├── references/
│   ├── papers/                # 论文 PDF（不进 Git，太大）
│   ├── references.bib         # 进 Git
│   └── code/                  # 第三方参考代码（如 MiTA 官方仓库），不进 Git
└── paper/                     # NeurIPS LaTeX 论文（后期使用）
```

### 2.3 环境与目录初始化（Windows PowerShell）

本地有 NVIDIA RTX 4060（8 GB 显存）。先运行 `nvidia-smi` 查看驱动支持的 CUDA 版本，再安装对应的 CUDA 版 PyTorch。DeiT-S 推理在 4060 上 batch 128 左右即可，显存不是问题。

```powershell
cd "D:\HHB\HKUST(GZ)_Library_code\DSAA2012_Deep_learning\MITA_vit"

# 1) 建目录
$dirs = "configs","src\mita_probe","scripts","hpc","tests","data","checkpoints","experiments","figures",`
        "docs\course","docs\hpc","references\papers","references\code","paper"
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

# 2) 已有文档归位并改为 ASCII 文件名（中文全角冒号在部分工具中会出问题）
Move-Item -LiteralPath (Get-ChildItem "2026_Fall_DSAA_2012_Deep_Learning_*.pdf").FullName "docs\course\rubric.pdf"
Move-Item -LiteralPath (Get-ChildItem "研究设计书*.md").FullName "docs\research_design.md"
Move-Item -LiteralPath (Get-ChildItem "研究叙事稿*.md").FullName "docs\research_narrative.md"
Move-Item -LiteralPath "HPC.txt" "docs\hpc\HPC.txt"

# 3) Python 虚拟环境（建议 Python 3.11）
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 4) 安装 PyTorch：到 https://pytorch.org/get-started/locally/ 选择 Windows + pip + 与驱动匹配的 CUDA 版本
#    装完验证：python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 5) 安装本项目（pyproject.toml 写好之后）
pip install -e ".[dev]"

# 6) 读取 .env 中的 HF_TOKEN：代码中用 python-dotenv 加载，或在 PowerShell 中临时设置环境变量
#    不要把 token 打印到日志里，也不要写进任何会提交的文件
```

**依赖**（写进 `pyproject.toml`，torch/torchvision 单独按平台安装）：

| 包 | 用途 |
| --- | --- |
| `torch`、`torchvision` | 模型与数据（torchvision 自带 OxfordIIITPet） |
| `timm>=1.0` | 加载 DeiT-S 等预训练模型；**锁定具体版本**，因为要替换其 Attention 模块 |
| `datasets`、`huggingface_hub` | 下载 ImageNet 验证集（Hugging Face 门控数据集） |
| `numpy`、`pandas` | 计算与结果表 |
| `matplotlib` | 画图 |
| `pyyaml` | 配置文件 |
| `tqdm`、`pillow` | 进度条、图像 |
| `python-dotenv` | 从 `.env` 读取 `HF_TOKEN` |
| `pytest`（dev） | 单元测试 |

K-means 用 PyTorch 自己写（批量、可复现），不需要 scikit-learn。

**Git 与 GitHub**：

```powershell
git init
# 写好 .gitignore 后，先确认 .env 和 docs/hpc/HPC.txt 没有被暂存：
git add .
git status
git commit -m "chore: project scaffold, move docs into docs/"

# 在用户的 GitHub 上新建私有仓库 MITA-VIT 并推送
# 有 GitHub CLI 时：
gh repo create MITA-VIT --private --source . --remote origin --push
# 没有 gh 时，用你已连接的 GitHub 能力新建私有仓库，再 git remote add origin <url> 并 git push -u origin main
```

`.gitignore` 至少包含：

```
.env
.venv/
__pycache__/
*.pyc
data/
checkpoints/
experiments/*/raw/
references/papers/
references/code/
docs/hpc/HPC.txt
*.pth
*.pt
*.safetensors
.ipynb_checkpoints/
```

`experiments/` 下每次运行的 `results.csv`、`config.yaml`、`log.txt` 很小，**要提交**，方便本地和 HPC 之间通过 GitHub 同步结果。

### 2.4 要写的两个项目文档

**README.md**：一段项目简介（可取第 0 节）、目录说明、快速开始（建环境、跑测试、跑 RQ1 的调试版）、文档索引。

**AGENTS.md**：写明你（以及之后的 Codex 会话）的工作规范，至少包括：

1. **研究设计以 `docs/research_design.md` 为准。** 不要擅自改变研究问题、三个 RQ、核心对比或主指标。需要改动时先写进 `docs/decisions.md` 的“提议”一栏，经用户同意后再改。
2. **范围控制。** 核心实验只有第 5 节列出的内容；选做实验只有在核心实验完成并经用户同意后才做。
3. **所有实验由配置驱动。** 每次运行在 `experiments/<日期>_<rq>_<简述>/` 下保存配置、日志、结果 CSV，并记录 git commit。
4. **固定随机种子。** 图片子集用种子 0 抽取，随机对照用种子 0–4。
5. **先测试后实验。** 改动 `probe.py`、`grouping.py`、`models.py` 后必须先跑 `pytest`。
6. **结果表格式统一**（见 5.5），画图脚本只读结果 CSV。
7. **进度与分析。** 按第 9 节维护 `docs/progress.md`、`docs/experiment_log.md` 和 `docs/analysis/`。每次运行实验都要在实验日志里留一条记录。
8. **检查点。** 第 6 节的每个检查点都必须停下来向用户汇报，等用户确认后再继续。
9. **叙事稿中的【待填】** 只有在结果确认后、经用户审阅才填写。
10. **需要先问用户的事**：首次在 HPC 上提交作业、任何会改变研究设计的决定、删除文件，以及任何你拿不准的事。宁可多问一句。
11. **密钥。** `HF_TOKEN` 只从 `.env` 或环境变量读取；`.env` 和 `docs/hpc/HPC.txt` 永远不要提交，也不要打印到日志。
12. **语言**：代码、注释、提交信息用英文；面向用户的文档和汇报用中文，风格见第 9.4 节。
13. **HPC 相关操作一律先读 `docs/hpc/HPC.txt`**，不要猜测集群配置。

**docs/decisions.md**：从本文档第 7 节的内容初始化，分“已确定 / 已放弃 / 待定 / 提议”四栏。

**docs/experiment_log.md**：按日期记录。

### 2.5 本地与 HPC 的分工

| 场景 | 在哪里做 |
| --- | --- |
| 写代码、单元测试、画图、论文写作 | 本地 |
| 小实验：Imagenette 试跑、ImageNet 几百张的快速验证、调参前后的对比 | 本地 RTX 4060 |
| 大实验：ImageNet 5,000 张的 RQ1、RQ2 正式实验；RQ3 正式实验；随机对照多种子；选做实验 | HPC |

经验规则：单次运行预计超过 30 分钟，或需要同时跑多组设置时，放到 HPC。核心实验总量不大（约 30 万次 DeiT-S 前向），HPC 排队太久时，RQ1 的正式实验也可以在 4060 上跑完，先问用户。

**HPC 的具体用法全部以 `docs/hpc/HPC.txt` 为准**。读完之后你需要：

1. 弄清楚调度系统（例如是否是 Slurm）、GPU 分区、存储路径（home / scratch）、能否联网、Python 环境的建法；
2. 在 `hpc/` 下写：环境建立脚本、每个 RQ 的作业脚本模板、数据下载脚本（数据集**直接在 HPC 上下载**，不要从本地上传）；
3. 代码同步：通过 GitHub 私有仓库 MITA-VIT，本地 push、HPC pull。如果 HPC 不能访问 GitHub，按 HPC.txt 的说明改用 `scp`/`rsync`；
4. 结果回传：HPC 上把 `experiments/` 下的小文件（CSV、配置、日志）提交并 push，本地 pull；大文件留在 HPC；
5. 在 HPC 上把 `HF_HOME`、`TORCH_HOME` 指向 scratch 或项目的 `checkpoints/`，避免占满 home 配额；在 HPC 上单独创建 `.env`（手动拷贝，不经过 Git）；
6. 把以上做法写进 README 的“在 HPC 上运行”一节。

算力需求很小：核心实验约 30 万次 DeiT-S 前向（每张图 4.6 GMACs），单张 GPU 一小时以内。不需要 flash-attn。

### 2.6 需要下载的资源

**模型**（timm 首次调用时自动下载到 `HF_HOME`；可写 `scripts/download_models.py` 预下载）：

| 用途 | 模型 | timm 名称 | 来源 |
| --- | --- | --- | --- |
| 核心 | DeiT-S/16，ImageNet-1k 有监督 | `deit_small_patch16_224.fb_in1k` | https://huggingface.co/timm/deit_small_patch16_224.fb_in1k |
| 选做 | DINO ViT-S/16，自监督，无分类头 | `vit_small_patch16_224.dino` | https://huggingface.co/timm/vit_small_patch16_224.dino |
| 选做 | DINO 官方线性分类头（线性评估 77.0%） | — | https://github.com/facebookresearch/dino （README 中 “linear weights” 链接） |
| 选做 | DINOv2 ViT-S/14 + 4 registers | `vit_small_patch14_reg4_dinov2.lvd142m` | https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m |

DeiT-S 关键参数：22.1M 参数，224×224 输入，patch 16，14×14 = 196 个 patch token + 1 个 CLS = 197 个 token，embed dim 384，6 个 head（每个 64 维），12 层。

**数据集**（存放在 `data/` 下，HPC 上放 scratch）：

| 用途 | 数据集 | 获取方式 | 存放位置 |
| --- | --- | --- | --- |
| 调试 | Imagenette（160px 版即可） | https://github.com/fastai/imagenette ，README 中有下载链接 | `data/imagenette2-160/` |
| RQ1、RQ2 | ImageNet-1k 验证集（50,000 张，只用其中随机 5,000 张） | 先查 HPC 上是否已有共享的 ImageNet。没有的话用 Hugging Face 门控数据集 https://huggingface.co/datasets/ILSVRC/imagenet-1k ，token 用 `.env` 中的 `HF_TOKEN`，只下载 validation 分割。如果报“未授权”，说明用户账号还没在该页面同意条款（非商业研究与教育用途），请用户去页面点一下同意 | `data/imagenet_val/` |
| RQ3 | Oxford-IIIT Pet 测试集，37 类猫狗，带分割 trimap | `torchvision.datasets.OxfordIIITPet(root="data", split="test", target_types="segmentation", download=True)` | `data/oxford-iiit-pet/` |
| 选做 | Waterbirds | https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz | `data/waterbirds/` |

注意：Imagenette 的文件夹名是 WordNet ID，需要映射到 ImageNet 的 1000 类编号（用于检查原模型准确率）。一致率本身不需要标签。

**文献**（PDF 存 `references/papers/`，BibTeX 写进 `references/references.bib`）：

| 优先级 | 论文 | 链接 | 读什么 |
| --- | --- | --- | --- |
| 必读 | MiTA: Mixture-of-Top-k Attention (2026) | https://arxiv.org/abs/2602.01219 | §3.1 快权重视角、§3.2 方法与算法 1、图 4、附录 A 消融、附录 C |
| 必读 | Fast Transformers with Clustered Attention (Vyas et al., NeurIPS 2020) | https://arxiv.org/abs/2007.04825 | 按 query 聚类 + 每簇 top-k，是我们 K-means 对照的来源 |
| 必读 | Quantifying Attention Flow in Transformers (Abnar & Zuidema, ACL 2020) | https://arxiv.org/abs/2005.00928 | attention rollout；深层 token 信息混合 |
| 相关工作 | Attention is not Explanation (Jain & Wallace, NAACL 2019) | https://arxiv.org/abs/1902.10186 | 注意力能否当解释的争论 |
| 相关工作 | Attention is not not Explanation (Wiegreffe & Pinter, EMNLP 2019) | https://arxiv.org/abs/1908.04626 | 同上 |
| 相关工作 | Transformer Interpretability Beyond Attention Visualization (Chefer et al., CVPR 2021) | https://arxiv.org/abs/2012.09838 | ViT 热图解释方法 |
| 相关工作 | ERASER (DeYoung et al., ACL 2020) | https://arxiv.org/abs/1911.03429 | “充分性”：只保留解释看预测是否保持 |
| 相关工作 | Dissecting Recall of Factual Associations (Geva et al., 2023) | https://arxiv.org/abs/2304.14767 | attention knockout |
| 相关工作 | DeiT (Touvron et al., ICML 2021) | https://arxiv.org/abs/2012.12877 | 核心模型 |
| 相关工作 | DINO (Caron et al., 2021) | https://arxiv.org/abs/2104.14294 | CLS 注意力图 |
| 参考 | Memory-efficient Transformers via Top-k Attention (Gupta et al., 2021) | https://arxiv.org/abs/2106.06899 | 推理时 top-k 截断 |
| 参考 | Vision Transformers Need Registers (Darcet et al., ICLR 2024) | https://arxiv.org/abs/2309.16588 | 背景伪影 token |
| 参考 | RISE (Petsiuk et al., BMVC 2018) | https://arxiv.org/abs/1806.07421 | pointing game、删除/插入指标 |
| 参考 | Teaching Matters (Walmer et al., CVPR 2023) | https://arxiv.org/abs/2212.03862 | 不同监督方式下 ViT 注意力的比较 |

**参考代码**（可选，不是核心依赖）：MiTA 官方实现 https://github.com/QishuaiWen/MiTA ，clone 到 `references/code/MiTA/` 仅供对照阅读。我们不用它的模型权重，也不需要它依赖的 flash-attn。

---

## 3. 目标二：理解这个研究

### 3.1 文档阅读顺序

1. **本文档**（全部）。
2. **`docs/research_design.md`**（最重要）：
    - §1 研究问题与主张、§2 研究故事：理解为什么做；
    - §4 必备基础知识：注意力即快权重 MLP，含手算例子；
    - §5 方法：读取模式与执行模式的全部公式、手算例子、指标定义，**这是写代码的依据**；
    - §6 实验与评估：三个 RQ 的具体设置和主图；
    - §7 核心贡献与结果预案；
    - 附录 D：一层的核心代码示意和容易出错的地方。
3. **`docs/research_narrative.md`**：完整的研究故事，用来理解每个实验在故事中的位置；其中【待填】是将来要用实验结果填的地方。
4. **`docs/course/rubric.pdf`**：§1 Overview、Page Limits、Time Line、§3 Grading Rubric。
5. **`references/papers/` 中的 MiTA 论文**：§3.2 和图 4。
6. **`docs/hpc/HPC.txt`**：开始准备 HPC 时再读。

### 3.2 研究背景（为什么做）

- ViT 一张图在 DeiT-S 中产生约 280 万个注意力数值（197×197 × 6 heads × 12 layers），人读不懂。
- 现有解释方法（最后一层 CLS 注意力、attention rollout 等）把它们压成一张热图，有两个问题：丢失“哪个区域在读哪个区域”的结构；忠实性难以检验（“attention is (not) explanation” 之争）。
- MiTA 论文给了两个启发：（1）注意力可以写成一个两层 MLP，key 是隐藏神经元的输入权重、value 是输出权重，每个 token 贡献一个神经元，权重由当前输入临时生成（快权重）；（2）许多 query 激活的是同一小组神经元：只让每个 query 读一小组共享的 top-k token，模型仍能正常工作，标准注意力训练的模型推理时换成 MiTA 能保留 95% 以上精度（MiTA 附录 C）。MiTA 图 4 还显示深层只有物体上的少数 token 被选中。
- 我们把 MiTA 的 landmark + top-k 机制**当作解释工具**，用在一个普通的、未改动的 ViT 上，并用执行的方式检验解释是否忠实。

### 3.3 方法的精确定义（写代码时以此为准）

在某一层、某一个 head 中（每个 head 独立计算）。记 q_i, k_j, v_j ∈ R^64，d = 64，token 0 是 CLS，1–196 是 14×14 的 patch。

**读取模式**：

1. 区域划分：把 14×14 patch 网格用 2D 自适应平均池化的窗口分成 m 块（m = 16 即 4×4）。CLS 不属于任何区域。
2. landmark：`q̃_r = mean_{i ∈ R_r} q_i`。
3. 读取集合：`s_{r,j} = q̃_r · k_j / sqrt(d)`，对全部 197 个 token（含 CLS）取 top-k，得 `S_r`。
   性质：`q̃_r · k_j` 等于区域内各 `q_i · k_j` 的平均（线性），所以 `S_r` 是“这块区域整体最想读的 k 个 token”。
4. 分配：`e(i) = argmax_r q_i · q̃_r`，所有 197 个 query（含 CLS）都要分配。query 不一定被分到自己所在的区域。

说明书 = {S_1, …, S_m}（共 m×k 个 token 编号）+ {e(i)}。

**执行模式（严格版本，核心）**：query i 只能读 `S_{e(i)}`，其余 token 的 logit 设为 −∞，在剩余 token 上 softmax，再乘 V。打分仍用 q_i 自己，模型权重不变。

**执行模式（宽松版本，选做）**：额外让每个 query 读 m 个区域摘要：key 为 q̃_r，value 为 `ṽ_r = Σ_j softmax_j(s_{r,j}) v_j`，与读取集合拼在一起做一次 softmax（即完整 MiTA）。

**替换范围**：RQ1 用“所有层同时替换”；RQ2 用“每次只替换一层，其余层保持原样”。

**指标**：

- 说明规模（可读性）：每层每 head m×k 个编号（原注意力 197×197 ≈ 3.9 万）。
- 预测一致率（主指标）：`Agree = mean_n 1[argmax f(x_n) == argmax f̃(x_n)]`，比较的是原模型自己的预测，不是真实标签。
- 辅助：替换前后预测分布的 KL 散度；无分类头模型用 CLS 最终特征的余弦相似度。
- 前景比例（RQ3）：读取集合中落在前景的 token 占比；对照值为前景面积占比。

`docs/research_design.md` 附录 D 有一段已验证过的参考实现（随机张量上，k = 197 时与完整注意力的差为 0；landmark 平均性质的数值误差约 5e-6），可直接作为 `probe.py` 的起点。

### 3.4 预期贡献与三种结果

- C1 工具、C2 检验方法：不依赖结果。
- C3 发现：取决于结果，任何结果都能写成发现：
    - A：k 很小时一致率就很高 → “注意力可以被很短、可读、可检验的说明书概括”；
    - B：一致率普遍很低 → “ViT 注意力比 MiTA 图 4 暗示的更个性化，难以用少数共享模式概括；我们第一次把这一点量化”；
    - C：有的层高、有的层低（最可能）→ “注意力在哪些层可读，以及可读层的内容”。

---

## 4. 课程评分标准与本项目的对应

报告 35 分、视频 20 分、proposal 5 分（已提交别的方向，老师确认最终项目可以换方向）。报告按五项打分并取平均：

| Rubric 评分项 | 本项目如何满足 |
| --- | --- |
| Contribution | C1 工具 + C2 检验方法 + C3 发现；RQ1 本身就构成闭环。超出“复现已有方法”：机制来自 MiTA / Clustered Attention，但用途（解释）和检验方式（执行）是新的 |
| Clarity | 论证链：相关工作 → 空白 → 方法 → 三个 RQ → 结论回到空白；三张主图风格统一；版面按 `research_design.md` §9 |
| Depth | 受控对比（随机读取集合 5 个种子、K-means 分组）；bootstrap 置信区间；逐层分析；单元测试保证实现正确 |
| Literature Review | 三类相关工作（注意力解释、分组 + top-k 高效注意力、“只保留解释”的忠实性检验），见 `research_design.md` §3 与本文 2.6 |
| Formatting | NeurIPS preprint 模板；正文 ≤ 4 页（含图表），附录与参考文献不计；编号章节；规范引用 |

视频要求：8–10 分钟，MP4，slides 为 PDF、最多 15 页。

---

## 5. 实验规划

### 5.1 共同设置

- 模型：`deit_small_patch16_224.fb_in1k`，eval 模式，使用 timm 的默认 eval 预处理（`timm.data.resolve_model_data_config` + `create_transform`）。
- 数据：调试用 Imagenette；RQ1、RQ2 用 ImageNet 验证集中以种子 0 随机抽取的 5,000 张（抽样索引存成文件，所有实验共用）；RQ3 用 Oxford-IIIT Pet 测试集。
- 说明书：m = 16（4×4），k ∈ {4, 8, 16, 32, 64}，严格版本。

### 5.2 RQ1：说明书够忠实吗？

- 做法：所有层同时替换，对每个 k 计算一致率。
- 两个对照：
    - **随机读取集合**：每块区域从 197 个 token 中均匀无放回随机选 k 个，分配规则 e(i) 不变；种子 0–4，报告均值和标准差。
    - **K-means 分组**：对每张图、每层、每个 head 的 196 个 patch query 聚成 m = 16 类（PyTorch 批量实现，10 次迭代），类中心作为 landmark，其余步骤（top-k、分配、执行）与网格版本完全相同。
- 统计：我们的方法与随机对照的一致率之差，用 bootstrap（1,000 次重采样图片）给 95% 置信区间。
- 参照线：一致率 90%（事先约定，报告达到 90% 所需的最小 k）。
- **图 2**：横轴 k（对数刻度），纵轴一致率，三条线：ours（grid）、random read set（误差条）、k-means grouping。

### 5.3 RQ2：哪些层能被概括？

- 做法：每次只替换第 l 层（l = 1…12），其余层不变；k ∈ {8, 32}，m = 16。
- **图 3**：横轴层号 1–12，纵轴一致率，两条线（k = 8、k = 32）。

### 5.4 RQ3：说明书看到了什么？

- 数据：Oxford-IIIT Pet 测试集。trimap 通常为 1 = 前景、2 = 背景、3 = 边界，**下载后核对**。mask 要做与图片完全相同的几何变换（resize + center crop，最近邻插值）。
- token 级前景判定：每个 patch 对应 16×16 像素；在非边界像素中前景占比 ≥ 0.5 记为前景 token，全是边界像素的 token 不参与统计（这是默认方案，见 7.3 待定项）。
- 指标：
    - 逐层前景比例，与前景面积占比对照；
    - 选中次数图：每个 patch token 被多少个读取集合选中（对 16 块区域、6 个 head 求和），取最后几层的平均作为热图；
    - pointing game：热图最大值所在 token 是否为前景。
- 两个参照：最后一层 CLS 注意力热图（CLS 那一行的 patch 部分，6 个 head 平均）；中心高斯图（以图片中心为均值的二维高斯）。
- **图 4**：左边逐层前景比例曲线（带前景面积占比参照线）；右边 2–3 个例子，依次为原图、选中次数图、CLS 注意力图，以及一块背景区域的读取集合。另可附一张小表：方法 × {前景比例, pointing game}。

### 5.5 结果 CSV 格式

所有实验写同一格式（每行一个设置）：

```
run_id, timestamp, git_commit, model, dataset, n_images, rq,
replace_scope (all | layer), layer (int 或空), grouping (grid | kmeans),
readset (ours | random), seed, mode (strict | lenient), m, k,
agree, agree_ci_low, agree_ci_high, kl, acc_original, acc_executed, notes
```

RQ3 的结果另存一张表：`run_id, method, layer, fg_ratio, fg_area_ratio, pointing_acc, n_images`。

### 5.6 必须通过的单元测试（`tests/`）

1. **k = 197 恒等**：执行模式的注意力输出与完整注意力一致（同一数学路径下误差为 0；与 timm 原始前向相比误差 < 1e-5，且 5,000 张图的预测一致率为 100%）。
2. **平均性质**：landmark 打分等于区域内各 query 打分的平均（误差 < 1e-4）。
3. **手算例子**（`research_design.md` §5.2，token 编号从 1 开始）：
    - q1 = (3, 1)，q2 = (−1, 1)，q3 = q4 = (1, −1)；k1 = (1, 0)，k2 = (0, 2)，k3 = (2, 2)，k4 = (0, −1)；区域 A = {1, 2}，B = {3, 4}；k = 2；不除 √d；
    - landmark A = (1, 1)，B = (1, −1)；S_A = {3, 2}，S_B = {1, 4}；q1、q2 分给 A，q3、q4 分给 B；
    - q1 的完整权重约 (0.0067, 0.0025, 0.9907, 0.0001)，严格执行后约 (0, 0.0025, 0.9975, 0)。
4. **无 NaN**：任意 k ≥ 1 时执行结果无 NaN（每个 query 至少能读 k 个 token）。
5. **原模型准确率**：Imagenette / ImageNet 子集上原模型准确率合理（DeiT-S 公布的 ImageNet top-1 为 79.9%）。

### 5.7 接入 timm 的注意事项

- timm 的 ViT 注意力模块在 `model.blocks[i].attn`，其 forward 可能走融合的 `F.scaled_dot_product_attention`（`fused_attn` 开关），拿不到中间的 Q、K。需要用一个自定义 forward 替换：复现 `qkv` 线性层、reshape 成 heads、`q_norm` / `k_norm`、缩放、softmax、`attn @ v`、`proj`，然后在 softmax 前插入我们的掩码逻辑。
- **以安装的 timm 版本源码为准**，特别注意缩放因子是乘在 q 上还是乘在 logits 上；锁定 timm 版本。
- 读取模式只需要“观察”，可以在替换后的 forward 中把读取集合、分配、注意力矩阵存到一个缓存对象里，不改变输出。

### 5.8 选做实验（核心完成并经用户同意后才做）

按建议的优先顺序：

1. 在 DINO ViT-S/16 上重复图 2、图 3（与 DeiT-S 架构相同，只差训练方式）；
2. 宽松版本 vs 严格版本；
3. 其他 m 值（4、9、25、49）的扫描，放附录热图；
4. 更多热图参照：attention rollout、Grad-CAM（pytorch-grad-cam）、Chefer 等；
5. 删除/插入 AUC、参数随机化合理性检查；
6. DINOv2 + registers 的伪影检查；
7. Waterbirds 应用（需要微调）。

---

## 6. 分阶段实施路线与检查点

不要“先把所有代码写完，再一次性跑完所有实验”。RQ1 的结果会影响 RQ2、RQ3 的细节，所以每个 RQ 都走一个小循环：**写代码 → 小规模试跑 → 正式实验 → 写分析报告 → 和用户讨论**。

标 ★ 的是检查点：到了必须停下来，按第 9 节的格式向用户汇报，等用户确认后再继续。

| 阶段 | 内容 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| P0 环境 | 第 2 节全部：目录、文档归位、venv、依赖、Git 与私有仓库 MITA-VIT、README、AGENTS、decisions、progress、文献与模型下载 | 可运行的项目骨架 | `pytest` 能跑；DeiT-S 能在 4060 上对一张图推理；`.env`、`HPC.txt` 没有进 Git |
| ★ 检查点 1：环境汇报 | 汇报环境版本、目录结构、仓库链接、已下载和未下载的资源 | — | 用户确认 |
| P1 核心代码 | `models.py`（替换注意力）、`probe.py`（读取 + 严格执行）、`grouping.py`（grid、kmeans）、`baselines.py`（随机读取集合）、`metrics.py`（一致率、bootstrap）、`data.py`（Imagenette） | 核心包 + 测试 | 5.6 的测试 1–4 全部通过；Imagenette 上原模型准确率合理 |
| ★ 检查点 2：代码讲解 | 用 `research_design.md` §5.2 的手算例子，一步步讲清 `probe.py` 如何读取、如何执行，并展示测试结果 | — | 用户确认看懂 |
| P2 RQ1 试跑 | 本地 4060：Imagenette 约 500 张，跑完整 RQ1（含随机对照与 K-means） | 试跑 CSV + 草图 | 曲线趋势合理；k = 197 时一致率 100% |
| ★ 检查点 3：试跑汇报 | 草图、有无异常、正式实验的计划与预计耗时 | — | 用户确认后再上 HPC |
| P3 HPC 适配 | 读 HPC.txt，写 `hpc/` 脚本，在 HPC 上建环境、准备数据、跑通一个小作业 | `hpc/` 脚本、README 的 HPC 一节 | 小作业成功，结果能通过 GitHub 同步回本地 |
| P4 RQ1 正式实验 | ImageNet 5,000 张 | 图 2、结果 CSV、`docs/analysis/rq1.md` | — |
| ★ 检查点 4：RQ1 讨论 | 按 9.3 的模板汇报。**至此研究闭环已经完成** | — | 讨论后由用户决定：进入 RQ2 / 补充实验 / 调整设计 |
| P5 RQ2 | 逐层替换 | 图 3、`docs/analysis/rq2.md` | — |
| ★ 检查点 5：RQ2 讨论 | 同上 | — | 用户决定下一步 |
| P6 RQ3 | Pet 数据、前景比例、两个参照、例子图 | 图 4（+ 小表）、`docs/analysis/rq3.md` | — |
| ★ 检查点 6：RQ3 讨论 | 同上，并讨论做哪些选做实验 | — | 用户决定 |
| P7 选做 | 5.8 中经用户同意的项目 | 附加图表与分析 | — |
| P8 收尾 | 定稿三张图（统一风格，png + pdf）；把结果整理成可填入叙事稿【待填】的句子 | `figures/`、结果摘要 | 用户审阅确认 |
| P9 论文 | `paper/` 下建 NeurIPS 模板，按 `research_design.md` §9 的版面与用户协作写作 | 论文草稿 | 正文 ≤ 4 页 |

---

## 7. 讨论中形成的决策

### 7.1 已确定的设计决策

| 决策 | 内容 | 原因 |
| --- | --- | --- |
| 研究方向 | 可解释性；把 MiTA 的机制当工具，去解释一个普通的、未改动的 ViT | 用户被 MiTA 的两个 idea 打动（快权重视角、深层自动“剪枝”聚焦物体），希望以 MiTA 为思路起点做可解释性 |
| 解释的形式 | 读取模式：从原模型的 Q、K 中读出说明书，不替换注意力 | 读取不改动模型，比“把 MiTA 插进模型当探针”更干净 |
| 忠实性检验 | 执行模式：只按说明书计算，看预测一致率 | 把忠实性变成可测数字，对应 ERASER 的“充分性”思路 |
| 研究结构 | 3 个 RQ，3 张主图（外加论文中的方法示意图） | 课程项目重视论证闭环，不追求大规模对比 |
| 核心对比只有 4 个 | RQ1：随机读取集合、K-means 分组；RQ3：CLS 注意力热图、中心高斯图 | 每个对比都服务于我们自己的分析 |
| 分层交付 | 按 RQ1 → RQ2 → RQ3 推进，每一步都是闭环，做到哪写到哪 | 时间不确定时保证有完整成果 |
| 可读性的定义 | 说明规模 m×k | 简单、诚实；用户调研超出范围 |
| 主执行范围 | RQ1 所有层同时替换；RQ2 每次替换一层 | 前者一个数字回答主问题，后者定位哪层难 |
| 严格版本为主 | 核心实验只用严格版本；宽松版本（完整 MiTA）是选做 | 严格版本直接对应“说明书够不够” |
| 主设置 | m = 16 固定，k ∈ {4, 8, 16, 32, 64}；RQ2 用 k ∈ {8, 32} | 曲线一目了然；其他 m 放附录 |
| 模型 | DeiT-S（timm） | 小、标准、有分类头；与 DINO ViT-S/16 架构相同，便于选做对比 |
| 数据 | Imagenette 调试；ImageNet 验证集 5,000 张（种子 0）；Oxford-IIIT Pet 测试集 | 规模足够、获取简单；Pet 有前景标注 |
| 不用 flash-attn | 自己用掩码在完整注意力矩阵上实现 | N = 197 很小，掩码实现简单且精确 |
| 写作基调 | 讲清楚我们自己的贡献，不写成防御性的“逐条回应质疑” | 用户明确要求 |
| 文档风格 | 直接、具体、带数字和公式，避免元解释和堆砌术语 | 用户是大三本科生，要求清晰易懂 |
| 贡献表述 | 不主张“机制是新的”（来自 Clustered Attention / MiTA），也不主张“注意力等价于快权重 MLP”是新结论（已有代数事实） | 新意在于用途（可读解释）和检验方式（执行） |

### 7.2 已放弃的方案及原因

| 方案 | 放弃原因 |
| --- | --- |
| 加速方向：在深层丢弃从未被选中的 token，测提速 | 需要与成熟的 token pruning（EViT、ToMe 等）竞争；用户更想做可解释性 |
| 研究 MiTA 本身“为什么会聚焦物体” | 用户希望把 MiTA 当工具去解释别的模型，而不是研究 MiTA |
| 把 MiTA 注意力插进标准 ViT 当探针，再解释替换后的模型 | 替换后已不是原模型；改为“读取（不改模型）+ 执行检验” |
| 跨模型测量注意力的“有效宽度” | 注意力熵、距离的跨监督方式比较已有大量工作（如 Teaching Matters, CVPR 2023）；推理时 top-k 截断可行也已知（Gupta 2021） |
| 以 Waterbirds 捷径诊断为主线 | Waterbirds 上的定性注意力可视化已很多；降为选做应用 |
| 用文本作 landmark 解释 CLIP | 风险高：CLIP 的 patch token 与文本空间对齐较差 |
| 7 条假设、4 个实验层次、十几种对比方法的大设计 | 过于复杂，偏离论证闭环；收敛为 3 个 RQ + 4 个对比 |
| 用户调研衡量可读性 | 超出课程项目范围 |

### 7.3 尚未确定的问题

| 问题 | 当前默认方案 | 谁来定 |
| --- | --- | --- |
| ImageNet 验证集从哪里获取 | 先查 HPC 是否有共享副本；没有则用 Hugging Face（`.env` 中的 token） | 查 HPC.txt |
| RQ3 热图用哪几层 | 最后 4 层（9–12）的选中次数平均；同时报告逐层结果 | 看结果后与用户商定 |
| trimap → token 前景判定 | 非边界像素中前景占比 ≥ 0.5 | 可调整，写进 decisions |
| 中心高斯图的宽度 σ | σ = 图像边长的 1/4 | 可调整，写进 decisions |
| K-means 的初始化 | 用网格 landmark 初始化（确定性），10 次迭代 | 可调整 |
| K-means 版本的 query 分配 | 与网格版本相同（argmax q_i · landmark），以保证只有分组方式不同 | 可调整 |
| 90% 一致率参照线是否合适 | 先用 90%，看曲线后再定 | 看结果后与用户商定 |
| 做哪些选做实验 | 优先 DINO（5.8 的顺序） | 核心完成后问用户 |
| DINO 的忠实性指标 | CLS 特征余弦相似度；有时间再加载官方线性头算一致率 | 做到时再定 |
| 论文题目 | 暂定 *Reading ViT Attention through Verifiable Summaries* | 用户 |

---

## 8. 已与用户确认的事项

| 事项 | 结论 |
| --- | --- |
| 本地 GPU | NVIDIA RTX 4060（8 GB），小实验在本地跑 |
| HPC | 大实验在 HPC 跑，用法见 `docs/hpc/HPC.txt` |
| 代码仓库 | 在用户 GitHub 上新建私有仓库 MITA-VIT |
| Hugging Face | token 在 `.env` 的 `HF_TOKEN`，可直接使用 |
| AI 工具 | 课程允许使用 |
| 工作方式 | 按第 6 节的检查点推进；有任何不确定的地方随时问用户 |

仍需在做到时确认：首次在 HPC 上提交作业前，请用户看一下作业脚本。

---

## 9. 协作规范：进度管理、结果分析与沟通方式

用户之后会和你一起讨论实验结果，所以你需要自己把进度和分析组织好，让用户随时能看懂“做到哪了、结果说明什么、下一步有哪些选择”。

### 9.1 进度看板 `docs/progress.md`

每次开始和结束工作时更新。结构固定：

```markdown
# 进度看板（最后更新：YYYY-MM-DD）

## 当前阶段
P? xxx；下一个检查点：★ 检查点 ?

## 任务
| 任务 | 状态（待做 / 进行中 / 完成 / 卡住） | 备注 |

## 需要用户决定或帮忙的事
- ...

## 已完成的里程碑
- YYYY-MM-DD：...
```

### 9.2 实验日志 `docs/experiment_log.md`

每次运行实验（包括失败的）都加一条，最新的放最上面：

```markdown
## YYYY-MM-DD  <run_id>
- 目的：回答哪个 RQ / 检查什么
- 设置：配置文件路径；和默认设置的差别
- 在哪跑：本地 4060 / HPC；耗时
- 结果：关键数字（例如 k=16 时一致率 xx.x%，随机对照 xx.x% ± x.x）
- 异常：有就写，没有写“无”
- 下一步：...
```

### 9.3 结果分析报告 `docs/analysis/rq1.md`（rq2、rq3 同理）

每个 RQ 的正式实验完成后写一份，作为检查点讨论的材料。固定结构：

1. **一句话结论**：带关键数字。
2. **这个实验回答什么问题**：对应 `research_design.md` §6 的哪一部分。
3. **设置**：与设计书是否一致；如有差异，说明原因。
4. **结果**：主图，加一张关键数字小表。
5. **和预期对照**：设计书预期什么，实际得到什么，属于 `research_design.md` §7 的哪种情况（A / B / C）。
6. **可信度检查**：单元测试是否通过；k = 197 时是否为 100%；原模型准确率是否正常；随机对照不同种子之间的波动；置信区间。
7. **异常与可能原因**：每个可能原因写一个“怎么验证”的小实验。
8. **下一步选项**：给 2–3 个选项，每个写清楚目的、要改什么、预计耗时、对论文有什么用；给出你的推荐和理由，由用户决定。
9. **可以写进论文的句子**：草稿，供用户修改后填入叙事稿的【待填】。

### 9.4 调整实验时的规则

- 设计书中事先定好的设置（m = 16、k 的取值、种子、5,000 张图）**不为了让结果好看而改**。
- 需要改动时，先在 `docs/decisions.md` 的“提议”栏写清楚：为什么改、改什么、预期会看到什么；用户同意后再执行。
- **一次只改一个因素**，改动前后的结果放在同一张表里对比。
- 不理想的结果和理想的结果一样记录、一样汇报。按 `research_design.md` §7，不理想的结果也能写成发现。

### 9.5 和用户沟通的风格

用户是大三本科生，希望沟通清楚、好懂。请学习 `docs/research_design.md` 的写法：

- **先说结论，再说细节。**
- **具体**：用数字、公式和小例子，不说空泛的形容词。
- **一步一步讲**：先讲要解决什么问题，再讲怎么做、结果是什么。
- **新名词第一次出现时，用一句话解释。**
- **不写元解释**（如“这一节将回答……”），不堆砌术语。
- **需要用户决定时**，给出选项和你的推荐，说明理由。
- 中文为主，关键术语附英文。

一个好的汇报示例：

> RQ1 试跑完成：在 Imagenette 的 500 张图上，k = 16 时一致率是 93.2%，同样大小的随机读取集合只有 41.5%（5 个种子，标准差 2.1）。也就是说，选对 token 很重要。
>
> 一个需要你决定的地方：K-means 分组在 k = 4 时一致率反而比网格分组高 6 个百分点，k ≥ 16 后两者几乎一样。我建议正式实验照原计划做，在分析报告里把 k = 4 的差异作为一个观察写出来。你同意吗？

（示例中的数字只是格式示意，不是真实结果。）
