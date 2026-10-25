# P3：首次 HPC 作业审阅

本轮在 `feature/hpc-adaptation` 分支准备迁移。首次 GPU 作业和正式 RQ1 均须用户确认后启动。

执行结果（2026-10-26）：用户已确认脚本，并授权按可用性调整资源。10 GB MIG 排队无资源后，采用平台可分配的 2×A800 / CPU 16 核规格，研究脚本使用单 GPU。首次 29 项测试和 50 张 × 35 组验收通过，随后正式 5,000 张 RQ1 完成；结果已取回，环境已停止。下文保留原审阅方案；正式分析见 `docs/analysis/rq1.md`。

## 环境与数据

- 平台是按需 JupyterLab 开发环境，不使用 Slurm；集群 `AI-X86_NVIDIA`，镜像 `admin/webide-jupyter:v6.2`。
- 已创建 CPU 2 核环境 `mita-p3-env` 并通过 SSH 实测。镜像自带 Python 3.10.13 / torch 2.3.1+cu121，不能直接运行本项目。
- 在持久化 SSD 上独立安装 Python 3.12，再以 `PYTHON_BIN=<解释器绝对路径> bash hpc/setup.sh` 创建项目 `.venv`。安装固定的 torch 2.7.0、torchvision 0.22.0、timm 1.0.30 和其余锁定依赖，不修改平台镜像环境。
- `MITA_DATA_ROOT` 可指向持久化数据目录；`HF_HOME` 可指向持久化缓存。账号、动态 SSH 凭据和 token 只保存在忽略文件中。
- 数据版本固定为 `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`；权重版本固定为 `91327a9c99f98fe6b524cd4d397b7226b80e1365`，均与 P2 一致。
- 环境准备只做依赖安装、权重/数据下载与数据检查；CPU 阶段不执行模型实验。
- 远程环境 preflight 与 `pip check` 已通过：Python 3.12.15，torch 2.7.0+cu128，torchvision 0.22.0+cu128，timm 1.0.30；完整 72 项依赖清单随验收记录保存。
- HPC 直连 Hugging Face API 出现连接重置，准备数据时通过临时 SSH 隧道接入本机已有代理；官方 `.hf.co` CDN 可直接访问，最终大文件通过 CDN 下载。正式作业使用离线缓存，临时隧道已关闭。
- 远程代码 `8946e29` 已完成全部资产验收：ImageNet validation 共 50,000 张、1,000 类；14 个官方分片 SHA-256 全部通过；固定 5,000 张全部解码成功。模型权重、标签顺序、抽样索引等 12 项本地/远程对比全部一致，记录见 `experiments/2026-10-25_hpc_environment/`。
- 本轮 CPU 环境已停止，平台显示“退出”。Python、项目、数据和缓存保存在持久化 SSD；首次 GPU 环境须复用该挂载。尚未执行远程模型推理。

## 需要审阅的首次作业

脚本：`hpc/first_job.sh`。资源选择：同一集群与镜像，CPU 4 核 + A800 MIG 10 GB（1 个 GPU 实例），复用相同持久化挂载。平台已列出此规格，实际是否可分配以启动时为准；不自动申请多卡。选择依据是本地 8 GB 显卡已完成同样 batch=32 的完整流程。

在准备好数据的 GPU 开发环境内执行：

```bash
bash hpc/first_job.sh experiments/2026-10-25_hpc_first
```

1. 检查代码已提交、解释器隔离、依赖版本、CUDA 可见性、缓存权重和完整 ImageNet 数据、固定抽样索引；保存 `preflight.json`。
2. 执行完整 pytest（含 CUDA 测试），保存 `tests.xml`。
3. 对正式 seed=0 的 5,000 张子集取前 50 张。先验证读取无扰动与 k=197 恒等，再运行完整 35 组：grid 五个 k、随机五个种子 × 五个 k、K-means 五个 k。
4. 保存逐图结果、配置、版本、日志、CSV 和曲线。全部启用离线模式；任一步失败立即停止；不覆盖已有输出。

FP32、batch=32、m=16、k={4,8,16,32,64}、全 12 层严格执行保持不变。50 张仅用于迁移验收，不用于论文结论。GPU 具体型号和驱动须在首次作业 preflight 记录；验收须达到预测完全一致、最大恒等 logit 误差 ≤1e-5、无 NaN。

## 确认后的正式实验

首次验收通过后，按用户确认的范围另行运行：

```bash
bash hpc/run.sh rq1 experiments/2026-10-25_rq1_imagenet
```

这条命令使用原来的 `configs/rq1.yaml`：固定 5,000 张、35 组完整对比、随机种子 0–4、K-means，以及 bootstrap 置信区间。首次作业脚本不自动调用它。完成后停止开发环境，取回小型结果与曲线，进入检查点讲解。

## 验证记录

- 本地 29 项 pytest 通过（含 CPU/CUDA 核心探针与新增子集前缀、持久化数据路径测试）。
- Git Bash 对 setup / first_job / run / download_data 的语法检查通过。
- 本地完整预演：`experiments/2026-10-25_hpc_local_smoke/`，代码 `8d02692`，50 张 × 35 组，耗时 19.9 秒（不含启动与载入）；读取与 k=197 的最大输出误差均为 0，恒等预测一致率 100%。PNG/PDF 已生成并检查。
- 本地 ready preflight 通过，数据 50,000 张、固定子集 5,000 张与模型 revision 均通过检查。远程 GPU 兼容性仍须审批后的首次作业验收。
- 粗略预算：首次作业预留 10 分钟；正式实验先预留 2 小时，再根据首次作业的实测吞吐量修正。本地 500 张试跑约 171 秒，按数量外推约 29 分钟；MIG 与本地 GPU 性能不同，HPC 时间以实测为准。
