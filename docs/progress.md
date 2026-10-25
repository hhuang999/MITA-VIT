# 进度看板（最后更新：2026-10-25）

## 当前阶段

用户已确认检查点 3，进入 P3 HPC 适配；独立分支 `feature/hpc-adaptation`。CPU 环境、完整数据与脚本准备已完成，CPU 开发环境已停止；首次 GPU 作业提交用户审阅，确认后执行 GPU 验收和 ImageNet 5,000 张正式 RQ1。

## 任务

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 阅读交接、设计、叙事与评分标准 | 完成 | 保留研究设计与原文 |
| 目录及文档归位、凭据隔离 | 完成 | env 改为 .env；HPC 说明不进 Git |
| 独立 Python 环境与依赖锁定 | 完成 | Python 3.12.11；torch 2.7.0+cu128；torchvision 0.22.0+cu128；timm 1.0.30 |
| 模型与公开数据准备 | 完成 | DeiT-S；Imagenette 验证集 3,925 张；Pet 测试集 3,669 对 |
| ImageNet 验证集 | 完成 | 50,000 张 / 1,000 类；14 个分片 SHA-256 通过；seed=0 的 5,000 张覆盖 994 类，全部可解码；32 张 CUDA 前向通过 |
| 文献 PDF 与 BibTeX | 完成 | 交接 §2.6 的 14 篇；PDF 可解析，保留 SHA-256 与原始 BibTeX |
| GPU 真实图片推理与 pytest | 完成 | 离线预训练权重推理、1 项结构/前向测试；不是 RQ 结果 |
| 私有仓库与首个提交 | 完成 | [hhuang999/MITA-VIT](https://github.com/hhuang999/MITA-VIT)，已确认 private；main 已推送并跟踪 origin/main |
| HPC 环境与脚本适配 | GPU 验收待审阅 | 独立 Python 3.12.15、固定依赖与 pip check 通过；完整 50,000 张及固定 5,000 张解码通过；CPU 环境已停止 |
| P1 核心探针实现 | 完成 | grid/k-means、读取与固定说明书执行、随机对照、指标、Imagenette 加载 |
| P1 单元与集成测试 | 完成 | 22 项通过，包含 CPU/CUDA、手算、平均性质、k=197 恒等、无 NaN、作用域和恢复 |
| P1 真实数据验收与讲解 | 完成 | Imagenette 3,925 张，准确率 80.28%；读取与 k=197 全层执行输出误差均为 0；讲解见 checkpoint_2.md |
| P2 完整 RQ1 试跑 | 完成 | 500 张、35 组；grid k=64 一致率 88.2%，random 51.88%，K-means 86.8%；耗时 170.7 秒 |
| P2 测试、曲线与检查点报告 | 完成 | 27 项测试；k=197 一致率 100%、输出误差 0；checkpoint_3.md 与 PNG/PDF 曲线 |

## 需要用户决定或帮忙的事

- P3：审阅 `docs/hpc_first_job_review.md` 与 `hpc/first_job.sh`，确认后进行 GPU 验收，通过后才运行正式 RQ1。

## 已完成的里程碑

- 2026-10-25：P3 远程 CPU 环境与资产验收通过（代码 `8946e29`）；14 个分片哈希、模型与抽样索引均与本地一致。CPU 环境已停止，临时网络隧道已关闭。首次 GPU 脚本见 `docs/hpc_first_job_review.md`；远程推理与正式 RQ1 尚未启动。
- 2026-10-24：确认 RTX 4060 8 GB 可用；建立独立项目目录和本地 Git。
- 2026-10-24：完成本地 CUDA 环境、模型/公开数据/文献准备；验收记录见 `experiments/2026-10-24_environment/`。
- 2026-10-24：本地初始化提交 `f114977`；经用户明确授权创建 MITA-VIT 私有仓库并推送。凭据、数据、权重和文献 PDF 均未纳入 Git。
- 2026-10-24：用户确认进入 P1；在独立分支完成核心探针，22 项测试通过，预训练模型在 Imagenette 上通过恒等验收；验收代码提交 `2b17f53`。
- 2026-10-25：用户完成 ImageNet 授权；用项目 `.env` 中的 token 请求 `validation-00000-of-00014.parquet` 的前 4 字节，返回 HTTP 206、`Content-Range: bytes 0-3/479922797`，内容为 `PAR1`。实际文件访问已验证，尚未下载完整验证集。
- 2026-10-25：用户确认检查点 2 并授权进入 P2；完成 Imagenette 500 张完整 RQ1，代码提交 `b436890`，结果见 `experiments/2026-10-25_rq1_pilot/`。预设 k≤64 的样本一致率均未达到 90%。
- 2026-10-25：完成 ImageNet validation 下载和数据验收，数据位于 `data/imagenet_val/`；源版本、14 个哈希、标签与解码验证见 `experiments/2026-10-25_imagenet_download/verification.json`。正式 5,000 张 RQ1 与恒等验收留待 P4。
