# MITA-VIT

## 研究与范围

当前研究主线以 `docs/decisions.md` 的 B01–B16 和 `docs/experiment_plan_v3.md` 为准：把 MiTA 的 top-k 读取用作 ViT 的解释方法（RQ1 忠实、RQ2 结构、RQ3 常识、RQ4 与解释方法比较）。`docs/research_design.md`、`docs/research_design_v2.md` 和 `HANDOFF.md` 是第一、二代设计，只作历史参考。选做内容须核心完成并经用户同意后开始。研究问题、核心对比或主指标的变更先写入 `docs/decisions.md` 的"提议"，确认后实施。新实验的判定标准和预测在正式运行前写入决策日志。叙事稿的【待填】须结果确认并经用户审阅后填写。

## 环境与实现

- 本地 Windows 使用 `.venv/Scripts/python.exe`，不修改邻近课程环境；Colab 环境由 `colab/remote/setup.sh` 建立。CUDA PyTorch 按 README 单独安装；timm 固定为 pyproject 中的版本。其他依赖由 requirements.txt 锁定。
- 代码、注释、提交信息用英文；面向用户的文档用中文。直接解决当前问题，不添加未使用的抽象或空实现。
- 代码在 `src/mita_probe/`。实验由 `configs/` 配置驱动，用 `scripts/run.py <实验名>` 运行，输出到 `experiments/<日期>_<简述>/`，保存配置、日志、CSV 和 git commit；源代码有未提交改动时拒绝运行。图表只读取结果文件（结果 CSV 和 `analysis_*.csv`）。
- 图片子集种子为 0，随机对照种子为 0–4。不得为改善结果擅自修改预设参数。一次只调整一个因素。
- 改动 `src/mita_probe/` 后必须先跑 pytest，再跑实验。
- 每次开始和结束工作更新 `docs/progress.md`；包括失败实验在内，都写入 `docs/experiment_log.md`。正式实验完成后在 `docs/analysis/` 写分析报告。

## 检查点与算力

GPU 实验用 Colab CLI 运行（脚本在 `colab/`）。首次 HPC 作业须让用户审阅脚本后确认。HPC 操作先读本地 `docs/hpc/HPC.txt`；平台为按需 JupyterLab 开发环境，不假设有 Slurm。个人账号、动态 SSH 地址和密码不写入公共脚本。

会改变研究方向、设计或质量标准的取舍由用户决定；已授权范围内的安装、验证和实现由代理自行完成。不删除原始资料和用户改动。

## 凭据与版本管理

`HF_TOKEN` 只从 `.env` 或环境变量读取，不打印凭据。`.env`、`docs/hpc/HPC.txt`、数据、权重和第三方代码不进 Git。实验的小型 CSV、配置、日志应提交，原始大输出留在运行机器。代码通过私有 MITA-VIT 仓库同步。
