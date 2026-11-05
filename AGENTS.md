# MITA-VIT

## 研究与范围

当前研究主线以 `docs/decisions.md` 为准：v5（B33，替代 B21）——以快慢权重视角为主角，对预训练 ViT 的注意力走一轮“读 → 诊断 → 修正”闭环，依据和盘点在 `docs/v5/`，执行规格见 `docs/experiment_plan_v5.md`。此前的定位 C（B21）——用经过执行验证的 top-k 读取名单，在中心偏差对照下分析预训练 ViT 逐层读了什么；实验设计见 `docs/experiment_plan_v3.md` 和 B22–B25，结果见 B20、B26。补充的注意力预算研究（7 个模型，稀疏化和测试时额外 token 两个杠杆）见 `docs/budget/`，决策 B27–B32。最终交付物在 `paper/`（论文）、`slides/`（幻灯片）、`notebooks/`（研究笔记本）。早期设计稿、交接文档和中间报告在 `docs/archive/`，只作历史参考；早期运行和带日期的旧图保存在 Git 分支 `archive/pre-cleanup-2026-11-03`。研究问题、核心对比或主指标的变更先写入 `docs/decisions.md` 的"提议"，确认后实施。新实验的判定标准和预测在正式运行前写入决策日志。

## 环境与实现

- 本地 Windows 使用 `.venv/Scripts/python.exe`，不修改邻近课程环境；Colab 环境由 `colab/remote/setup.sh` 建立。CUDA PyTorch 按 README 单独安装；timm 固定为 pyproject 中的版本。其他依赖由 requirements.txt 锁定。
- 代码、注释、提交信息用英文；面向用户的文档用中文。直接解决当前问题，不添加未使用的抽象或空实现。
- 代码在 `src/mita_probe/`。实验由 `configs/` 配置驱动，用 `scripts/run.py <实验名>` 运行，输出到 `experiments/<日期>_<简述>/`，保存配置、日志、CSV 和 git commit；源代码有未提交改动时拒绝运行。图表只读取结果文件（结果 CSV 和 `analysis_*.csv`）。
- 图片子集种子为 0，随机对照种子为 0–4。不得为改善结果擅自修改预设参数。一次只调整一个因素。
- 改动 `src/mita_probe/` 后必须先跑 pytest，再跑实验。
- 每次开始和结束工作更新 `docs/progress.md`；包括失败实验在内，都写入 `docs/experiment_log.md`。正式实验完成后在 `docs/analysis/` 写分析报告。

## 检查点与算力

GPU 实验用 Colab CLI 运行（脚本在 `colab/`，用法见 README）。个人账号、动态地址和密码不写入公共脚本。

会改变研究方向、设计或质量标准的取舍由用户决定；已授权范围内的安装、验证和实现由代理自行完成。不删除原始资料和用户改动；整理时先归档（`docs/archive/` 或存档分支），再从主线移除。

## 凭据与版本管理

`HF_TOKEN` 只从 `.env` 或环境变量读取，不打印凭据。`.env`、数据、权重和第三方代码（如 NeurIPS 样式文件）不进 Git。实验的小型 CSV、配置、日志应提交，原始大输出留在运行机器。代码通过私有 MITA-VIT 仓库同步。**推送到 GitHub 只在用户明确要求时进行**（用户 2026-11-05 指示）：平时只提交到本地分支，不主动推送；自动提醒推送的提示也不作为推送依据。
