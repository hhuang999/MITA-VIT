# MITA-VIT

## 研究与范围

研究设计以 `docs/research_design.md` 为准，交接上下文见 `HANDOFF.md`。核心范围为交接文档 §5 的 RQ1–RQ3；选做内容须核心完成并经用户同意后开始。研究问题、核心对比或主指标的变更先写入 `docs/decisions.md` 的“提议”，确认后实施。叙事稿的【待填】须结果确认并经用户审阅后填写。

## 环境与实现

- 本地使用 `.venv/Scripts/python.exe`，不修改邻近课程环境。CUDA PyTorch 按 README 单独安装；timm 固定为 pyproject 中的版本。其他依赖由 requirements.txt 锁定。
- 代码、注释、提交信息用英文；面向用户的文档用中文。直接解决当前问题，不添加未使用的抽象或空实现。
- 实验由 `configs/` 配置驱动，输出到 `experiments/<日期>_<rq>_<简述>/`，保存配置、日志、CSV 和 git commit。图表只读取结果 CSV；列名遵循 `HANDOFF.md` §5.5。
- 图片子集种子为 0，随机对照种子为 0–4。不得为改善结果擅自修改预设参数。一次只调整一个因素。
- 改动 `probe.py`、`grouping.py`、`models.py` 后必须先跑 pytest，再跑实验。核心测试标准见 `HANDOFF.md` §5.6。
- 每次开始和结束工作更新 `docs/progress.md`；包括失败实验在内，都写入 `docs/experiment_log.md`。正式 RQ 完成后按交接文档 §9.3 写 `docs/analysis/rq*.md`。

## 检查点与 HPC

遵循 `HANDOFF.md` §6 的检查点。P0 环境完成后先汇报，用户确认后进入 P1；其后每个检查点同理。首次 HPC 作业须让用户审阅脚本后确认。HPC 操作先读本地 `docs/hpc/HPC.txt`；平台为按需 JupyterLab 开发环境，不假设有 Slurm。个人账号、动态 SSH 地址和密码不写入公共脚本。

会改变研究方向、设计或质量标准的取舍由用户决定；已授权范围内的安装、验证和实现由代理自行完成。不删除原始资料和用户改动。

## 凭据与版本管理

`HF_TOKEN` 只从 `.env` 或环境变量读取，不打印凭据。`.env`、`docs/hpc/HPC.txt`、数据、权重和第三方代码不进 Git。实验的小型 CSV、配置、日志应提交，原始大输出留在运行机器。代码通过私有 MITA-VIT 仓库同步。
