# 归档说明

这里保存已被最终版本取代、但值得留作记录的文档。它们不再更新，其中的链接可能指向已移出主线的运行。

| 文件 | 内容 |
| --- | --- |
| `HANDOFF.md`、`research_design.md`、`research_design_v2.md`、`research_narrative.md` | 第一、二代研究设计和叙事稿 |
| `checkpoint_2.md`、`checkpoint_3.md`、`hpc_first_job_review.md`、`colab_setup.md`、`overnight_validation_plan_2026-10-27.md` | 早期阶段的检查点、HPC 与 Colab 记录 |
| `README_2026-10-27.md` | 第二代时的 README |
| `progress_history.md` | 逐日的完整进度看板（至 2026-11-03） |
| `analysis/` | 早期分析报告：RQ1、RQ2、Pet 上的 RQ3、夜间运行、V00–V03 验证 |

**移出主线的运行和图**（共 70 个运行文件夹、30 组带日期的旧图）保存在 GitHub 分支 `archive/pre-cleanup-2026-11-03`，用下面的命令即可查看或取回：

```bash
git fetch origin archive/pre-cleanup-2026-11-03
git checkout origin/archive/pre-cleanup-2026-11-03 -- experiments/<运行文件夹>
```

移出的内容包括：环境检查和 HPC 运行、冒烟运行（smoke）、第一代 RQ1/RQ2/RQ3 运行、V00–V03 验证、`slides/figures/` 中三张未被幻灯片使用的图片。最终报告、论文、幻灯片和笔记本引用的运行都留在主线。
