# 进度（最后更新：2026-11-05）

## v5（分支 `study/v5`，2026-11-05 开始）

- 作者确认新方向：快慢权重视角下的“读 → 诊断 → 修正”闭环（B33，替代 B21）。依据文件存于 `docs/v5/`。
- 已完成：盘点（`docs/v5/inventory.md`）；读法修正 C1–C3、合用读法、三个对照的实现（`src/mita_probe/readings.py`）；`diagnose`、`correct` 实验和配置；单元测试（全部 113 项通过）；Colab 冒烟运行。
- 已写：执行规格 `docs/experiment_plan_v5.md`、提议 B34–B40、“需要作者决定的问题”（规格第 10 节，Q1–Q10）。
- **暂停点：等作者确认后冻结判定参数，再正式运行（约 13 小时、26 单元）。** Colab 实例已停止。

## 当前状态：研究完成，交付物为初稿

| 交付物 | 位置 | 状态 |
| --- | --- | --- |
| 课程论文（NeurIPS 格式，正文 4 页） | `paper/main.pdf` | 初稿；待作者补充姓名信息和 AI 辅助说明，并审阅措辞 |
| 报告幻灯片（13 页，含英文讲稿） | `slides/mita_probe_talk.pptx` | 初稿；待作者修改后导出 PDF |
| 报告视频（8–10 分钟） | — | 未开始 |
| 研究笔记本 | `notebooks/mita_probe_research.ipynb` | 完成（CPU 约 2 分钟可重跑） |
| 完整结果报告 | `docs/analysis/v3_report.md`、`docs/analysis/supplement_s1_s3.md`、`docs/budget/report.md` | 完成 |

课程截止：论文、幻灯片、视频均为 12 月 11 日。

## 注意力预算研究（分支 `study/attention-budget`，2026-11-03 完成）

- 文档：[研究设计](budget/research_design.md)（用户提供）、[工程设计](budget/engineering_design.md)、[预试验报告](budget/pilot_report.md)、[正式报告](budget/report.md)；决策 B27–B32。
- 判定：H1 不成立（MiTA-DeiT-T 与 DINO 一样聚焦）；H2 不成立（Small −0.011，Tiny +0.037，方向依赖规模）；H6 成立（DeiT-S 移走 8 个神经元后，高范数 token 减少 92%，物体聚焦只 +0.006）。DeiT-B 没通过 H3 关卡，RQ3 只作描述。
- 按事先写定的规则，论文以原有主线为主，新增第 4.4 节和附录 C；同时修正执行证据（12 层全部删掉注意力只剩 0.1%）。幻灯片 13 页，笔记本新增 5.12 节。
- Colab：本轮约用 7.6 单元，余额 182.10；实例已停止。
- 这一分支还没有合并到 `main`，也没有推送（GitHub 需要重新连接）。

## 研究过程（摘要）

1. **第一、二代（10-24 至 10-28）**：建立探针、RQ1/RQ2 解剖（DeiT-S、DINO、MiTA-DeiT-S）、Pet 上的 RQ3 和中心对照。报告：`docs/analysis/probe_report.md`、`docs/analysis/v2_anatomy.md`、`docs/analysis/rq3_diagnosis.md`。
2. **研究重新定位与 v3（10-29）**：把探针作为解释方法，与 5 种方法在统一检验下比较；3 个模型、5 个数据集，33 个正式运行。报告：`docs/analysis/v3_report.md`；决策 B01–B20。
3. **定位 C 与补充实验 S1–S3（10-31）**：完整注意力对照、三类逐层图、高范数 token；21 个正式运行。报告：`docs/analysis/supplement_s1_s3.md`；决策 B21–B26。
4. **交付物（11-01 至 11-03）**：研究笔记本、论文初稿、幻灯片初稿；仓库整理（旧材料归档，见 `docs/archive/README.md`）。
5. **注意力预算研究（11-03）**：7 个模型、11,919 张图，检验 top-k 训练和测试时额外 token 能否把注意力引向物体；报告 `docs/budget/report.md`，决策 B27–B32。

逐日的详细看板保存在 `docs/archive/progress_history.md`；每次运行（包括失败的）见 `docs/experiment_log.md`。

## 待作者处理

- `paper/main.tex` 中的两处 `TODO(author)`：作者信息；按课程规定说明 AI 辅助。
- 补全参考文献中写成 "et al." 的作者列表（AttnLRP、TokenTM、PartImageNet、Zero-TPrune）。
- 幻灯片封面和讲稿中的作者信息；录制视频。
- Hugging Face 令牌在项目结束后撤销。
