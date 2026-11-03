# 进度（最后更新：2026-11-03）

## 当前状态：研究完成，交付物为初稿

| 交付物 | 位置 | 状态 |
| --- | --- | --- |
| 课程论文（NeurIPS 格式，正文 4 页） | `paper/main.pdf` | 初稿；待作者补充姓名信息和 AI 辅助说明，并审阅措辞 |
| 报告幻灯片（12 页，含英文讲稿） | `slides/mita_probe_talk.pptx` | 初稿；待作者修改后导出 PDF |
| 报告视频（8–10 分钟） | — | 未开始 |
| 研究笔记本 | `notebooks/mita_probe_research.ipynb` | 完成（CPU 约 2 分钟可重跑） |
| 完整结果报告 | `docs/analysis/v3_report.md`、`docs/analysis/supplement_s1_s3.md` | 完成 |

课程截止：论文、幻灯片、视频均为 12 月 11 日。

## 研究过程（摘要）

1. **第一、二代（10-24 至 10-28）**：建立探针、RQ1/RQ2 解剖（DeiT-S、DINO、MiTA-DeiT-S）、Pet 上的 RQ3 和中心对照。报告：`docs/analysis/probe_report.md`、`docs/analysis/v2_anatomy.md`、`docs/analysis/rq3_diagnosis.md`。
2. **研究重新定位与 v3（10-29）**：把探针作为解释方法，与 5 种方法在统一检验下比较；3 个模型、5 个数据集，33 个正式运行。报告：`docs/analysis/v3_report.md`；决策 B01–B20。
3. **定位 C 与补充实验 S1–S3（10-31）**：完整注意力对照、三类逐层图、高范数 token；21 个正式运行。报告：`docs/analysis/supplement_s1_s3.md`；决策 B21–B26。
4. **交付物（11-01 至 11-03）**：研究笔记本、论文初稿、幻灯片初稿；仓库整理（旧材料归档，见 `docs/archive/README.md`）。

逐日的详细看板保存在 `docs/archive/progress_history.md`；每次运行（包括失败的）见 `docs/experiment_log.md`。

## 待作者处理

- `paper/main.tex` 中的两处 `TODO(author)`：作者信息；按课程规定说明 AI 辅助。
- 补全参考文献中写成 "et al." 的作者列表（AttnLRP、TokenTM、PartImageNet、Zero-TPrune）。
- 幻灯片封面和讲稿中的作者信息；录制视频。
- Hugging Face 令牌在项目结束后撤销。
