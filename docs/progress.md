# 进度（最后更新：2026-11-08）

## 第二阶段：发布整理（2026-11-08 开始，分支 `release/v1`）

- 作者决定（B49）：第 4.5 节用版本 B；图 2(b) 改为第 9–12 层合并，第 12 层版本作附录图；开始 `02_release_cleanup.md`。
- 进行中。没有推送。

## 第一阶段补充实验（2026-11-07，交接包 `docs/handoff/2026-11-07/`）

- E-C2 和 X1–X9 全部完成，结果在 `docs/analysis/v5_report.md` 第 9 节（B45–B48，全部为探索性）。第 4.5 节版本建议：B。
- 作者已确认（B49），第二阶段已开始。

## v5（分支 `study/v5`，2026-11-05）

- 作者确认新方向（B33，替代 B21）和 Q1–Q10（B41 冻结）。全部正式实验完成，结果见 **[docs/analysis/v5_report.md](analysis/v5_report.md)**；图 2、图 3 在 `figures/figure_v5_*`，表 1 在 `experiments/2026-11-05_v5_analysis/table1.{csv,md}`。
- 结论：读出的神经元确实被使用（P1 成立）；三个前提都在深层失效（P2 成立）；P3 部分成立（1/4）：只有“按贡献计分”（C3）稳定改进读法，“重新定位”（C2）让读法变差，合用的结果依赖顺序；阴性对照 P4 全部成立；改模型两行都不比改读法好（P6 成立）。
- 意外：ClickMe 验证分片的图片几乎都不在 ImageNet 验证集中（B43），ClickMe 结果应打折扣。
- 待作者：按报告写论文正文；决定是否把 `study/v5` 合并进 `main`（B41 已同意，结果出来后在本地合并）。

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
