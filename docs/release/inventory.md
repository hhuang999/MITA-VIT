# 发布整理追溯表（第二阶段第一步）

**这张表回答一个问题：论文里的每个要素，靠哪些运行、配置、脚本、函数和数据算出来。** 找得到的保留；找不到的进入“删除候选”；拿不准的进入“待定”。整理结束时，这张表改写成 `docs/reproduce.md`。

**一句话总结：** 从 04 号清单的每个编号出发，沿“运行 → 配置 → 脚本 → 函数 → 数据”往下找；表外的东西都不进 `release/v1`。

- 起点：`study/v5` 提交 81bad1c（2026-11-08）。
- 简写：`A/` = `experiments/2026-11-05_v5_analysis/`，`E/` = `experiments/2026-11-07_explore_c2_analysis/`，`S/` = `experiments/2026-11-07_v5_supp_analysis/`。模块名省略 `src/mita_probe/` 前缀。
- “三个模型”指 `deit`、`deit_b`、`dino`。

---

## 1. 正文要素

| 论文要素 | 运行文件夹 | 配置 | 脚本 | 用到的 `src` 模块和函数 | 数据与模型 |
| --- | --- | --- | --- | --- | --- |
| F1 图 1 | — | — | — | — | 作者手画 |
| F2a 图 2(a) | `2026-11-05_read_ksweep_{三个模型}`；“无注意力”水平线来自 `2026-11-03_ab_t7_{三个模型}` | `read_ksweep_*.yaml`（继承 `anatomy_*.yaml`）；T7 用 `attention_budget.yaml` | `run.py anatomy --stage rq2`、`run.py attention_off`；`analyze_v5.py: ksweep`；`plot_paper_figures.py: figure_2a` | `experiments/anatomy.py: stage_rq2`；`experiments/attention_off.py`；`probe/`（`specs.expand_rq2, make_plan, resolve`；`pattern.build_pattern, token_mask, random_pattern`；`execute.Plan, execute, capture_qk`）；`experiments/common.py: dense_identity, outcomes`；`metrics.kl_divergence` | ImageNet 验证集固定 5,000 张（种子 0）；DeiT-S、DeiT-B、DINO + 线性头 |
| F2b 图 2(b) | `2026-11-05_correct_ablation_clickme_deit`（gallery npz） | `correct_ablation_clickme.yaml`（继承 `v5.yaml`） | `run.py correct`；`plot_paper_figures.py: figure_2b` | `experiments/correct.py: gallery_rows`、`all_maps`；`readings.py` | ClickMe 17,408 张；DeiT-S |
| F3 图 3(a) M1 | `2026-11-05_diagnose_m1_{三个模型}` | `v5.yaml`（diagnose 部分） | `run.py diagnose --stage m1`；`analyze_v5.py: diagnose` | `experiments/diagnose.py: stage_m1`；`readings.py: accumulate_means, m1_sums, radial_share`；`stats.ratio_interval` | 留出图片 2,000 张（均值图）；ImageNet-S 12,419 张 |
| F3 图 3(b) M2 | `2026-11-05_diagnose_probe_{三个模型}` | 同上 | `run.py diagnose --stage probe`；`analyze_v5.py: diagnose` | `experiments/diagnose.py: stage_probe, heldout, collect_tokens`；`identity.collect`；`readings.py: fit_position_probe, displacement, match_sigma, save_probes`；`data/heldout.py` | 留出图片 2,000 + 2,000 张（排除 ImageNet-S 与 ClickMe 重复）；ClickMe 缩略图 |
| F3 图 3(c) M3 | `2026-10-31_s_layers_{三个模型}`（旧流程，经 `2026-10-31_s_analysis/s3_layers.csv`） | `layer_reading.yaml` | `run.py layer_reading`；`analyze_supplement.py: s3` | `experiments/layer_reading.py`；`explain.attention_mass`；`interventions.sink_tokens`；`probe.capture_input_norms` | ImageNet-S 12,419 张 |
| F3 图 3(d) | `2026-10-31_s_e7_clickme_{三个模型}`（旧流程，经 `s2_layers.csv`） | `human_alignment.yaml` | `run.py human_alignment`；`analyze_supplement.py: s1_s2_clickme` | `experiments/human_alignment.py: alignment`；`explain.attention_mass` | ClickMe 17,408 张 |
| T1 表 1 | `2026-11-05_correct_ablation_{imagenet_s,clickme}_{三个模型}`；`correct_register_*_deit`；`correct_mita_*_mita` | `correct_ablation_*.yaml`、`correct_variant_*.yaml`、`correct_register_*.yaml`、`correct_mita_*.yaml` | `run.py correct`；`analyze_v5.py: correct, per_image, variant_rows, markdown_table` | `experiments/correct.py`；`readings.py`（全部读法与替身）；`explain.py: gae, centre, random_scores`；`plausibility.measures`；`faithfulness.rank, curve`；`human_alignment.alignment`；`budget.instrument, mita_attention`（变体）；`registers.edit_neurons, as_layers`；`stats` | ImageNet-S 12,419 张（富集）与固定 2,000 张（SRG）；ClickMe 17,408 张；+ DeiT-S register、MiTA-DeiT-S |
| N-RQ1 4.2 节 | 同 F2a、T1 | 同上 | `analyze_v5.py` | 同上 | 同上 |
| N-RQ2 4.3 节 | 同 F3；高范数份额同 F3(c)；值向量比例来自 `2026-11-03_ab_t3_{deit,deit_b}`（经 `ab_analysis_formal`）；平均图径向比例来自 `diagnose_m1_*/radial.csv` | `attention_budget.yaml`（T3） | `run.py budget --stage formal`；`analyze_budget.py` | `experiments/budget.py: ledger, head_rows`；`budget.instrument` | ImageNet-S（预算研究用统一预处理和 formal 划分） |
| N-RQ3 4.4 节 | 同 T1；顺序检验 `2026-11-05_correct_order_*` | `correct_order_*.yaml` | `analyze_v5.py: correct, judge`（`A/correct_tests.csv`、`A/variants.csv`、`A/a7_order.csv`） | 同 T1 | 同 T1 |
| N-last 4.4 节 | 同 T1（`R0_l8`–`R0_l12`、`*_l12` 行） | `v5.yaml` | `analyze_v5.py`（`A/a6_last_layer.csv`） | 同 T1 | 同 T1 |
| N-set 4.1 节 | 数据规模：各运行 metadata；ClickMe 泄漏：`diagnose_probe_*/heldout.json`；P4a/P4b：`2026-11-05_p4_*.csv`；健全性：`2026-11-05_correct_sanity_*` | `correct_sanity.yaml` | `check_p4.py`；`run.py sanity`；`analyze_v5.py: sanity` | `experiments/sanity.py`；`experiments/diagnose.py: heldout`；`data/heldout.py` | ImageNet-S 1,000 张（健全性） |
| N-intro 引言 0.41 | `2026-10-29_v3_e7_clickme_deit/analysis_alignment.csv`（centre 的 `spearman_pixel` 0.414） | `human_alignment.yaml` | `analyze_human_alignment.py` | `explain.centre`；`human_alignment.alignment` | ClickMe 17,408 张 |
| N-probe 4.5 节 0.27 | `2026-11-05_diagnose_probe_deit/identity.csv`（第 12 层 `top_probability`） | `v5.yaml` | `run.py diagnose --stage probe` | 同 F3(b) | 同 F3(b) |

## 2. 附录要素

| 论文要素 | 运行文件夹 | 配置 | 脚本 | 用到的 `src` 模块和函数 | 数据与模型 |
| --- | --- | --- | --- | --- | --- |
| AB1 一次限制 / 删除一层 | `2026-10-28_v2_rq1`、`2026-10-28_v2_dino_rq1`、`2026-10-29_v3_deit_b_rq1`（旧流程，经各 `*_analysis/rq1_anatomy.csv`）；`2026-11-03_ab_t7_*` | `anatomy_*.yaml`；`attention_budget.yaml` | `run.py anatomy --stage rq1`；`run.py attention_off`；`analyze_v5.py: extension_b` | `experiments/anatomy.py: stage_rq1`（只用 `L3` 设定）；`experiments/attention_off.py`；`probe/` | ImageNet 固定 5,000 张 |
| AB2 删头部 vs 身体图块 | `2026-10-29_v3_e8_parts_{三个模型}`（旧流程，`reliance.csv`） | `parts.yaml`、`parts_deit_b.yaml` | `run.py parts`；`analyze_parts.py: part_drops` | `experiments/parts.py: assignment, removals`（reliance 部分）；`faithfulness.run_kept, measures`；`data/partimagenet.py` | PartImageNet（部件标注） |
| AB3 top-k 图与完整注意力图的相似度 | `2026-10-31_s_e4_imagenet_s_{三个模型}`（旧流程，`raw/scores/*.npz`） | `explanations*.yaml` | `run.py explanations`；`analyze_supplement.py: s1_e4` | `explain.probe`（`weight='mass'`）、`explain.attention_mass`；`metrics.spearman` | ImageNet-S 12,419 张 |
| AC1 P1–P7 | `A/predictions.csv` | — | `analyze_v5.py: judge` | — | `docs/decisions.md` B33–B43 |
| AC2 P4a、P4b | `2026-11-05_p4_*.csv` | `v5.yaml`（`R0_early`、`C2_early`） | `check_p4.py` | 同 T1 | 同 T1 |
| AC3 复现检查 | `A/reproduction.csv` | — | `analyze_v5.py: reproduction` | — | 比较新旧两套代码；旧代码不进 release，只留 CSV |
| AC4 随机化健全性 | `2026-11-05_correct_sanity_*` | `correct_sanity.yaml` | `analyze_v5.py: sanity` | `experiments/sanity.py: stages, reinitialise`；`correct.load_lens`；`explanations.method_scores`（centre） | ImageNet-S 1,000 张 |
| AD1–AD3 | `A/table1.md`、`A/a7_order.csv`、`A/correct_tests.csv`、`A/a6_last_layer.csv` | 同 T1 | `analyze_v5.py` | 同 T1 | 同 T1 |
| AE C2 探索性分析 | `2026-11-07_explore_c2_{三个模型}`、`_probefix_*`；`2026-11-07_supp_heldout_*`、`supp_eval_*` | `explore_c2.yaml`、`explore_c2_probe.yaml`、`supp.yaml`、`supp_clickme.yaml` | `run.py explore_c2`、`run.py supp --stage heldout|eval`；`analyze_explore_c2.py`、`analyze_supp.py` | `experiments/explore_c2.py`；`experiments/supp.py`（MLPProbe、globalisation、supp_maps）；`readings.capture(per_head=True)`；`faithfulness.curve(prefix=)` | 留出图片；ImageNet-S；ClickMe |
| AE-fig | `E/` | 同上 | `analyze_explore_c2.py: figure` | — | — |
| AF2b-L12 | 同 F2b（`map_*_l12`） | 同 F2b | `plot_paper_figures.py: figure_2b(suffix='_l12')` | 同 F2b | 同 F2b |
| AX X1、X2、X3、X6 | `2026-11-07_supp_eval_*`；X6 用 v5 逐图结果 | `supp*.yaml` | `analyze_supp.py` | `experiments/supp.py: supp_maps, shuffled` | 同 T1 |

## 3. 模型的来源

| 模型 | 来源 | 代码 |
| --- | --- | --- |
| DeiT-S、DeiT-B | timm（Hugging Face，固定 revision） | `models/deit.py` |
| DINO ViT-S/16 + 线性头 | timm 主干 + Meta 官方线性头（URL + SHA256） | `models/dino.py` |
| DeiT-S + 测试时 register | DeiT-S；8 个 FFN 神经元由 T2 找到（`2026-11-03_ab_t2_deit/gate.json`） | `registers.py: edit_neurons, as_layers`；寻找：`experiments/register.py: search`、`registers.py: NeuronStats, ranked_neurons, random_neurons, matched_neurons, first_high_layer`；`budget.instrument` |
| MiTA-DeiT-S | 不是本仓库训练的；`colab/remote/setup.sh` 用 gdown 从 Google Drive 下载 `checkpoints/mita/best_checkpoint.pth`（SHA256 固定在 `models/mita.py`） | `models/mita.py`；`probe/execute.py: install`（MiTA 的注意力） |

## 4. 删除候选（不进 `release/v1`，仍保留在 `study/v5`）

**实验模块：** `experiments/` 下的 `matched_deletion`、`center_control`、`explanations`（只保留 AB3 需要的 `explain.probe`、`explain.attention_mass` 和表 1 参照用的 `gae`、`centre`、`random_scores`）、`shifted_crops`、`identity`、`gallery`、`model_check`、`budget`（只保留 `budget.instrument`、`mita_attention` 给变体和 register 寻找）、`sink_use`；`anatomy` 中除 `rq1`（只留 L3 设定）、`rq2`（只留 k 扫描）以外的阶段（noise、repro、mita_repro、mita_anatomy、pet）；`register` 的 formal 阶段；`human_alignment` 的运行部分（保留 `alignment` 函数）；`parts` 的 map 部分（保留 reliance）。

**库代码：** `sinks.py`；`results.py`；`interventions.py` 中的 `replace_category`、`replace_random_patches`、`replace_matched_patches`（保留 `sink_tokens`、`centre_rings`）；`explain.py` 中的 `rollout`、`gradcam`、`integrated_gradients`、`cls_attention`；`faithfulness.saco`、`keep_top` 条件；`stats.shapley`、`anatomy_table`、`trapezoid`；`budget.py` 中的 `Edits`、`effective_rank`、`categories`、`shares`、`ring_enrichment`（register 寻找若用到则保留）；`data/pet.py`、`data/hard_imagenet.py`；`models` 中的 `deit_t`、`mita_t`、`deit3_s`（预算研究的 7 个模型中论文不用的 4 个）。

**脚本：** `analyze_anatomy.py`、`analyze_budget.py`、`analyze_deit_b.py`、`analyze_dino.py`、`analyze_explanations.py`、`analyze_human_alignment.py`、`analyze_identity.py`、`analyze_matched_deletion.py`、`analyze_parts.py`、`analyze_sanity.py`、`analyze_shifted_crops.py`、`analyze_supplement.py`、`analyze_v3.sh`、`plot_*.py`（除论文图）、`paper_assets.py`、`tidy_results.py`、`download_references.py`、`verify_imagenet.py`（并入 `prepare_data.py` 的校验）。`analyze_v5.py`、`analyze_explore_c2.py`、`analyze_supp.py`、`check_p4.py`、`plot_paper_figures.py` 改写为 `make_tables.py`、`make_figures.py`。

**配置：** 除 v5、探索性分析和上述附录要素以外的全部（anatomy 的 rq2 设定表、explanations*、matched_deletion*、center_control、shifted_crops、identity、gallery、human_alignment*、parts_dense*、attention_budget*、layer_reading 等）。

**目录：** `slides/`、`paper/`、`notebooks/`、`colab/`（只保留新写的 `reproduce/colab.ipynb`）、`references/`、`docs/` 下除 `preregistration.md`、`reproduce.md` 以外的全部（`docs/release/` 中的整理文件保留到汇报为止）、`experiments/`（汇总 CSV 移到 `results/`）、旧图（`figures/` 只留论文图）。

## 5. 待定（先不删，写进汇报）

| 事项 | 现状 | 本次做法 |
| --- | --- | --- |
| F3(c)、F3(d) 的旧来源 | 来自 10-31 的 `layer_reading`、`human_alignment` 运行（旧流程）。按 R0 的定义，`attention_mass` 与 R0 只差一个正的常数倍 | 与 AB1–AB3 同样处理：用新代码（R0 逐层读法）重新实现并重跑，与旧结果比较 |
| N-RQ2 值向量比例 | 来自预算研究 T3：统一预处理、formal 划分 | 并入 `diagnose`，用论文的预处理在 ImageNet-S 全部图片上重算；与 0.29 / 0.23 的差别写进汇报 |
| N-intro 0.41 | 来自 v3 的 human_alignment 运行 | 表 1 的 ClickMe 运行已经算了 centre 的 `spearman_pixel`，改为从那里读；数值核对写进汇报 |
| fig3_diagnose.csv 中的 “E3” 行 | v3 的 100 张图探针，图 3 不用 | 不再生成 |
| `figures/` 中 v5 报告的其他图（figure_v5_*、figure13_* 等） | 论文不用 | 不进 release；附录图只留 AE-fig、AF2b-L12 |
| 逐图输出的总大小 | 待第二层验证后统计 | 超过 100 MB 就不进 git |
