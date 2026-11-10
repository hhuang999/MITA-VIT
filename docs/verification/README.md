# Paper v8 release verification

实际验证时间（UTC）：2026-11-10T15:33:38.428544+00:00。验证代码提交：`02a0e73e50944a70a5383bd83417ee91053ae2bd`。

本轮验收范围是发布整理、论文与代码一致性、原始产物校验。实验阶段已完成，本轮未运行 GPU 实验、未重训模型或原始位置探针，也未生成新的原始推理输出。

## P0 / P1 / P2 交付

- P0：恢复完整历史预注册，附加 P1-P7 → R1-R3 映射；恢复真实运行记录与旧版 GPU 核查材料；保留原始数据并执行 CPU 重建。
- P1：同步作者 v8 PDF/LaTeX/图表；修正文档与注释；补齐 Figure 1b/D.1；新增严格校验、回归测试、安全构建和逐条证据映射。
- P2：新增 Python 3.12 CPU CI、固定版本的可选 gdown 依赖、完整证据包导出、校验和和清单。CI 云端运行状态另见 PR，本地检查结果如下。

## 实际验证结果

- 干净 Git ZIP 解压后安装成功；**106 项测试通过**，仅一个现有 NumPy 弃用提示。
- 19 张汇总 CSV 全部一致：18 张逐字节相同，最大绝对差 `1.6263032587282567e-19`，小于绝对容差 `1e-12`（相对容差 0）。
- 模型/方法/层集合、样本数、缺失值、置信区间及统计判定全部匹配。三个生成论文表及配对差值源表逐字节相同。
- Figure 1b、D.1、2 的文字、尺寸及 144 dpi 渲染像素完全相同；PDF 元数据不同，因此 PDF 文件字节不同。
- 175 份原始观察/参考结果/原图文件保持原 SHA-256；14 份配置语义和 37 个原有科学实现的可执行 AST 指纹一致。
- 42 个运行组合有真实来源，恢复 232 份原记录；12 个论文图表/代码项、26 条实证映射均指向实际文件。
- 原始八个 register 单元与 pilot CSV 精确一致。ZIP 实际解压检查通过，预注册、原始输出及来源资料均包含在内。

## 科学不变量（实际值）

| 项目 | 观察值 |
| --- | --- |
| C3_deit_enrichment | 0.0375663428926721 |
| C3_deit_b_enrichment | 0.0653888540864858 |
| C3_deit_clickme | 0.0333341463690849 |
| C3_deit_b_clickme | 0.0575702764539142 |
| register_enrichment | 0.0119796443956672 |
| register_clickme | 0.0128495262140819 |
| R1 | 3/3 and 6/6 comparable |
| R2 | 1/4 higher: deit_b/clickme |
| R3 | 4/4 not higher |
| relocation | 9/9 lower |
| deletion | 3/3 comparable |
| interpretation | post hoc |

## 逐表比较

| CSV | 行数 | 最大绝对差 | 字节相同 |
| --- | ---: | ---: | --- |
| `results/c2_analysis/errors.csv` | 378 | 0 | True |
| `results/c2_analysis/mean_prediction.csv` | 147 | 1.6263032587282567e-19 | False |
| `results/correct_tests.csv` | 99 | 0 | True |
| `results/diagnose.csv` | 252 | 0 | True |
| `results/k_sweep.csv` | 39 | 0 | True |
| `results/k_sweep_tests.csv` | 33 | 0 | True |
| `results/last_layer.csv` | 24 | 0 | True |
| `results/negative_controls.csv` | 9 | 0 | True |
| `results/numbers.csv` | 9 | 0 | True |
| `results/probe_accuracy.csv` | 36 | 0 | True |
| `results/radial.csv` | 288 | 0 | True |
| `results/supplementary/c2_variants.csv` | 33 | 0 | True |
| `results/supplementary/c3_mechanism.csv` | 18 | 0 | True |
| `results/supplementary/dose.csv` | 8 | 0 | True |
| `results/supplementary/globalisation.csv` | 462 | 0 | True |
| `results/supplementary/mlp_probe.csv` | 216 | 0 | True |
| `results/table1_long.csv` | 201 | 0 | True |
| `results/value_norm.csv` | 26 | 0 | True |
| `results/variants.csv` | 10 | 0 | True |

## 命令与环境

等价的仓库内命令如下；实际运行使用独立环境、干净 ZIP 及临时输出目录。机器私有临时路径不作为仓库接口。

```bash
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -r requirements-download.txt
python -m pip install -e . --no-deps
python -m pip check
python -m pytest -q
python scripts/validate_release.py --strict
bash reproduce/from_outputs.sh
python scripts/validate_release.py --strict --archive-check
```

完整数值/环境记录：[verification.json](verification.json)。逐文件变更：[changed-files.txt](changed-files.txt)。

CPU 测试与最终重建在验证提交上完成；shell 入口也在此前 534c8b6 上通过，此后数值构建器没有功能变化。验证记录提交仅增加文档；最终 ZIP 的精确提交由 `.git_archival.txt` 与导出清单记录。

WSL 的 root 与 Windows 仓库所有者不同，实际源仓库归档检查使用了仅对当前进程生效、仅指向当前仓库的 `safe.directory` 配置；没有修改全局 Git 配置。

## 未运行的事项与边界

GPU smoke/full、模型或原始探针训练、原始推理输出重生成：按作者范围要求不执行，不属于本轮验收条件。旧 2026-11-08 GPU 核查只证明它记录的旧版代码，未声称本轮进行了新的 GPU 完整复现。

旧 register 全验证集准确率/减少比例保留原始 CSV 和执行来源；当前流程不重算。缺失的旧启动命令、driver/full pip freeze/执行时 checkpoint 摘要明确为未知。

没有运行 LaTeX 编译（本机无工具链）；作者完整 11 页 PDF 保持原样，文献键与交叉引用已静态核查，无缺失项。

## 留给作者的两处措辞建议

1. Appendix A：object patch 的分母写为 non-ignored pixels，并说明 patch 至少有一个有效像素。
2. Appendix A：补充 value-norm 支撑测量使用 crop 0.875，主 DeiT 评估使用 0.9。

这两项不通过改变实验定义来处理；本次同步的作者正文未作修改。

## 提交包

`python scripts/export_release.py --output dist/course_code.zip` 导出完整证据包，并生成逐文件 SHA-256 清单和 ZIP 校验和。`archive/`、外部数据/模型、凭据和本地写作资料不进入 ZIP。提交包包含所需原始输出，可在无 GPU/外部数据的环境中重建报告结果。

`paperv8` 供审阅，PR 目标为 `release/v1`；不自动合并，`main`、`study/v5` 不变。
