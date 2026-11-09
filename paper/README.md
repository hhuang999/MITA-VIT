# 课程论文初稿

DSAA 2012 课程论文（NeurIPS 格式，正文 4 页，另附参考文献和附录）。定位见 `docs/decisions.md` B21、B26。

| 文件 | 内容 |
| --- | --- |
| `main.tex` | 正文和附录 |
| `references.bib` | 参考文献 |
| `tables/`、`figures/` | 由 `python scripts/paper_assets.py` 从结果文件生成，不要手改 |
| `main.pdf` | 编译好的初稿 |
| `review/` | 最终版：`main_review.tex`（正文）、`appendix.tex`（附录 A–E）、`references.bib`、`main_review.pdf`；`tables/` 由 `make_appendix_tables.py` 从 release/v1 的 `results/` 生成；`figure_1.pdf` 由 `make_figure1.py` 画出；`figure_A_c2.pdf` 由 `make_figure_c2.py` 画出 |

编译：

```bash
bash paper/get_style.sh
cd paper && latexmk -pdf main.tex
```

`main.tex` 中有两处 `TODO(author)`：作者信息，以及按课程规定说明 AI 辅助的情况。

最终版的表和图（在仓库根目录运行；`<results>` 是 release/v1 的 `results/` 文件夹）：

```bash
python paper/review/make_appendix_tables.py <results> paper/review/tables
python paper/review/make_figure1.py
python paper/review/make_figure_c2.py <results> experiments
```

最终版编译：

```bash
bash paper/get_style.sh
cd paper/review && TEXINPUTS=..: latexmk -pdf main_review.tex
```
