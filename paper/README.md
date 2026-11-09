# 课程论文初稿

DSAA 2012 课程论文（NeurIPS 格式，正文 4 页，另附参考文献和附录）。定位见 `docs/decisions.md` B21、B26。

| 文件 | 内容 |
| --- | --- |
| `main.tex` | 正文和附录 |
| `references.bib` | 参考文献 |
| `tables/`、`figures/` | 由 `python scripts/paper_assets.py` 从结果文件生成，不要手改 |
| `main.pdf` | 编译好的初稿 |
| `review/` | 送审版：`main_review.tex`（文字取自作者的 `paper_review_text.md`，原样排版）、`references.bib`、`main_review.pdf` |

编译：

```bash
bash paper/get_style.sh
cd paper && latexmk -pdf main.tex
```

`main.tex` 中有两处 `TODO(author)`：作者信息，以及按课程规定说明 AI 辅助的情况。

送审版编译：

```bash
bash paper/get_style.sh
cd paper/review && TEXINPUTS=..: latexmk -pdf main_review.tex
```
