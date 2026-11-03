# 课程论文初稿

DSAA 2012 课程论文（NeurIPS 格式，正文 4 页，另附参考文献和附录）。定位见 `docs/decisions.md` B21、B26。

| 文件 | 内容 |
| --- | --- |
| `main.tex` | 正文和附录 |
| `references.bib` | 参考文献 |
| `tables/`、`figures/` | 由 `python scripts/paper_assets.py` 从结果文件生成，不要手改 |
| `main.pdf` | 编译好的初稿 |

编译：

```bash
bash paper/get_style.sh
cd paper && latexmk -pdf main.tex
```

`main.tex` 中有两处 `TODO(author)`：作者信息，以及按课程规定说明 AI 辅助的情况。
