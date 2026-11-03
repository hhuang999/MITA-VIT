# 报告幻灯片初稿

`mita_probe_talk.pptx`：12 页，16:9，适合 8–10 分钟报告。风格是顶会口头报告常见的简洁样式：内容页白底，封面和结尾深色底，每页一个信息点。配色与论文图一致（蓝 = top-k 探针，深灰 = 完整注意力，橙 = CLS 注意力，金 = GAE）。每页都有英文讲稿（PowerPoint 的"备注"栏）。

| 文件 | 内容 |
| --- | --- |
| `slide_data.py` | 从结果文件取图表数字，写出 `data.json` 和 `layers_three.png` |
| `build_deck.js` | 用 pptxgenjs 生成幻灯片（主题、版式、原生图表） |

重新生成（需要 pptxgenjs 和 pptx 技能里的 `apply_theme.js`）：

```bash
python slides/slide_data.py
NODE_PATH=<pptxgenjs 所在目录> node slides/build_deck.js <apply_theme.js 路径>
```

直接在 PowerPoint 中修改也可以：颜色和字体在"设计 → 变体"的主题里统一修改；图表是原生图表，可以右键"编辑数据"。封面和讲稿中的 `Author Name`、`[name]` 需要换成你的信息。
