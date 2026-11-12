# Manuscript

[How Should We Read ViT Attention Maps? A Fast-Weight Perspective](fast-weight-attention-maps.pdf) contains four pages of main text, the bibliography, and the complete appendix. The AI Use Statement follows the final code excerpt.

The LaTeX entry point is [main.tex](main.tex). [appendix.tex](appendix.tex) contains the registered rules, additional results, exploratory protocols and key code. [figures/](figures/) holds the six included figures; [tables/](tables/) holds the included tabular material. The [repository README](../README.md) maps each display to its source data.

Install the project requirements and [Tectonic](https://tectonic-typesetting.github.io/en-US/). From the repository root:

```sh
python scripts/display/build.py
python paper/build.py
```

The build script uses the official [NeurIPS preprint style archive](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip) and checks the style file's SHA-256. Tectonic downloads TeX resources on first use. To use an offline style, pass `--style path/to/neurips_2025.sty`; a populated Tectonic cache is also needed. `--tectonic` accepts an explicit executable path.

The compiled result is `build/paper/fast-weight-attention-maps.pdf`. The script checks the four-page main text, 36 numbered references, embedded fonts, internal links and placement of the final disclosure. It leaves the distributed PDF in `paper/` intact.
