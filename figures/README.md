# Repository figures used by the paper

Each figure is supplied as PDF and 300-dpi PNG.

| File stem | Paper use | Input | Size (inches) |
| --- | --- | --- | --- |
| `schematic_fast_weight_network` | Figure 1 | Static drawing in `scripts/make_figures.py` | 2.6 x 1.9 |
| `execution_k_sweep` | Figure B.1 | `results/k_sweep.csv` | 2.6 x 1.9 |
| `gallery_layers9_12` | Figure D.1 | `outputs/table1/clickme/deit/gallery_*.npz` | 2.8 x 1.9 |

Figure 2 is `paper/figures_own/diagnosis.pdf`, generated from `results/diagnose.csv`.
Run `bash reproduce/from_outputs.sh` in a copy to rebuild all paper figures and keep the supplied originals.
Figure numbers, panel labels and captions are set in the paper.
