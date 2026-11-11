# Paper v8 figures

| Figure | Distributed source | Rebuilt output (under build/rebuilt/) |
| --- | --- | --- |
| 1a | `schematic_fast_weight_network.{pdf,png}` | `figures/schematic_fast_weight_network.{pdf,png}` |
| 1b | `paper/figures_own/example_maps.pdf`, `figures/example_maps.png` | `figures/example_maps.{pdf,png}` |
| 2 | `paper/figures_own/diagnosis.pdf` | `paper/figures_own/diagnosis.pdf` |
| B.1 | `execution_k_sweep.{pdf,png}` | `figures/execution_k_sweep.{pdf,png}` |
| D.1 | `paper/figures_own/example_maps_more.pdf`, `figures/example_maps_more.png` | `figures/example_maps_more.{pdf,png}` |

The original `gallery_layers9_12.{pdf,png}` is retained as a legacy layout of the same two Figure 1b examples.
The full seeded 24-image DeiT-S ClickMe gallery is sorted by stored index: 47/287 for Figure 1b;
712/1308/3048/4691 for Figure D.1. The strict builder fails on missing, duplicate or incomplete arrays.
`gallery_selection.json` records all indices and both selections; no image is chosen for visual quality.
Maps sum layers 9-12 and are individually min-max normalised. PDF and 300-dpi PNG are generated with labels
at least 7 pt. `bash reproduce/from_outputs.sh` writes into a fresh directory, keeping original figures unchanged.
