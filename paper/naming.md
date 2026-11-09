# Naming used in figures, tables and text

Use these names in the paper. Internal codes (R0, C1–C3, M1–M3, P1–P7, X1–X7, D2, A/, S/) must not appear in figures or tables. The evidence dossier and `results/` still use the internal codes, so this table is also the key for reading them.

| Internal code (in `results/` and the dossier) | Name in the paper |
| --- | --- |
| R0 | received attention |
| C3 | contribution weighting |
| C2 | probe relocation |
| C1 | mean-map subtraction |
| stand_in_no_high / no_high (stand-in of C3) | zero high-norm tokens |
| stand_in_blur / blur (stand-in of C2) | matched blur |
| stand_in_centre / centre_sub (stand-in of C1) | centre-Gaussian subtraction |
| centre / mean_map / random / gae | centre Gaussian / mean map / random / GAE |
| register, +register | + test-time register |
| mita, MiTA, top-k training | top-k training (MiTA) |
| R_all (C3 → C2 → C1, pre-registered) | all three corrections (order A) |
| R_all_alt (C3 → C1 → C2) | all three corrections (order B) |
| M1 | share of variance explained by the mean map |
| M2 | position-probe accuracy |
| M3 | share of received attention on high-norm tokens |
| quality (per-layer panel) | image-specific ClickMe alignment of received attention |
| reading sets / random sets / no attention | top-k reading sets / random sets of the same size / attention removed |
| deit / deit_b / dino | DeiT-S / DeiT-B / DINO ViT-S |
| srg | deletion faithfulness (SRG) |
| enrichment | same-ring object enrichment |
| clickme | image-specific ClickMe alignment |
| D2 | probability discount |
| D2_shuffle | shuffled probability discount |
| C3D2 | contribution weighting + probability discount |
| C2MLP | probe relocation (MLP probe) |
| C2conf | probe relocation (confident tokens only) |
| C1C3 / C2C3 / C1C2 | contribution weighting + mean-map subtraction / contribution weighting + probe relocation / probe relocation + mean-map subtraction |
| C3v, ‖v‖ variant | value-norm weighting |
| C3 shuffled / head-mean | shuffled contribution weights / head-mean contribution weights |
| P1 | reading sets beat random sets (execution) |
| P2 | premises fail with depth (diagnosis) |
| P3 | all three corrections beat received attention (main prediction) |
| P4 (P4a / P4b) | negative controls (contribution weighting on DINO ViT-S / probe relocation at layers 1–4) |
| P5 | last-layer recovery (descriptive) |
| P6 | model-side variants not higher |
| P7 | primary comparisons (no direction) |
