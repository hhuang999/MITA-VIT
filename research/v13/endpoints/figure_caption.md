# Proposed appendix caption

**Figure A.X | Continuous deletion contrast between contribution weighting and the no-high control.** (a) Difference in target-class probability separation, `[(p_bottom − p_top)_C3 − (p_bottom − p_top)_no-high]`, where the target is the original top-1 class. (b) Difference in `KL(p_original || p_reduced)` separation, `[(KL_top − KL_bottom)_C3 − (KL_top − KL_bottom)_no-high]`. In both panels, positive values indicate stronger top-versus-bottom deletion separation for C3. The no-high control sets R0 scores of patches whose block-input norm exceeds three times the image median to zero, per layer. Points show paired means for the fixed 2,000 ImageNet-S images; bars show pointwise 95% percentile intervals from 1,000 paired image-bootstrap resamples (seed 0). All nine original positive budgets are plotted. The zero-budget difference is exactly zero and is shown without an interval. Horizontal positions follow `log2(1 + budget)` and are slightly offset by model to reveal overlapping intervals. This exploratory analysis has no multiplicity-adjusted single-budget tests or registered practical threshold.

## What the figure can support

The combined curve area and corresponding intervals in `deletion_areas.csv` support a positive full-range C3–no-high separation in DeiT-S and DeiT-B on both continuous endpoints. The figure shows that the effect varies with budget: small-budget KL differences can be negative, and DINO has no clear integrated gain. The figure does not identify a causal mechanism for the ranking difference or establish generalization to a different perturbation family.

## Rendering audit

The figure is 5.5 × 2.55 inches (396 × 183.6 pt), vector PDF with editable text and no raster images. The final PDF passes the panel-alignment gate (1.5 pt), rendered glyph floor (minimum 6.4 pt), and PDF collision audit (zero failures or warnings). The source preflight has four nonblocking warnings: the requested 5.5-inch width is between common Nature column widths, the PNG is a 300-dpi preview and no TIFF was requested, and the log-axis heuristic does not recognise the explicit `1 + budget` guard. The manuscript should use the vector PDF.
