# v10 editorial integration

## Outline and paragraph roles

1. Abstract/introduction: retain the v9 question, contributions and bounded experimental conclusions.
2. Method 3.1 (definition): retain the exact fixed-image fast-weight representation and coupled softmax;
   condense prose without changing the attention operation or layer aggregation.
3. Method 3.2 (design): map Figure 2a–c diagnostics to d–f effects, then define C1/C2/C3 and their
   controls in prose. Preserve the distinction between index decoding and content provenance,
   and between contribution magnitude and subsequent model computation.
4. Method 3.3 (evaluation): retain all metrics, bootstrap procedure, practical margins and verdict rules.
   The entire subsection fits on page 2.
5. Experiments (evidence): Figure 2 supplies the diagnostic-to-correction comparison; the full
   numerical table remains as Table 1. Contribution, other corrections and model-side results retain
   separate paragraphs. Discussion and conclusion retain their original scope.
6. Appendix (detail): preserve A–F and all numerical tables. Redirect the former Figure 2d ClickMe
   citation to Table B.1, where the per-layer ClickMe values remain available.

## Claim/evidence map

| Claim | Evidence | Status |
| --- | --- | --- |
| Three proxies motivate three controlled corrections. | Method 3.2 and Figure 2 columns; original diagnosis and paired-effect CSVs. | Supported as a framework. |
| Norm weighting improves DeiT-S/B alignment, with three of four control wins. | Figure 2f, Table 1, Table B.2; original C3 comparisons. | Supported; DeiT-B ClickMe control exception retained. |
| Probe relocation fails in all nine deep-layer cells. | Figure 2e for alignment, Table 1/B.2 including deletion. | Supported. |
| Mean subtraction lacks consistent joint baseline/control benefit. | Figure 2d and Table B.2. | Supported. |
| A stronger diagnostic generally predicts a larger correction gain. | Only post-hoc correspondence and weak dose associations exist. | Unestablished; no strengthened claim. |
| The shaded practical-margin range establishes equivalence. | Registered higher/lower/comparable rule, unchanged in Method 3.3 and Appendix A. | Unsupported; shading described only as the practical margin. |

## Five-dimension self-review

| Dimension | Review question | Resolution |
| --- | --- | --- |
| Contribution | Does integration imply a new attribution method? | Existing norm weighting remains credited; contribution is controlled evaluation. |
| Clarity | Are definitions lost when conceptual Table 1 is removed? | All three formulas and matched controls move into Method 3.2; panels are referenced explicitly. |
| Experimental strength | Does the figure change outcomes or imply new inference? | Approved vector asset is copied byte for byte; all 108 diagnostic and 36 paired-effect source rows audited. |
| Evaluation completeness | Are deletion and model-side comparisons omitted? | Full original numerical table stays in the main text; Appendix B retains intervals for every metric. |
| Method soundness | Is the diagnosis–benefit link presented as registered or causal? | Caption and results label it post hoc; probe/provenance and cancellation limits remain explicit. |

## Final checks

- Four main pages, references on pages 5–6, appendices A–F on pages 7–12.
- Original font sizes, margins and NeurIPS preprint style retained; Figure 2 used at its native 5.5-inch width.
- All 12 rendered pages visually reviewed. No clipping, overlapping content or unresolved references.
- Homepage Figure 1 and its assets preserved byte for byte. Approved Candidate 2 preserved byte for byte.
- Main table retains all 86 numerical cells, original verdicts, seven control markers and three joint-benefit bold cells.
- Original-result integrity audit and figure source/geometry/text/collision checks pass.
- Registration, original experiment outputs, original v8/v9 artifacts and all scientific definitions remain unchanged.
- This editorial-only revision uses integrity and rendering checks; the prior v9 full CPU suite had 106 passing tests.
