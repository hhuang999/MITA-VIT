# v11 review implementation

## Story and reverse outline

1. Abstract and Introduction (problem/contribution): frozen-model reading changes; retain numerical
   positive and negative results. The full contribution list now fits on the homepage with Figure 1.
2. Related Work (positioning): distinguish norm-weighted components from whole algorithms, token
   decodability from provenance, and CNN centre bias from ViT evidence. Add verified GlobEnc/ALTI context.
3. Method (definition/design): affine Q/K/V, output bias, scaled input weights, coupled softmax;
   three proxy/correction/control definitions and unchanged decision rule.
4. Experiments (evidence): distinguish primary, descriptive and registered-negative-control comparisons;
   preserve C3 gains, all nine C2 losses and R2's one higher/one comparable/two lower outcome.
5. Discussion/Conclusion (scope): weak dose correlations, unresolved relocation mechanism, fixed
   checkpoints/probes/pairing, and limits on generalisation.
6. Appendices A–F (reproduction/support): correct C1 calibration and C2 displacement matching;
   explain SRG-style top-1 adaptation, GAE, linear/MLP probes, shuffle averaging and globality criteria.

## Implemented review items

- Preserve all three homepage/Figure 2 vector assets byte for byte; captions are clarified.
- Preserve every main-table value/verdict/marker; enlarge Appendix B.2 interval text to 8 pt and increase row spacing.
- Enlarge the stored execution plot from 2.1 to 3.25 inches without changing data or drawing content.
- Correct the C1 Gaussian control to calibration after layer summation and C2 to uniform token-displacement matching.
- State affine biases, attention-branch boundary and per-head norm aggregation; avoid claiming identity with prior whole methods.
- Label deletion as an SRG-style top-1-preservation adaptation, with existing budgets and integration convention.
- Replace harmless/must-match wording with observed registered/comparative outcomes.
- Disclose ClickMe split uncertainty, fixed within-batch pairing and conditional bootstrap uncertainty.
- Add existing GAE/linear-probe/MLP/globality/shuffle details from code/configs; no new results generated.
- Retain all six appendix sections and fixed-seed gallery; keep gallery, code and sanity blocks intact across page breaks.
- Use an isolated v11 bibliography, add confirmed pages/DOIs and GlobEnc/ALTI, and protect title capitalisation.
- Preserve the entire Ramsauer and MiTA entries unchanged; pending author-order/version questions are not resolved by guesswork.

## Claim/evidence map

| Claim | Evidence | Status |
| --- | --- | --- |
| C3 alignment gains are 0.038/0.065 and 0.033/0.058. | Existing `results/correct_tests.csv`, C3–R0. | Supported. |
| C3 beats its control in 3/4 supervised alignment comparisons. | Same CSV, C3–stand_in_no_high; B ClickMe is comparable. | Supported; exception retained. |
| Deletion gains remain below 0.05 and DINO changes below 0.02. | Existing paired contrasts and negative controls. | Supported under unchanged margins, not equivalence. |
| Deep C2 lowers nine model/metric comparisons. | Existing C2–R0 verdicts. | Supported. |
| R2 partly holds with one higher, one comparable, two lower. | Existing R_all–R0 on supervised alignment. | Supported; historical category unchanged. |
| DINO high-norm attention is negligible overall and zero in deep ImageNet-S layers. | Existing `results/diagnose.csv`; maximum mean 0.02455%. | Supported with dataset/layer scope. |
| Diagnostic proxies generally predict benefit or explain C2 causally. | Only limited post-hoc correspondence and tested alternatives. | Unestablished; no stronger claim. |

## Five-dimension adversarial self-review

| Dimension | Question | Resolution |
| --- | --- | --- |
| Contribution | Is norm weighting presented as a new method? | No; the protocol and bounded findings are the contribution. |
| Clarity | Do formulas and controls match the existing implementation? | Biases, scaling, C1 summed-map calibration, C2 unweighted displacement and SRG adaptation are explicit. |
| Experimental strength | Are limits and negative results retained? | All original values/verdicts remain; R2 losses, B ClickMe control exception and weak dose association remain explicit. |
| Evaluation completeness | Can the PDF reproduce reference and supplementary settings? | Existing GAE, probe, MLP, budgets, seeds, aggregation and provenance paths are documented. Unknown historical fields remain unknown. |
| Method soundness | Are comparable and post-hoc results overinterpreted? | No equivalence claim, no general diagnosis predictor or established failure mechanism; statistical uncertainty is conditional. |

## Delivery checks

Four main pages, two reference pages and eight appendix pages. All 14 pages visually reviewed;
citations/links resolved and fonts embedded. Main table's 86 values, 7 diamonds and 3 bold cells,
all paired measurements/verdicts, historical sanity summary, registration and immutable research
artifacts verified. No GPU runs, probe training, inference regeneration or scientific-code changes.
`verification.json` records the exact final PDF and checks.
