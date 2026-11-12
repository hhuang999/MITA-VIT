# Candidate figure contract

These are review candidates. The accepted v9 source and PDF remain unchanged.
Target: the existing NeurIPS preprint course report, with four main-content pages.
The skill supplies scientific-figure design guidance; no Nature submission is implied.
Backend: Python/Matplotlib, reusing the established repository workflow and saved preference.

## Candidate 1: benefits and limits of norm weighting

Question: what does contribution weighting improve, and what survives a matched control?

Conclusion: contribution weighting improves localisation/human alignment in DeiT-S/B;
three of four supervised improvements exceed zeroing high-norm tokens, while deletion
and DINO do not meet the registered practical-improvement criterion.

Archetype: quantitative grid; three complementary evidence roles.

- a: primary evidence, C3 minus received attention on enrichment and ClickMe in all three models.
- b: specificity/control, C3 minus zero-high-norm control on the same model/metric cells.
- c: orthogonal boundary, C3 minus received attention on deletion in all three models.

Use the original 15 paired means, 95% intervals and verdicts. Do not recompute statistics.
Filled markers denote the existing higher/lower category; open markers denote comparable.
Grey bands show the numerical practical-margin range, not an equivalence conclusion.
Proposed placement: replace the current first figure; the algebra remains in Method.

## Candidate 2: diagnostic signal versus intervention outcome

Question: does a changing diagnostic justify its corresponding map correction?

Conclusion: proxy signals and correction outcomes can diverge. Declining index decodability
does not yield useful relocation; contribution weighting has a more favourable, bounded pattern.
No causal explanation of relocation failure or general predictive validity is claimed.

Archetype: quantitative grid; each column is one reading assumption.

- a--c: diagnostic evidence for specificity, location and contribution across all 12 layers.
- d--f: corresponding corrections, each against received attention and its own matched control.

Use 108 original diagnostic rows (12 layers x 3 models x 3 proxies) and 36 original paired
comparison rows (3 corrections x 3 models x 2 centre-controlled metrics x 2 references).
The comparison with deletion is supplied by Candidate 1 and the unchanged main table.
Proposed placement: replace the combination of conceptual Table 1 and diagnostic Figure 2.
Correction/control definitions displaced from Table 1 need a concise Method paragraph.

## Data mapping and exclusions

- Diagnostic source: panel/variant/layer/model identify the proxy; mean/low/high are the original
  estimates and intervals. Multiply by 100 only to display percentages. n remains the source n.
- Paired source: a/b/model/metric identify a contrast; mean/low/high/verdict are displayed directly.
  Deduplicate by the exact test label rather than mixing the duplicated DINO negative-control rows.
- Source diagnostic rows: 252 total. Candidate 2 selects M1/R0, M2/probe and M3/R0 (108 rows).
  Exclude 108 alternative-reading M1 rows and 36 uncorrected-quality rows because the question is
  baseline diagnostic-to-correction correspondence, not diagnostic re-estimation or another quality metric.
- Source paired rows: 99 total. Select the contrasts specified above; R_all, model-side variants,
  GAE and the full numerical context remain in the unchanged main results table/appendix.
- No image/observation is sampled, removed or regenerated. All three models are included.

## Export and inference contract

- Natural placement width: 5.5 inches (139.7 mm); white background; editable PDF/SVG text.
- Candidate 1 height: 1.85 inches; Candidate 2: 2.85 inches, after final-size layout review.
- Text: 6.4--8.5 pt, with every exported glyph at least 5 pt. The first figure uses at least 7 pt;
  the denser second figure uses 6.4 pt legends and 6.5 pt tick/axis labels.
- Same model colours and shapes throughout; reference type also has a non-colour encoding.
- Uncertainty: original paired image-bootstrap 95% intervals, 1,000 resamples; not model-seed error bars.
- Practical margins: 0.02 for enrichment/ClickMe, 0.05 for deletion. Preserve original verdicts.
- No multiplicity correction is added; the practical margin is not described as multiplicity control.
- Enrichment n: 12,150 (DeiT-S/B), 12,144 (DINO); ClickMe n: 17,408; deletion n: 2,000.
- Frozen checkpoints, one training realization each; original preprocessing and splits remain unchanged.
- The diagnostic--benefit interpretation is post hoc, although its component comparisons were fixed.
- Export PDF, editable SVG, 300-dpi PNG preview, selected source rows and input hashes.
- Call the render-time alignment gate at 1.5 pt tolerance; run source, PDF text and collision audits.
  Fix all FAIL findings, review every WARN and inspect all panels at their final physical size.

## Layout proof

Build a separate placement-only manuscript copy using the candidate figures. Retain the main results
table and original font/page geometry. Verify that the conclusion still ends on page 4; do not overwrite
the v9 manuscript. The preview is a placement demonstration awaiting the author's decision, not adoption.
