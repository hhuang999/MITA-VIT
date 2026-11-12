# v9 editorial revision

## Story and section outline

1. Problem: distinguish the model from the reading rule in deep received-attention maps.
2. Framework: connect specificity, location and contribution assumptions to diagnostic proxies,
   corrections and matched controls using the established fast-weight representation.
3. Method: separate the exact algebraic representation from the interpretive assumptions;
   define centre-controlled metrics and preserve the original decision rule.
4. Evidence: report contribution weighting, failed corrections/combinations and model-side comparisons separately.
5. Interpretation: distinguish empirical outcomes, weak associations and untested causal explanations.
6. Conclusion: retain useful alignment gains and failed relocation as bounded findings.

## Paragraph roles and reverse outline

| Location | Role | Message and supporting material |
| --- | --- | --- |
| Abstract, opening | Problem | Deep maps may emphasise background; changing the reading is a testable alternative. |
| Abstract, framework | Method | Three assumptions, three proxies/corrections/controls; three frozen classifiers. |
| Abstract, results | Evidence | Four supervised alignment gains, three control wins, no deletion-margin gain; R1/R2 outcomes. |
| Abstract, closing | Limitation | Diagnostic prediction is not established by these checkpoints. |
| Introduction, paragraph 1 | Opening/challenge | Separate model-side changes from reading-side changes. |
| Introduction, paragraph 2 | Method/positioning | Prior alternative readings lead to a controlled test of assumptions. |
| Introduction, contribution list | Contribution | Framework, conditions/limits of an existing reading, and failed relocation/combination. |
| Related work, paragraph 1 | Positioning | Distinguish established norm weighting from the present controlled evaluation. |
| Related work, paragraph 2 | Motivation | Token behaviour and fast weights motivate the three assumptions without establishing them. |
| Method 3.1 | Technical definition | Fixed-image head representation is exact with a coupled softmax hidden activation. |
| Method 3.2 | Design/limitation | Proxies need validation; successful correction requires both baseline and control comparisons. |
| Method 3.3 | Evaluation | Explain what spatial controls remove and what the registered verdict does not prove. |
| Experiments 4.4, paragraph 1 | Evidence | C3 improves alignment in DeiT-S/B, wins 3/4 control comparisons and does not exceed deletion margin. |
| Experiments 4.4, paragraph 2 | Negative evidence | C1 lacks a joint benefit, C2 degrades maps and R2 is 1/4. |
| Experiments 4.4, paragraph 3 | Comparison/limitation | R3 holds in the tested cells; MiTA is not a controlled causal comparison. |
| Experiments 4.5 | Interpretation | Shuffling is informative, dose correlations weak, and the cause of relocation failure unresolved. |
| Discussion | Limitation/implication | Alignment differs from faithfulness; model/readout generalisation is untested. |

## Major claim/evidence map

| Claim | Evidence | Status |
| --- | --- | --- |
| The framework specifies three assumptions and corresponding tests. | Method/Table 1 definitions; established fast-weight and norm-weighting sources credited. | Supported as a framework, not a validated diagnostic predictor. |
| C3 raises enrichment by 0.038/0.065 and ClickMe by 0.033/0.058 in DeiT-S/B. | `results/correct_tests.csv`, C3 - R0, enrichment/ClickMe. | Supported. |
| C3 exceeds its control on 3/4 supervised alignment cells. | Same CSV, C3 - stand_in_no_high; DeiT-B ClickMe is comparable. | Supported; exception explicit. |
| C3 has no improvement beyond the deletion margin or in DINO. | Same CSV and `results/negative_controls.csv`; deletion margin 0.05, others 0.02. | Supported under the registered decision criterion. |
| Relocation lowers all nine deep-layer model/metric cells. | `results/correct_tests.csv`, C2 - R0. | Supported. |
| Combining corrections improves only 1/4 registered supervised cells. | Same CSV, R_all - R0; only DeiT-B ClickMe is higher. | Supported; registered label remains partly holds. |
| Index decoding directly establishes token-content origin. | Probe labels identify original index, not input-content provenance. | Unsupported; no such claim retained. |
| Proxy/target mismatch causes relocation failure. | Local-error and MLP checks narrow explanations without isolating cause. | Needs evidence; retained as an untested explanation. |
| Diagnostic signals generally predict correction benefit. | Post-hoc correspondence for C3 in three checkpoints only. | Needs evidence; explicitly unestablished. |
| Reading-side changes generally outperform model-side changes. | Register is one tested DeiT-S intervention; MiTA differs in training/preprocessing. | Unsupported; removed as a general conclusion. |

## Five-dimension adversarial self-review

| Dimension | Question | Answer and resolution |
| --- | --- | --- |
| Contribution | What is new beyond existing fast weights and norm weighting? | A controlled diagnostic framework and bounded comparative findings. Both ingredients are credited; no new weighting method claimed. |
| Clarity | Can readers distinguish representations, assumptions, proxies and outcomes? | Their roles are separated in Method; the multi-topic results paragraph is split into three. |
| Experimental strength | Are gains consistent and meaningful under the original rules? | Alignment gains in DeiT-S/B pass the criterion, but only 3/4 beat the matched control; deletion and DINO do not. These limits are in the abstract and main text. |
| Evaluation completeness | Are missing evaluations hidden by the presentation? | Larger/register-trained models, CLS maps and repeated training seeds remain untested and are explicit limitations. No new experiment is implied. |
| Method soundness | Do the proxies justify the corrections or prove the failure mechanism? | No. The exact representation is separated from heuristic interpretation; index decoding/provenance and the untested failure explanation are distinguished. |

Clarity, paragraph flow, terminology and claim/evidence alignment were checked against this outline.
Unresolved research questions remain limitations, rather than new experiments or changes to registration.

## Presentation and provenance

- Figure 1a redraws the same fixed-image attention representation as a legible vector schematic.
- Figure 1b displays the supplied C3 - R0 and C3 - control means/intervals for two metrics and three models.
- Table 2 retains the original rounded means and baseline verdicts, adds diamonds for original
  higher-than-control verdicts, and bolds only cells higher than both baseline and control.
- Original fixed-seed examples are retained in Appendix D with the original selection rule.
- Appendix F restores the historical parameter-randomisation summary from `archive/results/sanity.csv`;
  `evidence/sanity.csv` is an identical active copy, not a new experiment.
- Appendix A corrects the valid-pixel mask definition and documents actual preprocessing,
  other-image pairing and the interpretation of the unchanged statistical categories.
- Original v8 manuscript sources/PDF, authored figures/tables, all result CSVs and the registration record are preserved.

The legacy table generators received one portability fix: explicit UTF-8/LF output, matching the existing
`.gitattributes` rule and the supplied LaTeX bytes on Windows. Their numerical logic and authored outputs
are unchanged. All four generated legacy LaTeX files were checked byte for byte against the originals.
The v9 presentation builder also writes explicit LF newlines.

The initial full CPU check used the existing Conda environment (NumPy 2.3.5/Pandas 2.3.3) and exposed
enrichment dose--response rank differences up to 0.000113. Using the project's declared NumPy 2.5.4/Pandas
3.1.0 in a workspace-local dependency directory restores exact agreement for both supervised C3--R0
Spearman point estimates. No reference result, statistical rule, margin or inference output was changed.

## Final verification

- Main content ends on page 4; references begin on page 5; total length is 12 pages.
- All 12 rendered pages were inspected. Citations and cross-references resolve, fonts are embedded,
  and no overfull boxes, clipped figures or overlapping content were found.
- All 86 main-table numeric cells match v8. The 7 control diamonds and 3 joint-benefit bold cells
  use the existing verdicts. Figure 1b's 12 means/intervals match their supplied CSV rows.
- The original-artifact integrity audit passes. The historical sanity summary copy is byte-identical.
- The final full CPU suite passes: **106 passed** (325.08 seconds), including reconstruction of all
  19 CSV summaries, immutable original artifacts, and byte-identical legacy generated LaTeX tables.
- NumPy/Pandas were aligned with the declared project versions in `build/paper-v9/python-deps`;
  the user's Conda environment and the scientific dependency declarations were not changed.
- `verification.json` records the delivered PDF checksum and detailed checks.
