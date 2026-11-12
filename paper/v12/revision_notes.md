# v12 framework-focused rewrite

## Approved story

The core contribution is an analytical framework that connects an established fast-weight
representation to three assumptions behind reading internal attention as spatial importance.
Diagnostic proxy → correction → matched control operationalises each question. Existing
norm weighting is a component under evaluation, not a claimed new algorithm. Experiments
provide bounded insights and negative results; a useful correction does not by itself
validate the proposed interpretation or establish a causal mechanism.

Title: **Interpreting ViT Attention Maps Through a Fast-Weight Lens**.
The 113-word abstract covers motivation, representation, framework, evaluation design,
qualitative positive/negative findings and significance. The deletion-margin finding remains
because it limits the alignment claim; detailed effect sizes appear in the results.

## Outline and paragraph roles

1. **Introduction:** identify the translation from contextual token interactions to image
   evidence as the problem; use the preserved homepage figure to connect representation
   and observed effects; state framework, controlled tests and bounded empirical insights.
2. **Related work:** compare three themes. Fast-weight work supplies the representation;
   attribution/norm/token-correspondence work clarifies what a local score measures;
   explanation evaluation motivates spatial controls and separate outcome metrics.
3. **Methodology:** define the exact affine attention branch and received-attention map;
   give each assumption its own question, proxy, correction and control; define metrics
   and unchanged statistical/practical decision rules.
4. **Experiments:** state data/checkpoints and registered predictions; interpret diagnostics
   as distinct internal properties; separate contribution weighting, other corrections
   and model-side references; distinguish observed benefit from explanation validity.
5. **Discussion and conclusion:** bound the framework to tested checkpoints/readings;
   retain overlap, conditional uncertainty, GAE and generalisation limitations; conclude
   with the positive/negative findings and separate alignment/deletion evaluation.
6. **Appendices A–E:** retain experimental/protocol detail, additional results, unresolved
   relocation tests and fixed-seed examples. Move the complete stored parameter-randomisation
   protocol/table into Additional results (B.1/Table B.3). Key code (E) is the final content.

## Course-rubric alignment

| Criterion | Editorial response |
| --- | --- |
| Contribution | Explain the analytical perspective and controlled testing contribution; attribute the established representation and norm component. |
| Clarity | Use a short abstract, stable assumption names, explicit questions and a consistent proxy/correction/control structure. |
| Depth | Interpret negative results, separate diagnostics from outcomes and avoid resolving an unsupported failure mechanism. |
| Literature review | Compare three research themes and explain the framework's relation to them; retain all 33 references. |
| Formatting | Use the original NeurIPS preprint typography/geometry; four main pages include every main figure/table; references and appendices remain complete. |

The four-page main text carries the framework, definitions, principal comparisons, results,
registered decisions and limitations. Appendix details support it without carrying the core
argument alone. The rubric suggests numerical abstract results but does not require them;
the approved short abstract emphasises the framework and qualitative findings instead.

## Claim/evidence map

| Claim | Existing evidence | Interpretation |
| --- | --- | --- |
| A head has an image-conditioned two-layer representation. | Affine attention identity in Eq. 1 and Appendix A; established fast-weight/MiTA sources. | Representation is established; its use to organise reading assumptions is the contribution. |
| Specificity, spatial correspondence and output contribution are distinct questions. | Figure 2 proxies, C1–C3 and matched controls; Appendix A definitions. | Operational framework, not a proof that proxies diagnose causal failure. |
| C3 raises enrichment by 0.038/0.065 and ClickMe by 0.033/0.058. | `results/correct_tests.csv`, C3–R0; unchanged main/paired tables. | Alignment gain for frozen DeiT-S/B. |
| C3 beats its control in three supervised alignment comparisons. | C3–stand_in_no_high contrasts. | DeiT-B ClickMe remains comparable, preserving the exception. |
| C3 deletion gains remain below 0.05; DINO changes remain comparable. | Existing paired differences and R1 negative controls. | Separate outcomes and practical margins; comparable is not equivalence. |
| Deep relocation is lower in all nine comparisons. | C2–R0; unchanged main/paired tables. | Index-based relocation fails here; failure cause is unresolved. |
| The combined correction does not consistently improve alignment. | R_all–R0: one higher, one comparable, two lower supervised cells. | Registered R2 partly holds; category unchanged. |
| R3 holds within the supplied model-side comparisons. | Register gains 0.012/0.013; MiTA reference. | Training/preprocessing differences limit broader comparisons. |
| Assigned norm weights outperform shuffled weights. | Stored post-hoc shuffle results, Appendix B. | Does not establish diagnostic prediction; dose–gain Spearman remains 0.045–0.158. |
| Randomisation changes the five readings. | Preserved `evidence/sanity.csv`, Table B.3. | Necessary parameter sensitivity, not proof of prediction faithfulness; protocol retained. |

## Five-dimension adversarial self-review

| Dimension | Review question | Resolution |
| --- | --- | --- |
| Contribution | Is novelty understandable without mistaking prior components for new algorithms? | Title, abstract, introduction and related work locate the contribution in the analytical framework and controlled evaluation. |
| Writing clarity | Does each paragraph have a distinct role and support the same story? | Motivation, three assumptions, operational tests, observed effects and interpretation limits follow in order; repetitive numerical diagnostics remain in Appendix B. |
| Experimental strength | Are unsupported generalisations removed while findings remain visible? | DeiT alignment gains, deletion limit, DINO negative control, C2 failures, mixed combination and confounded MiTA reference remain explicit. |
| Evaluation completeness | Are main findings and reproduction details available? | Main text defines outcomes/decision rules; all original tables, Appendix A settings, sanity protocol, supplementary tests, galleries and key code remain included. |
| Method soundness | Are activation/contribution, token slot/provenance and association/causation distinguished? | Exact branch formula, local magnitude scope, probe-index target and post-hoc interpretation are stated. No general diagnostic predictor or established relocation mechanism is claimed. |

## Preservation and verification

- All three homepage/Figure 2 vector assets are byte-identical to v11.
- Main, paired and sanity table sources are identical to v11; 86 main-table measurements,
  seven control diamonds and three bold cells are retained. Paired values/verdicts also
  match the original scientific table.
- The complete v11 bibliography is copied unchanged. Pending Ramsauer author-order and
  MiTA version/author questions remain pending; no external verification is claimed here.
- Scientific outputs, implementation fingerprints and registration pass the strict CPU
  integrity audit. No model/probe training or inference was run for this revision.
- Final PDF: four main pages, references on 5–6, appendices on 7–15. Every page was
  rendered and visually reviewed. Fonts, bounds, cross-references and internal destinations
  are checked by `verify_build.py`; `verification.json` binds checks to the PDF hash.
