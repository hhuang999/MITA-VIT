# v13 submission review

The paper asks how the spatial assignment of value lengths and the choice of query affect a ViT readout. The contributions are controlled comparisons within frozen checkpoints, a geometry-aware interpretation of localisation scores, and a matched last-block intervention.

| Paragraph role | Evidence and scope |
| --- | --- |
| Motivation and contribution | Figure 1 illustrates the fast-weight head and the dependence of a measured gain on the localisation metric. |
| Positioning | The analysis builds on existing norm and context-mixing work. TokenTM, GlobEnc and value zeroing cover different modules, query sets and targets. |
| Method and registered evidence | Equation 1 defines the branch, Figure 2 compares three proxy-guided readings, and Table A.1 retains the original R1–R3 outcomes. |
| Spatial assignment | Table 1 uses 383 valid pet masks on the fixed supervised DeiT-S checkpoint. Own ranks exceed donor ranks and within-ring permutations. Unprojected value lengths slightly improve the mean score. |
| Behaviour | Original top-1 retention, exploratory probability/KL curves and a direct last-block value intervention have separately defined endpoints. Table 2 compares the identical layer-12 scores and images for classification sensitivity and masks. |
| Interpretation and limits | One external dataset and checkpoint bound the new forward experiments. The matched mask comparison and alternative geometry endpoint were selected after seeing the initial exploratory results. Appendix E records both. |

## Final checks against the rubric

- **Contribution.** The strongest finding is the opposite direction of the query-restriction effect on two targets on the same images. The claimed scope is the tested model and final CLS pathway.
- **Clarity.** Both sample sets and layer ranges are explicit. Figure 1b states its different axes and eligible cohorts. R1 separately names its deep DINO and shallow all-model tests.
- **Depth and soundness.** Per-image data, donor-pair resampling, original ClickMe batch resampling, a metric ceiling derivation and full-forward intervention checks accompany the mean effects. The old class-membership description was corrected against archived rows; Appendix E gives sensitivity on masks with both classes.
- **Literature.** All 33 baseline entries retain their reviewed metadata, including the version-sensitive MiTA record. New items were checked against the [TokenTM source](https://arxiv.org/abs/2403.14552), [Oxford dataset publication](https://intelligent-earth.ox.ac.uk/publication/355067/scopus) and [ACL Anthology paper](https://aclanthology.org/2023.eacl-main.245/). They substantiate token transformations, trimaps and value intervention, respectively.
- **Presentation and evaluation completeness.** Four main pages include both figures and both main tables. The 36 references occupy pages 5–6. Appendix A–F concludes with key code on page 18. Rendered pages, table signs and intervals, font embedding, citation keys and internal destinations were inspected. The PDF-specific verification record is in `verification.json`.
