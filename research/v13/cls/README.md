# Last-block CLS classification path: exploratory v13 result

The experiment asks whether a spatial fast-weight map that localizes an object also ranks patches by their immediate effect on the final classifier. The question, 128-image selection, intervention, five scores, and correlation endpoint were fixed in [protocol.md](protocol.md) before model scoring. [addendum_protocol.md](addendum_protocol.md) records a later, explicitly post-result comparison with object-mask endpoints before those additional values were computed. Neither study changes the v12 registered results.

## Design and reproduction

Use the exact frozen supervised DeiT-S model and Oxford-IIIT Pet **test** files already used by the spatial study. The 128 image names in [selected_names.txt](selected_names.txt) were selected from its fixed 384 by an independent SHA-256 ordering of `v13-cls-2026:` plus filename. No accuracy or mask filter affected this selection. Use the existing `vb` Python, timm dependency overlay, and data/cache under `build/paper-v13/spatial`. From the repository root:

```powershell
& 'D:/anaconda/envs/vb/python.exe' research/v13/cls/run_cls.py
& 'D:/anaconda/envs/vb/python.exe' research/v13/cls/analyze_endpoints.py
```

`run_cls.py` does one ordinary model forward per batch and vectorizes 196 final-block CLS value-path interventions per image. The original attention probabilities remain fixed. The baseline top-1 and runner-up are fixed from the original logits, and the endpoint is the absolute change in their logit margin. The script saves every per-source effect and all five ranking scores in `per_source.csv.gz`, one Spearman correlation per image and method in `per_image.csv`, and paired bootstrap results in `summary.csv` (10,000 draws, seed 2026). `metadata.json` records selection, source, protocol, script and checkpoint identifiers. Model scoring took 3.35 seconds after the existing local cache was available, on an RTX 4060 Laptop GPU; process startup and loading overhead are separate.

The vectorized no-intervention path reconstructs original logits with a maximum absolute difference of `4.77e-6`. For four fixed images and three fixed source patches each, a separate full-model forward with a last-block CLS value hook agrees with the vectorized deleted logits within `2.38e-6` (fixed-class margin within `1.91e-6`). The layer-12 all-query R0/C3 maps also agree with the repository reference capture to `1.83e-4` maximum. These checks support the numerical correctness of the intervention and score definitions.

## Direct classification-path endpoint

Per-image Spearman correlations compare the 196 patch scores with the absolute fixed-class margin change. Means and intervals are across the same 128 images; 95% percentile intervals come from paired image resampling.

| Layer-12 reading | Mean Spearman | 95% interval |
| --- | ---: | ---: |
| All-query R0 | 0.6965 | [0.6720, 0.7198] |
| All-query C3 | 0.7055 | [0.6813, 0.7287] |
| CLS attention | 0.7298 | [0.7123, 0.7468] |
| CLS summed-head magnitude | 0.7466 | [0.7301, 0.7626] |
| CLS net-term magnitude | 0.7558 | [0.7399, 0.7715] |

Paired differences from all-query C3 are +0.0243 [0.0094, 0.0403] for CLS attention, +0.0412 [0.0264, 0.0574] for CLS summed-head magnitude, and +0.0504 [0.0350, 0.0675] for CLS net-term magnitude. CLS net-term minus CLS summed-head magnitude is +0.00921 [0.00690, 0.01157]. The CLS net-term ranks higher than all-query C3 in 102/128 images and higher than CLS summed-head magnitude in 100/128. These results are expected to some extent because the intervention and the CLS-only scores use the same final-block path; they quantify the mismatch of an all-query spatial readout with this precise target, rather than validating a universal explanation score.

## Same-image endpoint alignment check

The post-result addendum reuses the saved five score vectors and loads the same images' Oxford trimaps. There was no new model forward. All 128 images have at least one valid object and background patch after preprocessing, so the three endpoints use **exactly the same images**. The source score definition and layer are also unchanged. `three_endpoint_per_image.csv` contains all per-image values; `three_endpoint_summary.csv` has the 3 × 5 method table and `three_endpoint_contrasts.csv` has the two requested paired comparisons. Every interval uses 10,000 shared image-bootstrap draws, seed 2026.

| Endpoint, mean [95% CI] | All-query R0 | All-query C3 | CLS attention | CLS summed-head | CLS net-term |
| --- | ---: | ---: | ---: | ---: | ---: |
| Margin Spearman | .6965 [.6720, .7198] | .7055 [.6813, .7287] | .7298 [.7123, .7468] | .7466 [.7301, .7626] | .7558 [.7399, .7715] |
| Same-ring object enrichment | .1169 [.1043, .1294] | .1274 [.1137, .1409] | .0813 [.0673, .0948] | .0946 [.0789, .1097] | .0983 [.0820, .1144] |
| Object/background AUROC | .7544 [.7389, .7696] | .7691 [.7530, .7847] | .6659 [.6382, .6921] | .6778 [.6496, .7044] | .6781 [.6484, .7063] |

| Endpoint | CLS summed-head − all-query C3, mean [95% CI] | CLS net-term − CLS summed-head, mean [95% CI] |
| --- | ---: | ---: |
| Margin Spearman | +0.04116 [0.02638, 0.05736] | +0.00921 [0.00690, 0.01157] |
| Same-ring enrichment | −0.03282 [−0.04304, −0.02327] | +0.00370 [0.00073, 0.00689] |
| AUROC | −0.09128 [−0.10953, −0.07402] | +0.00023 [−0.00308, 0.00332] |

The ranking reverses between the local classifier-path endpoint and the object-mask endpoints on this fixed sample: CLS-only maps align better with the final-block CLS intervention, whereas all-query C3 better localizes the annotated object. The small net-term increment over summed-head magnitude for margin Spearman is consistent with cross-head cancellation mattering for this local endpoint. The AUROC comparison between those two CLS maps is unresolved at this sample size. Absolute numbers across the three endpoints have different meanings and units; only within-endpoint contrasts are interpretable.

## Scope

The intervention changes one source patch's value **only for the CLS query in the final block**; it holds the attention softmax fixed and does not rerun earlier blocks or change other queries. It directly tests the chosen final-block path, while the original C3 map sums all queries over several layers for a spatial reading. This study therefore supports a clear distinction between spatial object localization and local final-class sensitivity, not a claim that C3 is causally faithful to the classifier. It uses one model and one class-narrow external test set. The object-mask comparison was motivated after the first result and should be reported as exploratory if included in the paper.
