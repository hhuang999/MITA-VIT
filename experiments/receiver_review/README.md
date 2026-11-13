# Receiver comparison in DeiT-B

This exploratory DeiT-B study repeats the original DeiT-S receiver
comparison on the same 128 Oxford-IIIT Pet images. The protocol was fixed
before computing DeiT-B scores. The original registration and DeiT-S results
are unchanged. The exact sample, all 25,088 patch interventions, five maps,
validation checks and metadata are saved under `deit_b/`.

The counterexample changes the interpretation: CLS-only improves classifier
effect ranking in both checkpoints, but improves mask AUROC in DeiT-B while
reducing it in DeiT-S. The evidence does not support a general
localisation–classifier-sensitivity trade-off.

| CLS head sum minus all-query value weighting | DeiT-S | DeiT-B |
| --- | --- | --- |
| Margin Spearman | +0.041 [0.026, 0.057] | +0.082 [0.068, 0.096] |
| Mask AUROC | −0.091 [−0.110, −0.074] | +0.024 [0.013, 0.036] |
| Same-ring enrichment | −0.033 [−0.043, −0.023] | +0.005 [−0.005, 0.013] |

Intervals use 10,000 paired image-bootstrap resamples, seed 2026, with no
registered practical margin. All 128 images are eligible for every endpoint.
These are checkpoint comparisons on a shared cohort, not independent sample
replications. The CLS intervention structurally favours CLS rankings.

From the repository root, with the CUDA environment, Oxford images/trimaps and
pinned timm setup described in `experiments/README.md`:

```sh
python -m experiments.cls.run_cls --model base --protocol experiments/receiver_review/protocol.md --output-dir outputs/rerun/receiver-review-deit-b
python -m experiments.cls.analyze_endpoints --input-dir outputs/rerun/receiver-review-deit-b --output-dir outputs/rerun/receiver-review-deit-b --protocol experiments/receiver_review/protocol.md
python -m experiments.receiver_review.audit
```

The audit reads the preserved study, rechecks sample identity and saved hashes,
recomputes every per-image ranking correlation from the saved interventions,
and reproduces all six paired interval estimates. To reproduce mask extraction
from images, run the second command against the preserved `deit_b/` input with
a fresh output directory. The independent full-forward check has maximum
logit/margin errors of 2.86e-6/1.91e-6, below the specified 0.002 tolerance.
