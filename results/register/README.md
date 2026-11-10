# Register evidence used in Appendix A

These are unchanged historical summaries, not new runs.
Source commit: `1d66c0f7b39119e3ee7777ad7d086df68769d427` on `study/v5`.

| File | Original path | Paper use |
| --- | --- | --- |
| `pilot_gate.csv` | `experiments/2026-11-03_ab_analysis_pilot/pilot_gate.csv` | Eight selected units, 92.75% reduction, 99.0% agreement; DeiT-B search |
| `register_deit.csv` | `experiments/2026-11-03_ab_analysis_formal/register_deit.csv` | Formal reduction 91.5% and unchanged 79.8% accuracy (R0 vs R2) |

`configs/register.yaml` reruns the pilot search for DeiT-S. Use `--model deit_b`
with a temporary copy of that config listing DeiT-B to repeat its search.
The historical full-validation accuracy and formal register-count summaries are supplied
as evidence; the current search script does not recompute those summaries.
The current model-side map evaluation uses `configs/table1_register.yaml`.
