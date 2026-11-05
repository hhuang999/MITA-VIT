"""Stop rule of B39/B41: the negative controls P4 must hold, otherwise the queue stops (exit 1).

  python scripts/check_p4.py <run prefix> <model>

Reads <prefix>_correct_ablation_{imagenet_s,clickme}_<model>. P4a (DINO only): C3 - R0 at layers 9-12 is
'comparable' on SRG, enrichment and ClickMe. P4b: C2 - R0 at layers 1-4 is 'comparable' on enrichment and
ClickMe. Writes <prefix>_p4_<model>.csv next to the runs.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_v5 import ROOT, find, p4_rows, per_image  # noqa: E402


def main():
    prefix, model = sys.argv[1], sys.argv[2]
    prefix = str(ROOT / prefix) if not Path(prefix).is_absolute() else prefix
    imagenet, clickme = find(prefix, f'correct_ablation_imagenet_s_{model}'), find(prefix, f'correct_ablation_clickme_{model}')
    if imagenet is None or clickme is None:
        print('P4 CHECK FAILED: missing runs')
        sys.exit(1)
    rows = []
    for metric, table in per_image(imagenet, clickme).items():
        rows += p4_rows(model, metric, table)
    frame = pd.DataFrame(rows)
    frame.to_csv(f'{prefix}_p4_{model}.csv', index=False)
    print(frame[['test', 'model', 'metric', 'mean', 'low', 'high', 'verdict']].to_string(index=False))
    expected = 5 if model == 'dino' else 2
    if len(frame) != expected or (frame.verdict != 'comparable').any():
        print('P4 CHECK FAILED')
        sys.exit(1)
    print('P4 CHECK PASSED')


if __name__ == '__main__':
    main()
