"""Run the authorized 42 RQ1 settings on one verified Colab GPU."""

import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/content/MITA_vit')
os.chdir(ROOT)
checks = ROOT / 'experiments/2026-10-27_colab_environment'
assert (checks / 'exit_code.txt').read_text() == '0'
preflight = json.loads((checks / 'preflight.json').read_text())
assert preflight['passed'] and preflight['cuda_available']
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
           OMP_NUM_THREADS='2', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
python = str(ROOT / '.venv/bin/python')
for config, name in [('rq1.yaml', '2026-10-27_rq1_colab'),
                     ('rq1_k128.yaml', '2026-10-27_rq1_colab_k128')]:
    subprocess.run([python, 'scripts/run_rq1.py', '--config', f'configs/{config}',
                    '--output', f'experiments/{name}'], env=env, check=True)
print('All 42 settings completed; retrieve artifacts before releasing the VM.', flush=True)
