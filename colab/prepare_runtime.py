"""Prepare an isolated Colab environment from the uploaded, committed Git bundle."""

import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path('/content/MITA_vit')
if ROOT.exists():
    raise FileExistsError('Preserve the existing checkout; inspect before retrying')
subprocess.run(['git', 'clone', '/content/mita-source.bundle', str(ROOT)], check=True)
os.chdir(ROOT)
if (3, 11) <= sys.version_info[:2] < (3, 13):
    subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
else:
    # Current Colab kernels use 3.13; keep the project on its supported 3.12.
    if not shutil.which('uv'):
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    subprocess.run(['uv', 'venv', '--python', '3.12', '--seed', '.venv'], check=True)
python = str(ROOT / '.venv/bin/python')
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
checks = ROOT / 'experiments/2026-10-27_colab_environment'
checks.mkdir(parents=True, exist_ok=False)

commands = [
    [python, '-m', 'pip', 'install', '--no-deps', '-r', 'requirements-torch.txt',
     '--index-url', 'https://download.pytorch.org/whl/cu128'],
    [python, '-m', 'pip', 'install', '-r', 'requirements-torch.txt', '-r', 'requirements.txt'],
    [python, '-m', 'pip', 'install', '--no-deps', '-e', '.'],
    [python, '-m', 'pip', 'check'],
    [python, 'scripts/hpc_preflight.py', '--stage', 'environment', '--output',
     str(checks / 'environment.json')],
    [python, '-m', 'pytest', '-q', '--basetemp', str(checks / 'pytest-tmp'),
     '--junitxml', str(checks / 'tests.xml')],
]
with (checks / 'setup.log').open('w') as log:
    for command in commands:
        print('Running:', ' '.join(command), flush=True)
        log.write('Running: ' + ' '.join(command) + '\n')
        log.flush()
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (checks / 'exit_code.txt').write_text(str(result.returncode))
            raise RuntimeError(f'Setup failed; see setup.log: {command[:3]}')
(checks / 'exit_code.txt').write_text('0')
print('Environment and tests passed; uploaded assets must be verified separately.', flush=True)

