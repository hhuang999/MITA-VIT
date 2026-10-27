"""Run smoke checks then V01-V03, exporting each completed task immediately."""
import json
import os
from pathlib import Path
import subprocess
import tarfile

root = Path('/content/MITA_vit')
os.chdir(root)
environment = root/'experiments/2026-10-27_validation_environment_ready'
assert (environment/'exit_code.txt').read_text() == '0'
assert json.loads((environment/'pet_verification.json').read_text())['all_sha256_match']
assert json.loads((environment/'source_comparison.json').read_text())['all_passed']
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
           OMP_NUM_THREADS='2', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
python = str(root/'.venv/bin/python')
status = {}


def export(name):
    folder = root/'experiments'/name
    destination = Path('/content') / f'{name}.tar.gz'
    with tarfile.open(destination,'x:gz') as archive:
        for path in folder.iterdir():
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root))


export(environment.name)
for stage in ['v01','v02','v03']:
    name = f'2026-10-27_{stage}_smoke'
    result = subprocess.run([python,'scripts/run_validation.py',stage,'--smoke','16',
                             '--output',f'experiments/{name}'], env=env)
    status[f'{stage}_smoke'] = result.returncode
    export(name)
    (environment/'run_status.json').write_text(json.dumps(status,indent=2))
    if result.returncode:
        raise RuntimeError(f'{stage} smoke failed; formal run not started')
for stage in ['v01','v02','v03']:
    name = f'2026-10-27_{stage}_colab'
    status[stage] = 'running'
    (environment/'run_status.json').write_text(json.dumps(status,indent=2))
    result = subprocess.run([python,'scripts/run_validation.py',stage,'--output',f'experiments/{name}'], env=env)
    status[stage] = result.returncode
    export(name)
    (environment/'run_status.json').write_text(json.dumps(status,indent=2))
    if result.returncode:
        raise RuntimeError(f'{stage} failed; preserve output and inspect')
    print(f'{stage} complete and exported.', flush=True)
print('All approved validations completed; retrieve artifacts and stop this runtime.', flush=True)
