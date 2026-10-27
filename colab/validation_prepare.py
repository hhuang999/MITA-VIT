"""Continue validation setup in the existing runtime, preserving first failure."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path('/content/MITA_vit')
os.chdir(root)
subprocess.run(['git', 'fetch', '/content/validation-update.bundle',
                'feature/overnight-validations-20261027'], check=True)
subprocess.run(['git', 'checkout', '--detach', 'FETCH_HEAD'], check=True)
checks = root / 'experiments/2026-10-27_validation_environment_ready'
checks.mkdir(exist_ok=False)
python = str(root / '.venv/bin/python')
env = os.environ.copy()
env.update(PYTHONUNBUFFERED='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
commands = [
    [python, '-m', 'pip', 'install', '--no-deps', '-r', 'requirements-torch.txt',
     '--index-url', 'https://download.pytorch.org/whl/cu128'],
    [python, '-m', 'pip', 'install', '-r', 'requirements-torch.txt', '-r', 'requirements.txt'],
    [python, '-m', 'pip', 'install', '--no-deps', '-e', '.'],
    [python, '-m', 'pip', 'check'],
    [python, 'scripts/hpc_preflight.py', '--stage', 'environment', '--output', str(checks/'environment.json')],
    [python, '-m', 'pytest', '-q', '--basetemp', str(checks/'pytest-tmp'), '--junitxml', str(checks/'tests.xml')],
]
with (checks / 'setup.log').open('x') as log:
    for command in commands:
        print('Running:', ' '.join(command), flush=True)
        log.write('Running: ' + ' '.join(command) + '\n')
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
(checks/'exit_code.txt').write_text('0')
os.environ['MITA_ENV_RUN'] = checks.name
exec(compile((root/'colab/download_assets.py').read_text(), 'download_assets.py', 'exec'))
with (checks/'pet.log').open('x') as log:
    subprocess.run([python, 'scripts/download_data.py', 'pet'], env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
manifest = json.loads(Path('/content/mita-pet-manifest.json').read_text())
for record in manifest:
    path = root / 'data/oxford-iiit-pet' / record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], record['path']
(checks/'pet_verification.json').write_text(json.dumps(dict(files=len(manifest), all_sha256_match=True)))
print('Environment, ImageNet and all 7339 Pet files passed.', flush=True)
