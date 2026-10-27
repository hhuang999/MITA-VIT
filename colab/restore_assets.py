"""Restore uploaded data and verify every byte before offline model inference."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/content/MITA_vit')
parts = sorted(Path('/content').glob('mita-assets.part*'))
assert parts, 'No uploaded parts'
archive_path = Path('/content/mita-assets.tar')
with archive_path.open('xb') as target:
    for part in parts:
        with part.open('rb') as source:
            shutil.copyfileobj(source, target, length=2**20)
manifest = json.loads(Path('/content/mita-assets-manifest.json').read_text())
with tarfile.open(archive_path) as archive:
    assert {m.name for m in archive.getmembers()} == {m['path'] for m in manifest}
    archive.extractall(ROOT, filter='data')
for entry in manifest:
    path = ROOT / entry['path']
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == entry['sha256'] and path.stat().st_size == entry['bytes'], entry['path']
checks = ROOT / 'experiments/2026-10-27_colab_environment'
(checks / 'asset_hashes.json').write_text(json.dumps({'all_passed': True, 'files': manifest}, indent=2))
env = os.environ.copy()
env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
result = subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/hpc_preflight.py',
                         '--stage', 'ready', '--output', str(checks / 'preflight.json')],
                        cwd=ROOT, env=env, capture_output=True, text=True)
print(result.stdout)
if result.returncode:
    raise RuntimeError(result.stderr)
print('Uploaded assets match the local data/model byte for byte; ready for identity validation.')
