"""Download pinned assets only after explicit user approval of HF credential transfer.

This file contains no credentials. The coordinator must obtain permission before
uploading the HF-only file to the user's own Colab runtime.
"""

import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/content/MITA_vit')
checks = ROOT / 'experiments' / os.environ.get('MITA_ENV_RUN', '2026-10-27_colab_environment_retry')
assert (checks / 'exit_code.txt').read_text() == '0'
credential = Path('/content/mita-hf.env')
assert credential.is_file(), 'Explicitly authorized credential upload required'
destination = ROOT / '.env'
if destination.exists():
    raise FileExistsError('Inspect existing credentials before retrying')
destination.write_bytes(credential.read_bytes())
destination.chmod(0o600)
python = str(ROOT / '.venv/bin/python')
commands = [
    [python, 'scripts/download_models.py'],
    [python, 'scripts/download_data.py', 'imagenet'],
    [python, 'scripts/verify_imagenet.py', '--skip-inference', '--output',
     str(checks / 'data_verification.json')],
    [python, 'scripts/hpc_preflight.py', '--stage', 'ready', '--output', str(checks / 'preflight.json')],
]
with (checks / 'assets.log').open('x') as log:
    for command in commands:
        print('Running:', ' '.join(command), flush=True)
        log.write('Running: ' + ' '.join(command) + '\n')
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Asset preparation failed; inspect and redact log before sharing')
original = json.loads((ROOT / 'experiments/2026-10-25_imagenet_download/verification.json').read_text())
fresh = json.loads((checks / 'data_verification.json').read_text())
assert fresh['shards'] == original['shards'], 'Source shard hashes differ'
assert fresh['indices_sha256'] == original['indices_sha256'], 'Sample indices differ'
assert fresh['decoded_subset_images'] == 5000 and fresh['label_order_matches_timm']
local_manifest = json.loads(Path('/content/mita-assets-manifest.json').read_text())
model_hash = next(r['sha256'] for r in local_manifest if r['path'].endswith('model.safetensors'))
ready = json.loads((checks / 'preflight.json').read_text())
assert ready['model_sha256'] == model_hash, 'Weights differ from the existing local model'
(checks / 'source_comparison.json').write_text(json.dumps({
    'all_passed': True, 'same_14_parquet_hashes': True, 'same_indices_file': True,
    'same_model_sha256': True, 'decoded_images': 5000,
}, indent=2))
print('Pinned data, fixed sample, and weights match the original experiment.')
