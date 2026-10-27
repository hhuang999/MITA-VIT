"""Upload only the allowlisted data archive parts through the installed CLI."""

from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'data/colab_transfer/assets'
assert (folder / 'manifest.json').exists(), 'Packaging must finish first'
paths = sorted(folder.glob('assets.part[0-9][0-9][0-9][0-9]')) + [folder / 'manifest.json']
prefix = '/mnt/' + ROOT.drive[0].lower() + ROOT.as_posix()[2:]
cli = '/home/hhuan/.local/share/google-colab-cli-venv/bin/colab'
log_path = folder / 'uploaded.txt'
completed = set(log_path.read_text().splitlines()) if log_path.exists() else set()

def upload(path):
    remote = ('/content/mita-assets-manifest.json' if path.name == 'manifest.json'
              else '/content/mita-' + path.name)
    local = prefix + '/' + path.relative_to(ROOT).as_posix()
    result = subprocess.run(['wsl', '-d', 'Ubuntu', '--exec', cli, '--auth', 'oauth2',
                             'upload', '-s', 'mita-rq1-20261027', local, remote],
                            capture_output=True, text=True, timeout=300)
    if result.returncode:
        print(result.stdout, flush=True)
        raise RuntimeError(f'Upload failed for {path.name}; exit {result.returncode}')
    return path.name


pending = [path for path in paths if path.name not in completed]
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(upload, path) for path in pending]
    for future in as_completed(futures):
        name = future.result()
        with log_path.open('a') as stream:
            stream.write(name + '\n')
        completed.add(name)
        print(f'Uploaded {len(completed)}/{len(paths)}: {name}', flush=True)
print('All allowlisted assets uploaded. No credential files included.', flush=True)
