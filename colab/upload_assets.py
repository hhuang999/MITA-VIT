"""Upload only the allowlisted data archive parts through the installed CLI."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'data/colab_transfer/assets'
assert (folder / 'manifest.json').exists(), 'Packaging must finish first'
paths = sorted(folder.glob('assets.part[0-9][0-9][0-9][0-9]')) + [folder / 'manifest.json']
prefix = '/mnt/' + ROOT.drive[0].lower() + ROOT.as_posix()[2:]
cli = '/home/hhuan/.local/share/google-colab-cli-venv/bin/colab'
log_path = folder / 'uploaded.txt'
completed = set(log_path.read_text().splitlines()) if log_path.exists() else set()
for index, path in enumerate(paths, 1):
    if path.name in completed:
        continue
    remote = ('/content/mita-assets-manifest.json' if path.name == 'manifest.json'
              else '/content/mita-' + path.name)
    local = prefix + '/' + path.relative_to(ROOT).as_posix()
    result = subprocess.run(['wsl', '-d', 'Ubuntu', '--exec', cli, '--auth', 'oauth2',
                             'upload', '-s', 'mita-rq1-20261027', local, remote],
                            capture_output=True, text=True, timeout=300)
    if result.returncode:
        print(result.stdout, flush=True)
        raise RuntimeError(f'Upload failed for {path.name}; exit {result.returncode}')
    with log_path.open('a') as stream:
        stream.write(path.name + '\n')
    print(f'Uploaded {index}/{len(paths)}: {path.name}', flush=True)
print('All allowlisted assets uploaded. No credential files included.', flush=True)
