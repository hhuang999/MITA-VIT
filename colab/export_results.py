"""Export only this Colab run's small experiment artifacts, never credentials."""

from pathlib import Path
import tarfile

ROOT = Path('/content/MITA_vit')
destination = Path('/content/mita-colab-results.tar.gz')
folders = [ROOT / 'experiments' / name for name in [
    '2026-10-27_colab_environment', '2026-10-27_rq1_colab', '2026-10-27_rq1_colab_k128']]
with tarfile.open(destination, 'x:gz') as archive:
    for folder in folders:
        for path in folder.iterdir():
            if path.is_file():
                archive.add(path, arcname=path.relative_to(ROOT))
print(f'Exported experiment artifacts: {destination.stat().st_size} bytes')
