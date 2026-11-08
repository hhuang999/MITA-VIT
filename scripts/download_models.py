"""Download and verify the models: DeiT-S, DeiT-B and DINO ViT-S (timm, pinned revisions), the DINO linear head and
MiTA-DeiT-S.

MiTA-DeiT-S is not trained here: its checkpoint comes from the MiTA authors' Google Drive folder (MITA_FOLDER) and
is fetched with gdown (pip install gdown) into checkpoints/mita/. Every checkpoint is checked against the SHA-256
recorded in fastweight_vit.models.
"""

import subprocess
import sys
import urllib.request

from fastweight_vit.runtime import configure_runtime, project_path

configure_runtime()

import timm  # noqa: E402

from fastweight_vit.models import load_model  # noqa: E402
from fastweight_vit.models.dino import HEAD, HEAD_URL  # noqa: E402

MITA_FOLDER = 'https://drive.google.com/drive/folders/175gOzODWbS_6sap-tXz-9ufh8-y2Mq7k'


def main():
    head = project_path(str(HEAD))
    if not head.exists():
        head.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEAD_URL, head)
    for name in ('deit', 'deit_b', 'dino'):
        model = load_model(name)
        print(f'{name}: {sum(p.numel() for p in model.parameters()):,} parameters, '
              f'{timm.data.resolve_model_data_config(getattr(model, "backbone", model))}')
    checkpoint = project_path('checkpoints/mita/best_checkpoint.pth')
    if not checkpoint.exists():
        folder = project_path('checkpoints/mita_download')
        subprocess.run([sys.executable, '-m', 'gdown', '-q', '--folder', MITA_FOLDER, '-O', str(folder)], check=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        (folder / 'best_checkpoint.pth').rename(checkpoint)
    load_model('mita')
    print('mita: checkpoint verified')


if __name__ == '__main__':
    main()
