"""Download and verify the studied models: DeiT-S and DINO (timm, pinned revisions) and the DINO linear head.

The MiTA-DeiT-S checkpoint comes from the authors' Google Drive folder; see colab/setup.sh for the gdown
command. Every checkpoint is checked against the SHA-256 recorded in mita_probe.models.
"""

import urllib.request

from mita_probe.runtime import configure_runtime, project_path

configure_runtime()

import timm  # noqa: E402

from mita_probe.models import load_model  # noqa: E402
from mita_probe.models.dino import HEAD, HEAD_URL  # noqa: E402


def main():
    head = project_path(str(HEAD))
    if not head.exists():
        head.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEAD_URL, head)
    for name in ('deit', 'dino'):
        model = load_model(name)
        print(f'{name}: {sum(p.numel() for p in model.parameters()):,} parameters, '
              f'{timm.data.resolve_model_data_config(getattr(model, "backbone", model))}')
    if project_path('checkpoints/mita/best_checkpoint.pth').exists():
        load_model('mita')
        print('mita: checkpoint verified')
    else:
        print('mita: checkpoint missing (download it with gdown, see colab/setup.sh)')


if __name__ == '__main__':
    main()
