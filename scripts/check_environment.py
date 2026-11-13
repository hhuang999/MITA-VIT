"""Report the software stack and run each available model once on a random image (GPU when present)."""

import json
import platform
from importlib.metadata import version

from fastweight_vit.runtime import configure_runtime

configure_runtime()

import torch  # noqa: E402

from fastweight_vit.models import NAMES, load_model  # noqa: E402


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report = dict(python=platform.python_version(),
                  packages={p: version(p) for p in ['torch', 'torchvision', 'timm', 'datasets']},
                  cuda=torch.version.cuda, device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
                  models={})
    image = torch.randn(1, 3, 224, 224, device=device)
    for name in NAMES:
        try:
            model = load_model(name, device)
        except (FileNotFoundError, ValueError) as error:
            report['models'][name] = f'unavailable: {error}'
            continue
        with torch.inference_mode():
            logits = model(image)
        report['models'][name] = dict(logits=list(logits.shape), finite=bool(torch.isfinite(logits).all()))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
