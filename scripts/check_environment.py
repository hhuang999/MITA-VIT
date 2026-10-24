"""Run pretrained DeiT-S on a real Imagenette validation image using CUDA."""

import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

from mita_probe.runtime import configure_runtime

ROOT = configure_runtime()

import torch
import timm
from torchvision.datasets import Imagenette


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the project interpreter")
    torch.manual_seed(0)
    model = timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=True)
    model.eval().to("cuda")
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)
    data = Imagenette(ROOT / "data", split="val", size="160px", transform=transform)
    image, label = data[0]
    with torch.inference_mode():
        logits = model(image.unsqueeze(0).to("cuda"))
    assert logits.shape == (1, 1000)
    assert torch.isfinite(logits).all()
    prediction = logits.argmax(-1).item()
    # Imagenette labels index ten classes; the pretrained head indexes ImageNet-1k.
    imagenet_ids = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    report = {
        "timestamp": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["torch", "torchvision", "timm", "datasets"]},
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "model": "deit_small_patch16_224.fb_in1k",
        "parameters": sum(p.numel() for p in model.parameters()),
        "preprocessing": config,
        "image": str(Path(data._samples[0][0]).relative_to(ROOT)),
        "target_imagenet": imagenet_ids[label],
        "prediction_imagenet": prediction,
        "finite_logits": True,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    output = ROOT / "experiments" / "2026-10-24_environment"
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
