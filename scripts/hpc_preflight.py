"""Inspect the interpreter, pinned assets, storage, and GPU without model inference."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root

ROOT = configure_runtime()

import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download

from mita_probe.assets import IMAGENET_REVISION, MODEL_REPO, MODEL_REVISION
from mita_probe.data import experiment_indices
from mita_probe.io import source_commit, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["environment", "assets", "ready"], default="environment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    commit = source_commit(ROOT)
    errors = []
    packages = {name: version(name) for name in ["torch", "torchvision", "timm", "datasets", "huggingface-hub", "numpy"]}
    for name, expected in {"torch": "2.7.0", "torchvision": "0.22.0", "timm": "1.0.30"}.items():
        if packages[name].split("+")[0] != expected:
            errors.append(f"{name} must be {expected}")
    if sys.version_info[:2] not in [(3, 11), (3, 12)] or sys.prefix == sys.base_prefix:
        errors.append("Use an isolated Python 3.11/3.12 virtual environment")
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "stage": args.stage, "platform": platform.system(), "python": platform.python_version(),
        "isolated_venv": sys.prefix != sys.base_prefix, "packages": packages,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "data_free_gib": round(shutil.disk_usage(root).free / 2**30, 2),
        "model_revision": MODEL_REVISION, "dataset_revision": IMAGENET_REVISION,
    }
    if torch.cuda.is_available():
        report["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15,
            )
            report["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired) as error:
            # CPU containers can expose a non-executable NVIDIA placeholder.
            report["nvidia_smi"] = f"unavailable ({type(error).__name__})"
    if args.stage in {"assets", "ready"}:
        if args.stage == "ready" and not torch.cuda.is_available():
            errors.append("The first GPU job requires a visible CUDA device")
        try:
            weights = Path(hf_hub_download(MODEL_REPO, "model.safetensors", revision=MODEL_REVISION, local_files_only=True))
            with weights.open("rb") as stream:
                report["model_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest = json.loads((root / "imagenet_manifest.json").read_text(encoding="utf-8"))
            if manifest["revision"] != IMAGENET_REVISION:
                errors.append("ImageNet revision differs from the accepted P2 dataset")
            dataset = load_from_disk(str(root / "imagenet_val"))
            if len(dataset) != 50000 or len(dataset.features["label"].names) != 1000:
                errors.append("ImageNet validation must contain 50,000 images and 1,000 classes")
            indices = experiment_indices(len(dataset), 5000, 0, root / "imagenet_val_indices_seed0.json")
            report["dataset_images"] = len(dataset)
            report["subset_images"] = len(indices)
            report["canonical_indices_sha256"] = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"Asset check failed: {type(error).__name__}; check local model/data preparation")
    report["errors"] = errors
    report["passed"] = not errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
