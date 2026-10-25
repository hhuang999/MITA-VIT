"""Verify validation shards, label order, saved subset, and decoded sample images."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root

ROOT = configure_runtime()

import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import HfApi, constants
from timm.data import ImageNetInfo

from mita_probe.data import ImageNetValidation, subset_indices
from mita_probe.io import source_commit, write_json
from mita_probe.models import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-inference", action="store_true", help="Check data only while preparing the environment")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    commit = source_commit(ROOT)
    root = data_root()
    manifest = json.loads((root / "imagenet_manifest.json").read_text())
    revision = manifest["revision"]
    cache = Path(constants.HF_HUB_CACHE) / "datasets--ILSVRC--imagenet-1k/snapshots" / revision
    paths = [f"data/validation-{i:05d}-of-00014.parquet" for i in range(14)]
    remote = HfApi().get_paths_info("ILSVRC/imagenet-1k", paths, repo_type="dataset",
                                  revision=revision, token=True)
    if len(remote) != 14:
        raise ValueError("Expected metadata for 14 validation shards")
    verified = []
    for info in sorted(remote, key=lambda item: item.path):
        path = cache / info.path
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != info.size or digest != info.lfs.sha256:
            raise ValueError(f"Shard checksum mismatch: {info.path}")
        verified.append({"file": info.path, "bytes": info.size, "sha256": digest})
        print(f"Verified {info.path}", flush=True)
    data = load_from_disk(str(root / "imagenet_val"))
    labels = np.asarray(data["label"])
    if len(data) != 50000 or not np.array_equal(np.bincount(labels, minlength=1000), np.full(1000, 50)):
        raise ValueError("Validation must contain 50 images per ImageNet class")
    names = data.features["label"].names
    reference = ImageNetInfo()
    # HF distinguishes construction crane (517) from bird crane (134) by name.
    aliases = {517: "crane2"}
    mismatches = [i for i, name in enumerate(names)
                  if name != aliases.get(i, reference.index_to_description(i))]
    if mismatches:
        raise ValueError(f"ImageNet label descriptions differ at {mismatches[:10]}")
    indices = json.loads((root / "imagenet_val_indices_seed0.json").read_text())
    if indices != subset_indices(50000, 5000, 0):
        raise ValueError("Subset indices differ from the fixed seed-0 sample")
    for position, index in enumerate(indices):
        image = data[index]["image"].convert("RGB")
        image.load()
        if min(image.size) <= 0:
            raise ValueError(f"Invalid image size at index {index}")
        if (position + 1) % 1000 == 0:
            print(f"Decoded {position + 1}/5000 subset images", flush=True)
    smoke = {"performed": False}
    if not args.skip_inference:
        torch.set_num_threads(2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(device)
        prepared = ImageNetValidation(root / "imagenet_val", model)
        batch = torch.stack([prepared[index][0] for index in indices[:32]]).to(device)
        with torch.inference_mode():
            logits = model(batch)
        if logits.shape != (32, 1000) or not torch.isfinite(logits).all():
            raise ValueError("ImageNet loader/model smoke check failed")
        smoke = {"performed": True, "n_images": 32, "output_shape": list(logits.shape),
                 "finite": True, "device": device, "preprocessing": prepared.preprocessing}
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "dataset": "ILSVRC/imagenet-1k", "revision": revision, "split": "validation",
        "n_images": len(data), "n_classes": 1000, "images_per_class": 50,
        "label_order_matches_timm": True,
        "label_name_aliases": {"517": "HF crane2 = timm crane (construction crane)"},
        "subset_seed": 0, "subset_size": 5000,
        "subset_unique": len(set(indices)) == 5000, "decoded_subset_images": len(indices),
        "indices_sha256": hashlib.sha256((root / "imagenet_val_indices_seed0.json").read_bytes()).hexdigest(),
        "download_bytes": sum(item["bytes"] for item in verified), "shards": verified,
        "loader_smoke": smoke,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print("ImageNet validation download verified", flush=True)


if __name__ == "__main__":
    main()
