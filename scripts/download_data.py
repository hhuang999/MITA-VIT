"""Download a dataset to the data folder: imagenet (validation, needs HF_TOKEN) or pet (Oxford-IIIT Pet test)."""

import argparse
import json
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root
from mita_probe.data import IMAGENET_REPO, IMAGENET_REVISION

ROOT = configure_runtime()

from torchvision.datasets import OxfordIIITPet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["pet", "imagenet"])
    parser.add_argument("--root", type=Path, default=data_root())
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "pet":
        data = OxfordIIITPet(
            args.root, split="test", target_types="segmentation", download=True
        )
        print(f"Oxford-IIIT Pet test: {len(data)} images")
    else:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
        import numpy as np

        revision = IMAGENET_REVISION
        destination = args.root / "imagenet_val"
        if destination.exists():
            raise FileExistsError(f"Dataset exists at {destination}; verify it instead of overwriting")
        print(f"Downloading ImageNet validation at revision {revision}", flush=True)
        snapshot = Path(snapshot_download(
            IMAGENET_REPO, repo_type="dataset", revision=revision, token=True,
            allow_patterns=["data/validation-*.parquet"], max_workers=4,
        ))
        shards = sorted((snapshot / "data").glob("validation-*.parquet"))
        if len(shards) != 14:
            raise ValueError("Expected 14 ImageNet validation Parquet shards")
        data = load_dataset(
            "parquet", split="validation",
            data_files={"validation": [str(path) for path in shards]},
            # Keep generated lock filenames below the Windows path-length limit.
            cache_dir=str(args.root / "hf"),
        )
        if len(data) != 50000:
            raise ValueError("Expected all 50,000 ImageNet validation images")
        data.save_to_disk(str(destination))
        indices = np.random.default_rng(0).choice(len(data), 5000, replace=False)
        index_path = args.root / "imagenet_val_indices_seed0.json"
        index_path.write_text(json.dumps(indices.tolist()), encoding="utf-8")
        manifest = {
            "dataset": "ILSVRC/imagenet-1k", "revision": revision,
            "split": "validation", "n_images": len(data),
            "features": data.features.to_dict(), "fingerprint": data._fingerprint,
            "subset_seed": 0, "subset_size": 5000,
            "index_file": index_path.name,
        }
        (args.root / "imagenet_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"ImageNet validation: {len(data)} images; saved 5,000 indices (seed 0)")


if __name__ == "__main__":
    main()
