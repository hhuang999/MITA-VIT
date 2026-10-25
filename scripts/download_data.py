"""Download public debug data or ImageNet validation on the chosen machine."""

import argparse
import json
from pathlib import Path

from mita_probe.runtime import configure_runtime

ROOT = configure_runtime()

from torchvision.datasets import Imagenette, OxfordIIITPet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["imagenette", "pet", "imagenet"])
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "imagenette":
        data = Imagenette(args.root, split="val", size="160px", download=True)
        print(f"Imagenette validation: {len(data)} images")
    elif args.dataset == "pet":
        data = OxfordIIITPet(
            args.root, split="test", target_types="segmentation", download=True
        )
        print(f"Oxford-IIIT Pet test: {len(data)} images")
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        import numpy as np

        revision = HfApi().dataset_info("ILSVRC/imagenet-1k", token=True).sha
        data = load_dataset(
            "ILSVRC/imagenet-1k", split="validation", token=True,
            revision=revision, data_files={"validation": "data/validation-*.parquet"},
            cache_dir=str(args.root / "imagenet_cache"),
        )
        if len(data) != 50000:
            raise ValueError("Expected all 50,000 ImageNet validation images")
        data.save_to_disk(str(args.root / "imagenet_val"))
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
