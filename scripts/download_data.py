"""Download a dataset to the data folder.

imagenet: ImageNet validation at a pinned HF revision (needs HF_TOKEN); pet: Oxford-IIIT Pet test split;
imagenet_s / hard_imagenet: mask annotations only (their images come from the ImageNet validation copy);
clickme: the 17 ClickMe validation shards (images, human maps, labels), each checked against its GCS MD5;
partimagenet: the PartImageNet_Seg archive (images and part masks; macOS metadata entries are skipped).
Archives are checked against a pinned SHA-256 before they are extracted.
"""

import argparse
import base64
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root
from mita_probe.data import IMAGENET_REPO, IMAGENET_REVISION, clickme, hard_imagenet, imagenet_s, partimagenet

ROOT = configure_runtime()

ANNOTATIONS = {
    "imagenet_s": (imagenet_s.ANNOTATION_URL, imagenet_s.ANNOTATION_SHA256),
    "hard_imagenet": (hard_imagenet.ANNOTATION_URL, hard_imagenet.ANNOTATION_SHA256),
}

from torchvision.datasets import OxfordIIITPet


def file_digest(path: Path, name: str) -> bytes:
    digest = hashlib.new(name)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["pet", "imagenet", "imagenet_s", "hard_imagenet", "clickme",
                                            "partimagenet"])
    parser.add_argument("--root", type=Path, default=data_root())
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    if args.dataset in ANNOTATIONS:
        url, sha256 = ANNOTATIONS[args.dataset]
        archive = args.root / f"{args.dataset}.zip"
        urllib.request.urlretrieve(url, archive)
        digest = file_digest(archive, "sha256").hex()
        if digest != sha256:
            raise ValueError(f"{archive} has SHA-256 {digest}, expected {sha256}")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(args.root / args.dataset)
        archive.unlink()
        print(f"{args.dataset}: annotations extracted to {args.root / args.dataset}")
    elif args.dataset == "clickme":
        destination = args.root / "clickme"
        destination.mkdir(exist_ok=True)
        for shard, md5 in enumerate(clickme.SHARD_MD5):
            path = destination / f"val-{shard}.tfrecords"
            urllib.request.urlretrieve(clickme.SHARD_URL.format(shard), path)
            if base64.b64encode(file_digest(path, "md5")).decode() != md5:
                raise ValueError(f"{path} does not match its published MD5 {md5}")
        print(f"ClickMe validation: {len(clickme.SHARD_MD5)} shards in {destination}")
    elif args.dataset == "partimagenet":
        archive = args.root / "PartImageNet_Seg.zip"
        urllib.request.urlretrieve(partimagenet.ARCHIVE_URL, archive)
        digest = file_digest(archive, "sha256").hex()
        if digest != partimagenet.ARCHIVE_SHA256:
            raise ValueError(f"{archive} has SHA-256 {digest}, expected {partimagenet.ARCHIVE_SHA256}")
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist()
                       if name.startswith("PartImageNet/") and "/." not in name and not name.endswith(".DS_Store")]
            bundle.extractall(args.root / "partimagenet", members)
        archive.unlink()
        print(f"PartImageNet_Seg: extracted to {args.root / 'partimagenet' / 'PartImageNet'}")
    elif args.dataset == "pet":
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
