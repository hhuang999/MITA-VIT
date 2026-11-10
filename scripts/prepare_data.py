"""Download and check the datasets (they are not distributed with this repository).

  python scripts/prepare_data.py imagenet imagenet_s clickme   # download
  python scripts/prepare_data.py --check                                     # check what is there

Data go to data/ (or $FASTWEIGHT_DATA_ROOT):

| dataset | source and licence | layout | check |
| --- | --- | --- | --- |
| imagenet | ILSVRC/imagenet-1k on Hugging Face, validation split at a pinned revision; ImageNet terms of access (accept them on the Hub; needs HF_TOKEN) | data/imagenet_val (datasets.save_to_disk) | 50,000 images |
| imagenet_s | ImageNet-S919 validation masks (LUSSeg/ImageNet-S release); images come from the ImageNet copy above | data/imagenet_s/ImageNetS919/validation-segmentation | 12,419 masks; archive SHA-256 |
| clickme | ClickMe validation shards of the Harmonization release (serrelab), 17 TFRecord files | data/clickme/val-<n>.tfrecords | 17 shards; published MD5 of each |

The fixed image lists (data/splits/*.json) are part of the repository; --check confirms that the 5,000-image
ImageNet list and the held-out list refer to images that exist.
"""

import argparse
import base64
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from fastweight_vit.runtime import configure_runtime, data_root

ROOT = configure_runtime()  # sets HF_HOME before any Hugging Face library is imported

from fastweight_vit.data import clickme, imagenet_s  # noqa: E402
from fastweight_vit.data.imagenet import IMAGENET_REPO, IMAGENET_REVISION  # noqa: E402

DATASETS = ('imagenet', 'imagenet_s', 'clickme')


def file_digest(path: Path, name: str) -> bytes:
    digest = hashlib.new(name)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b''):
            digest.update(chunk)
    return digest.digest()


def fetch_archive(url, sha256, archive, destination, members=None):
    urllib.request.urlretrieve(url, archive)
    digest = file_digest(archive, 'sha256').hex()
    if digest != sha256:
        raise ValueError(f'{archive} has SHA-256 {digest}, expected {sha256}')
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination, members(bundle) if members else None)
    archive.unlink()


def download(dataset, root):
    if dataset == 'imagenet_s':
        fetch_archive(imagenet_s.ANNOTATION_URL, imagenet_s.ANNOTATION_SHA256, root / 'imagenet_s.zip', root / 'imagenet_s')
    elif dataset == 'clickme':
        destination = root / 'clickme'
        destination.mkdir(exist_ok=True)
        for shard, md5 in enumerate(clickme.SHARD_MD5):
            path = destination / f'val-{shard}.tfrecords'
            urllib.request.urlretrieve(clickme.SHARD_URL.format(shard), path)
            if base64.b64encode(file_digest(path, 'md5')).decode() != md5:
                raise ValueError(f'{path} does not match its published MD5 {md5}')
    else:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
        destination = root / 'imagenet_val'
        if destination.exists():
            raise FileExistsError(f'Dataset exists at {destination}; check it instead of overwriting')
        snapshot = Path(snapshot_download(IMAGENET_REPO, repo_type='dataset', revision=IMAGENET_REVISION, token=True,
                                          allow_patterns=['data/validation-*.parquet'], max_workers=4))
        shards = sorted((snapshot / 'data').glob('validation-*.parquet'))
        if len(shards) != 14:
            raise ValueError('Expected 14 ImageNet validation Parquet shards')
        data = load_dataset('parquet', split='validation', data_files={'validation': [str(p) for p in shards]},
                            cache_dir=str(root / 'hf'))
        if len(data) != 50000:
            raise ValueError('Expected all 50,000 ImageNet validation images')
        data.save_to_disk(str(destination))
    print(f'{dataset}: done')


def check(root):
    report = {}
    try:
        from datasets import load_from_disk
        data = load_from_disk(str(root / 'imagenet_val'))
        names = json.loads((ROOT / 'data/splits/heldout_4000.json').read_text())
        rows = json.loads((ROOT / 'data/splits/imagenet_val_5000.json').read_text())
        report['imagenet'] = dict(images=len(data), ok=len(data) == 50000 and max(rows) < len(data)
                                  and max(names['rows']) < len(data))
    except Exception as error:  # noqa: BLE001 - report every dataset
        report['imagenet'] = f'missing or unreadable: {error}'
    masks = list((root / 'imagenet_s' / 'ImageNetS919' / 'validation-segmentation').glob('*/*.png'))
    report['imagenet_s'] = dict(masks=len(masks), ok=len(masks) == imagenet_s.N_VALIDATION)
    shards = [root / 'clickme' / f'val-{n}.tfrecords' for n in range(clickme.N_SHARDS)]
    report['clickme'] = dict(shards=sum(path.exists() for path in shards), ok=all(path.exists() for path in shards))
    print(json.dumps(report, indent=2))
    return all(isinstance(value, dict) and value['ok'] for value in report.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datasets', nargs='*', choices=DATASETS)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--root', type=Path, default=data_root())
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        download(dataset, args.root)
    if args.check or not args.datasets:
        raise SystemExit(0 if check(args.root) else 1)


if __name__ == '__main__':
    main()
