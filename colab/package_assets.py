"""Package only pre-existing data/model files, with hashes and bounded-size parts."""

import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/colab_transfer/assets'
OUT.mkdir(parents=True, exist_ok=False)
files = sorted((ROOT / 'data/imagenet_val').iterdir())
files += [ROOT / 'data/imagenet_manifest.json', ROOT / 'data/imagenet_val_indices_seed0.json']
model = ROOT / ('checkpoints/huggingface/hub/models--timm--deit_small_patch16_224.fb_in1k/'
                'snapshots/91327a9c99f98fe6b524cd4d397b7226b80e1365')
files += sorted(model.iterdir())
manifest = []
for path in files:
    assert path.is_file() and path.suffix in {'.arrow', '.json', '.safetensors'}
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest.append({'path': path.relative_to(ROOT).as_posix(),
                     'bytes': path.stat().st_size, 'sha256': digest})


class SplitWriter:
    """Keep each CLI upload small; no large in-memory archive."""

    def __init__(self):
        self.part = 0
        self.used = 0
        self.stream = None

    def write(self, data):
        length = len(data)
        view = memoryview(data)
        while view:
            if self.stream is None:
                self.stream = (OUT / f'assets.part{self.part:04d}').open('xb')
            count = min(len(view), 48 * 2**20 - self.used)
            self.stream.write(view[:count])
            self.used += count
            view = view[count:]
            if self.used == 48 * 2**20:
                self.stream.close()
                self.stream = None
                self.part += 1
                self.used = 0
        return length

    def close(self):
        if self.stream:
            self.stream.close()


writer = SplitWriter()
try:
    with tarfile.open(fileobj=writer, mode='w|') as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
finally:
    writer.close()
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Packaged {len(files)} files into {len(list(OUT.glob("assets.part*")))} parts; no credentials.')

