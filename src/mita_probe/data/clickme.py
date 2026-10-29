"""ClickMe human feature-importance maps on ImageNet validation images (Linsley et al. 2019; Fel et al. 2022).

The Harmonization validation set: 17 TFRecord shards of tf.train.Example records, each with a 256 x 256 JPEG
image, a 256 x 256 JPEG heatmap (how often players revealed each pixel) and the ImageNet-1k label. We follow
the Harmonization evaluation geometry: the image is resized (bilinear) to 224 x 224 without cropping; the
heatmap is resized to 64 x 64, blurred with their 10 x 10 Gaussian (sigma 10) and resized to 224 x 224.
Records are parsed without TensorFlow: TFRecord framing plus the three Example fields.
"""

import io
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

SHARD_URL = 'https://storage.googleapis.com/serrelab/prj_harmonization/dataset/click-me/val/val-{}.tfrecords'
# Base64 MD5 digests reported by Google Cloud Storage (x-goog-hash), checked after download.
SHARD_MD5 = [
    'XYfp9PsvieQNPzmRTVzndQ==', 'ibjns2kTS+5IgOYVMcSxlQ==', 'h2mJkSfdAbZV+v29fuLtUQ==', '5HyZFdH61gpL+2KXFSWK6w==',
    'zNeWngrRunmqudkfDoTA7A==', '07QFdWe0g4ofrtPuT9Sy+Q==', 'fUUoAvZaVJgysEg2/0B4vQ==', 'oFJzMHnz6rWLHl4g5oIcoQ==',
    'qqJTNs6jMdTpciLtuJNmhg==', 'vpyU7ViQDCKL6sRzqoT2SQ==', 'Bm8E492MXxDkrV9xzX/Pkg==', '/2WkC9F9eMLWCbMm0fkE+g==',
    'wWFHALEbVZIF/HSYu/G2TQ==', 'zCgUelajOl6nIf3wtvbr+w==', 'WkDiDj8d0C+IPUtA+O+4/Q==', 'ZWnySaY4EhGni6AeybjldA==',
    'kulbX9pXW1k5pFPwXyB1lg==',
]
N_SHARDS = len(SHARD_MD5)
HUMAN_CEILING = 0.65753  # Harmonization: mean split-half Spearman between players (evaluation/clickme.py)


def _varint(buffer, position):
    value = shift = 0
    while True:
        byte = buffer[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7


def _fields(buffer):
    """Yield (field number, value) of a protobuf message; length-delimited values are memoryviews."""
    position, end = 0, len(buffer)
    while position < end:
        key, position = _varint(buffer, position)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, position = _varint(buffer, position)
        elif wire == 2:
            length, position = _varint(buffer, position)
            value, position = buffer[position:position + length], position + length
        elif wire == 1:
            value, position = buffer[position:position + 8], position + 8
        elif wire == 5:
            value, position = buffer[position:position + 4], position + 4
        else:
            raise ValueError(f'Unsupported protobuf wire type {wire}')
        yield number, value


def parse_example(record: bytes) -> dict:
    """tf.train.Example -> {name: list of bytes (BytesList) or ints (Int64List)}; FloatList is not used here."""
    out = {}
    for number, features in _fields(memoryview(record)):
        if number != 1:
            continue
        for entry_number, entry in _fields(features):
            if entry_number != 1:
                continue
            name, feature = None, None
            for field, value in _fields(entry):
                if field == 1:
                    name = bytes(value).decode()
                elif field == 2:
                    feature = value
            values = []
            for kind, payload in _fields(feature):
                for field, value in _fields(payload):
                    if field != 1:
                        continue
                    if kind == 1:
                        values.append(bytes(value))
                    elif kind == 3 and isinstance(value, int):
                        values.append(value)
                    elif kind == 3:  # packed int64
                        position = 0
                        while position < len(value):
                            item, position = _varint(value, position)
                            values.append(item)
            out[name] = values
    return out


def record_offsets(path: Path):
    """[(offset, length)] of every record in a TFRecord file (length, CRC, data, CRC framing)."""
    offsets, size = [], path.stat().st_size
    with path.open('rb') as stream:
        position = 0
        while position < size:
            stream.seek(position)
            length, = struct.unpack('<Q', stream.read(8))
            offsets.append((position + 12, length))
            position += 12 + length + 4
    return offsets


def gaussian_kernel(size=10, sigma=10.):
    """The Harmonization blur kernel (common/blur.py), normalised to sum 1.

    Its TF ranges give x = -5..4 and y = 4..-5 for size 10, so the even-sized kernel peaks at row 4, column 5.
    """
    xs = torch.arange(-(size - 1) // 2, (size - 1) // 2 + 1, dtype=torch.float32)
    ys = torch.arange((size - 1) // 2, -(size - 1) // 2 - 1, -1, dtype=torch.float32)
    kernel = torch.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def human_map(heatmap: np.ndarray) -> torch.Tensor:
    """256 x 256 uint8 heatmap -> blurred 224 x 224 float map, as the Harmonization loader does."""
    x = torch.from_numpy(heatmap.astype(np.float32))[None, None]
    x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
    kernel = gaussian_kernel()
    total = kernel.shape[0] - 1  # TF 'SAME' padding puts the extra row/column after
    x = F.conv2d(F.pad(x, (total // 2, total - total // 2, total // 2, total - total // 2)), kernel[None, None])
    return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)[0, 0]


class ClickMe(Dataset):
    def __init__(self, root: Path, mean, std, *, expected_shards=N_SHARDS):
        """root holds val-0.tfrecords ... val-16.tfrecords; mean/std: the model's input normalisation."""
        self.shards = [Path(root, f'val-{n}.tfrecords') for n in range(expected_shards)]
        missing = [path.name for path in self.shards if not path.exists()]
        if missing:
            raise FileNotFoundError(f'Missing ClickMe shards: {missing}')
        self.records = [(shard, offset, length) for shard in self.shards for offset, length in record_offsets(shard)]
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.records)

    def _example(self, index):
        path, offset, length = self.records[index]
        with path.open('rb') as stream:
            stream.seek(offset)
            return parse_example(stream.read(length))

    def record(self, index):
        """(RGB PIL image 256 x 256, heatmap uint8 [256, 256], label)."""
        example = self._example(index)
        image = Image.open(io.BytesIO(example['image'][0])).convert('RGB')
        heatmap = np.array(Image.open(io.BytesIO(example['heatmap'][0])).convert('L'))
        return image, heatmap, int(example['label'][0])

    def heatmap(self, index):
        """The blurred 224 x 224 human map alone (no image decoding)."""
        example = self._example(index)
        return human_map(np.array(Image.open(io.BytesIO(example['heatmap'][0])).convert('L')))

    def image_tensor(self, image):
        x = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)[None]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)[0] / 255
        return (x - self.mean) / self.std

    def __getitem__(self, index):
        image, heatmap, label = self.record(index)
        return self.image_tensor(image), human_map(heatmap), label, index
