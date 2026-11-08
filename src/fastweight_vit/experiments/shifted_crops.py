"""RQ3 position intervention: put the same object at different places in the frame.

Each image is resized so its shorter side is `resize` pixels, then cropped to 224 x 224 at the centre and at
the given offsets; crops that keep less than `min_object_kept` of the object are skipped. If a map follows
the object, its object measures stay high when the object moves off-centre; a centre-driven map does not.
Output crops.csv: one row per image, crop and method, with the plausibility measures and the object's
distance from the centre in that crop.
"""

import logging
import time

import numpy as np
import torch
from timm.data import resolve_model_data_config
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ..data import token_foreground
from ..io import write_json
from ..models import IMAGENET_PREPROCESSING, data_reference, load_model
from ..plausibility import measures
from ..execution import resolve
from . import common, datasets
from .common import Sink
from .explanations import PLAUSIBILITY, method_scores


def crops(image, trimap, resize, offsets, mean, std, size=224):
    """[(dy, dx, tensor, trimap tensor, kept object share)] for each offset that stays inside the image."""
    image = TF.resize(image, resize, interpolation=InterpolationMode.BICUBIC)
    trimap = TF.resize(trimap, resize, interpolation=InterpolationMode.NEAREST)
    labels = torch.from_numpy(np.array(trimap, dtype=np.int64))
    total = int((labels == 1).sum())
    width, height = image.size
    out = []
    for dy, dx in offsets:
        top, left = (height - size) // 2 + dy, (width - size) // 2 + dx
        if top < 0 or left < 0 or top + size > height or left + size > width:
            continue
        window = labels[top:top + size, left:left + size]
        kept = int((window == 1).sum()) / total if total else 0.
        tensor = TF.normalize(TF.to_tensor(TF.crop(image, top, left, size, size)), mean, std)
        out.append((dy, dx, tensor, window, kept))
    return out


def run(args, config, output, commit):
    settings = config['shifted_crops']
    common.deterministic()
    device = common.DEVICE
    model = load_model(args.model, device)
    dataset, indices = datasets.build(settings['dataset'], model, args.model, config)
    rng = np.random.default_rng(settings['subset_seed'])
    indices = sorted(rng.choice(indices, min(settings['images'], len(indices)), replace=False).tolist())
    indices = indices[:args.smoke] if args.smoke else indices
    pre = IMAGENET_PREPROCESSING[args.model] or resolve_model_data_config(data_reference(model))
    spec = resolve(settings['probe_spec'], config['main'])
    offsets = [tuple(offset) for offset in settings['offsets']]
    write_json(output / 'metadata.json', common.provenance(commit, model=args.model, dataset=settings['dataset'],
                                                           images=len(indices), offsets=offsets, resize=settings['resize']))
    sink = Sink(output / 'crops.csv', ['model', 'index', 'dy', 'dx', 'object_kept', 'method'] + PLAUSIBILITY)
    started = time.perf_counter()
    for count, index in enumerate(indices):
        image, _, trimap = dataset.sample(index)
        views = [view for view in crops(image, trimap, settings['resize'], offsets, pre['mean'], pre['std'])
                 if view[4] >= settings['min_object_kept']]
        if not views:
            continue
        images = torch.stack([view[2] for view in views]).to(device)
        tokens = [token_foreground(view[3]) for view in views]
        foreground = torch.stack([t[0] for t in tokens]).to(device)
        valid = torch.stack([t[1] for t in tokens]).to(device)
        generator = torch.Generator(device=device).manual_seed(settings['random_seed'] * 1_000_003 + index)
        rows = []
        for entry in settings['methods']:
            scores = method_scores(entry, model, images, foreground, spec, generator).float()
            values = measures(scores, foreground, valid, torch.zeros_like(foreground),
                              generator=torch.Generator().manual_seed(index))
            rows += [dict(model=args.model, index=index, dy=view[0], dx=view[1], object_kept=view[4],
                          method=entry['name'], **{key: float(values[key][n]) for key in PLAUSIBILITY})
                     for n, view in enumerate(views)]
        sink.write(rows)
        if (count + 1) % 200 == 0:
            logging.info('%d/%d images; %.1fs', count + 1, len(indices), time.perf_counter() - started)
    common.finish(output, started)
