"""T0 (decision B28, M5): ImageNet validation accuracy of a model under its own preprocessing and under the
uniform preprocessing of the attention-budget study (Resize 256 bicubic, centre crop 224).

Per image (`predictions.csv`): label and the top-1 prediction under both. Also checks that a MiTA model's
dense re-implementation is in place (its plan is recorded in the metadata).
"""

import logging
import time

import torch

from ..data import ImageNetValidation
from ..io import write_json
from ..models import IMAGENET_PREPROCESSING, UNIFORM_PREPROCESSING, data_reference, load_model
from ..runtime import data_root
from . import common
from .common import Sink

FIELDS = ['index', 'label', 'own', 'uniform']


def run(args, config, output, commit):
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    reference = data_reference(model)
    own = ImageNetValidation(data_root() / 'imagenet_val', reference, preprocessing=IMAGENET_PREPROCESSING[args.model])
    uniform = ImageNetValidation(data_root() / 'imagenet_val', reference, preprocessing=UNIFORM_PREPROCESSING)
    indices = list(range(len(own)))[:args.smoke] if args.smoke else list(range(len(own)))
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, images=len(indices), own_preprocessing=own.preprocessing,
        uniform_preprocessing=uniform.preprocessing, mita_plan=str(getattr(model, 'mita_plan', None)),
        smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    sink = Sink(output / 'predictions.csv', FIELDS)
    pairs = zip(batches(f'own_{args.model}', own, indices), batches(f'uniform_{args.model}', uniform, indices))
    for (start, ids, (images, labels)), (_, _, (same, _)) in pairs:
        with torch.no_grad():
            first = model(images.to(device)).argmax(-1).tolist()
            second = model(same.to(device)).argmax(-1).tolist()
        sink.write([dict(index=index, label=int(labels[n]), own=first[n], uniform=second[n])
                    for n, index in enumerate(ids)])
    common.finish(output, started)
    logging.info('Complete: model check for %s', args.model)
