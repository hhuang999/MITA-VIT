"""T7 (decision B30): how much of the model's prediction survives with no attention at all?

On the fixed 5,000-image ImageNet subset with the model's own preprocessing (as in the paper's execution
figure), the attention branch of every layer (`all`) or of one layer at a time outputs zero; the residual
stream and the FFNs are unchanged. Per image and setting (`attention_off.csv`): top-1 agreement with the
intact model and KL(intact || edited).
"""

import logging
import time

import torch

from ..io import write_json
from ..models import load_model
from ..probe import Plan, execute
from . import common, datasets
from .common import Sink

FIELDS = ['index', 'layers', 'target', 'original', 'executed', 'agree', 'kl']


def run(args, config, output, commit):
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    dataset, indices = datasets.build('imagenet', model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    depth = len(model.blocks)
    settings = {'all': list(range(1, depth + 1))} | {str(layer): [layer] for layer in range(1, depth + 1)}
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, images=len(indices), settings=list(settings), smoke_images=args.smoke,
        resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    sink = Sink(output / 'attention_off.csv', FIELDS)
    for start, ids, (images, labels) in batches(f'attention_off_{args.model}', dataset, indices):
        images = images.to(device)
        labels = labels.tolist()
        rows = []
        with torch.no_grad():
            original = model(images)
            for name, layers in settings.items():
                with execute(model, {layer: Plan(drop=True) for layer in layers}):
                    edited = model(images)
                rows += common.outcomes(ids, labels, original, edited, layers=name)
        sink.write(rows)
    common.finish(output, started)
    logging.info('Complete: attention off for %s', args.model)
