"""Top-k map similarity: how close is the attention map read through the top-k reading sets to the full attention
map (R0 up to a positive factor)?

Per ImageNet-S image, layers `layers` summed: the top-k map (references.topk_map: attention each patch receives
when every query reads only its region's shared top-k set; the main budget of the execution checks) and the full
attention map (references.attention_mass). similarity.csv: their Spearman correlation per image.
"""

import logging
import time

import torch

from ..data.build import build
from ..execution import resolve
from ..io import write_json
from ..metrics import spearman
from ..models import load_model
from ..references import attention_mass, topk_map
from . import common
from .common import Sink


def run(args, config, output, commit):
    settings = config['similarity']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    dataset, indices = build(settings['dataset'], model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    spec = resolve(dict(mode='L3'), config['main'])
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, dataset=settings['dataset'], images=len(indices), spec=spec,
        layers=settings['layers'], smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    sink = Sink(output / 'similarity.csv', ['model', 'index', 'spearman'])
    for start, ids, batch in batches(f'similarity_{args.model}', dataset, indices):
        images = batch[0].to(device)
        topk = topk_map(model, images, spec, settings['layers']).float()
        full = attention_mass(model, images, settings['layers']).float()
        rho = spearman(topk.double(), full.double()).tolist()
        sink.write([dict(model=args.model, index=index, spearman=rho[n]) for n, index in enumerate(ids)])
    common.finish(output, started)
    logging.info('Complete: top-k map similarity for %s', args.model)
