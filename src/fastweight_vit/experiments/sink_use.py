"""Is the budget spent on high-norm patches used? (T4, decision B29) and what kind of sink are they?

High-norm patches are fixed from the unedited forward, per layer (block-input norm > norm_ratio x the
image's median). In every layer where an image has some, one edited forward per condition:

| condition | edit at those layers |
| --- | --- |
| high | no query reads the high-norm patches |
| c1 | no query reads as many ordinary patches, those receiving the most budget (sinks.top_ordinary) |
| c2_s<seed> | no query reads as many ordinary patches drawn within the same rings (sinks.ring_random) |
| mass | no query reads the fewest top ordinary patches holding as much budget (secondary; `feasible`) |
| zero | the high-norm patches' values are 0 (budget still spent on them) |
| mean | their values are the pilot mean of high-norm values for that layer and head (value_means.pt) |

Per image and condition (`use.csv`): drop in log-probability of the unedited top-1 class, top-1 agreement,
number of removed patch-layers and the budget share they held (summed over layers).
"""

import logging
import time

import torch

from ..budget import Edits, instrument
from ..io import write_json
from ..models import load_model
from ..execution import batch_generator
from ..runtime import project_path
from ..sinks import equal_mass, ring_random, top_ordinary
from . import common, datasets
from .budget import ledger
from .common import Sink

FIELDS = ['index', 'condition', 'layers', 'removed', 'removed_share', 'feasible', 'short', 'drop', 'agree']


def with_prefix(mask, prefix):
    return torch.cat([mask.new_zeros(mask.shape[0], prefix), mask], 1)


def run(args, config, output, commit):
    """args: model, smoke, resume. Settings in config['sink_use']; the formal split of config['budget']."""
    settings = config['sink_use']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    if getattr(model, 'mita_plan', None) is not None:
        raise ValueError('sink_use edits dense attention only')
    dataset, indices = datasets.build(config['budget']['dataset'], model, args.model, config | {'split': 'formal'})
    indices = indices[:args.smoke] if args.smoke else indices
    means_path = project_path(settings['value_means'].format(model=args.model))
    means = {layer: entry for layer, entry in torch.load(means_path, map_location=device, weights_only=True).items()}
    progress_path, done = common.progress(output, args.resume)
    prefix = model.num_prefix_tokens
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, images=len(indices), value_means=str(settings['value_means']),
        seeds=settings['seeds'], norm_ratio=config['budget']['norm_ratio'], smoke_images=args.smoke,
        resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    sink = Sink(output / 'use.csv', FIELDS)
    for start, ids, batch in batches(f'sink_use_{args.model}', dataset, indices):
        images, foreground, valid = (tensor.to(device) for tensor in batch[:3])
        logits, store, high = ledger(model, images, foreground, valid, config['budget'], values=False)
        target = logits.argmax(-1)
        reference = logits.log_softmax(-1).gather(1, target[:, None])[:, 0]
        layers = [layer for layer in sorted(store) if high[:, layer - 1].any()]
        received = {layer: store[layer]['received'].float().mean(1) for layer in layers}
        share = {layer: received[layer][:, prefix:] / received[layer].sum(1, keepdim=True) for layer in layers}
        conditions = {'high': {layer: high[:, layer - 1] for layer in layers},
                      'c1': {layer: top_ordinary(share[layer], high[:, layer - 1]) for layer in layers}}
        short = {}
        for seed in settings['seeds']:
            masks, short[seed] = {}, torch.zeros(len(ids), dtype=torch.long, device=device)
            for layer in layers:
                masks[layer], missing = ring_random(high[:, layer - 1], batch_generator('cpu', seed, layer, 0, start))
                short[seed] += missing
            conditions[f'c2_s{seed}'] = masks
        feasible = torch.ones(len(ids), dtype=torch.bool, device=device)
        mass = {}
        for layer in layers:
            mass[layer], ok = equal_mass(share[layer], high[:, layer - 1])
            feasible &= ok
        conditions['mass'] = mass
        rows = []
        for name, masks in list(conditions.items()) + [('zero', None), ('mean', None)]:
            if name == 'zero':
                edits = Edits(zero={layer: with_prefix(high[:, layer - 1], prefix) for layer in layers})
                masks = conditions['high']
            elif name == 'mean':
                usable = [layer for layer in layers if layer in means and bool((means[layer]['count'] > 0).all())]
                edits = Edits(mean={layer: (with_prefix(high[:, layer - 1], prefix), means[layer]['mean'])
                                    for layer in usable})
                masks = {layer: conditions['high'][layer] for layer in usable}
            else:
                edits = Edits(block={layer: with_prefix(mask, prefix) for layer, mask in masks.items()})
            with torch.no_grad(), instrument(model, edits=edits):
                edited = model(images)
            drop = (reference - edited.log_softmax(-1).gather(1, target[:, None])[:, 0]).tolist()
            agree = (edited.argmax(-1) == target).tolist()
            removed = sum((mask.sum(1) for mask in masks.values()), torch.zeros(len(ids), device=device)).tolist()
            held = sum(((share[layer] * mask).sum(1) for layer, mask in masks.items()),
                       torch.zeros(len(ids), device=device)).tolist()
            n_layers = sum((mask.any(1).long() for mask in masks.values()),
                           torch.zeros(len(ids), dtype=torch.long, device=device)).tolist()
            ok = feasible.tolist() if name == 'mass' else [True] * len(ids)
            missing = short[int(name[4:])].tolist() if name.startswith('c2_s') else [0] * len(ids)
            rows += [dict(index=index, condition=name, layers=n_layers[n], removed=int(removed[n]),
                          removed_share=held[n], feasible=int(ok[n]), short=missing[n], drop=drop[n],
                          agree=int(agree[n])) for n, index in enumerate(ids)]
        sink.write(rows)
    common.finish(output, started)
    logging.info('Complete: sink use for %s', args.model)

