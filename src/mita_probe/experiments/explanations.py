"""RQ3 and RQ4: our probe and the baseline explanation methods, scored with the same tests.

For each image of the chosen dataset and each configured method (see mita_probe.explain):
- plausibility.csv (mask datasets): object mass, pointing, AUROC (own and another image's mask), top-|object|
  precision, ring-matched enrichment, and the object's area and distance from the centre;
- faithfulness.csv (a fixed random subset): patches kept or removed by rank, at every budget and condition
  (keep_top = insertion, drop_top = deletion of the most important first, drop_bottom = of the least), plus
  keep/drop at the probe's own size (the number of patches the probe reads); agreement, p(top-1), KL;
- saco.csv (same subset): the salience-guided faithfulness coefficient;
- scores/<batch start>.npz: every method's patch scores (float16) for later analysis;
- cost.csv: seconds each method takes per batch (GPU synchronised), next to one plain forward pass.

faithfulness_images: null uses every image.
"""

import logging
import time
from contextlib import contextmanager

import numpy as np
import torch

from .. import explain, plausibility
from ..faithfulness import CONDITIONS, curve, keep_mask, measures, rank, run_kept, saco
from ..io import write_json
from ..models import load_model
from ..probe import resolve
from . import common, datasets
from .common import Sink

PLAUSIBILITY = ['mass', 'pointing', 'auroc', 'auroc_shuffled', 'precision', 'enrichment', 'object_area',
                'object_distance']


@contextmanager
def timer():
    """Yields a callable that returns the seconds spent in the block (CUDA work included)."""
    sync = torch.cuda.synchronize if common.DEVICE == 'cuda' else (lambda: None)
    sync()
    started, result = time.perf_counter(), []
    yield lambda: result[0]
    sync()
    result.append(time.perf_counter() - started)


LAYERWISE = {
    'probe': lambda layer: dict(name=f'probe_layer{layer}', method='probe', weight='mass', layers=[layer]),
    'attention': lambda layer: dict(name=f'attention_layer{layer}', method='attention', layers=[layer]),
    'cls_attention': lambda layer: dict(name=f'cls_layer{layer}', method='cls_attention', layers=[layer]),
}


def layerwise_methods(depth, families=('probe',)):
    """Single-layer maps for each family: the probe (attention mass under the reading sets), dense attention
    mass, CLS attention; named probe_layer<l>, attention_layer<l>, cls_layer<l>."""
    return [LAYERWISE[family](layer) for family in families for layer in range(1, depth + 1)]


def method_scores(entry, model, images, foreground, spec, generator, target=None):
    if entry['method'] == 'probe':
        return explain.probe(model, images, spec, entry['layers'], entry.get('weight', 'mass'))
    return explain.compute(entry['method'], model, images, layers=entry.get('layers'), foreground=foreground,
                           generator=generator, target=target)


def run(args, config, output, commit):
    """args: model, smoke, resume. Settings live in config['explanations']."""
    settings = config['explanations']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    deit = load_model('deit', device) if settings['dataset'] == 'pet' and args.model != 'deit' else None
    dataset, indices = datasets.build(settings['dataset'], model, args.model, config, deit_reference=deit)
    has_masks = settings['dataset'] in datasets.MASK_DATASETS
    methods = [entry for entry in settings['methods'] if has_masks or entry['method'] != 'mask']
    indices = indices[:args.smoke] if args.smoke else indices
    rng = np.random.default_rng(settings['subset_seed'])
    n_faith = len(indices) if settings['faithfulness_images'] is None else min(settings['faithfulness_images'],
                                                                                len(indices))
    faithful = set(np.asarray(indices)[rng.choice(len(indices), n_faith, replace=False)].tolist())
    spec = resolve(settings['probe_spec'], config['main'])
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, dataset=settings['dataset'], images=len(indices), faithfulness_images=n_faith,
        methods=[entry['name'] for entry in methods], probe_spec=spec, smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    plaus = Sink(output / 'plausibility.csv', ['model', 'index', 'method'] + PLAUSIBILITY) if has_masks else None
    faith = Sink(output / 'faithfulness.csv', ['model', 'index', 'method', 'condition', 'budget', 'agree',
                                               'p_target', 'kl'])
    saco_rows = Sink(output / 'saco.csv', ['model', 'index', 'method', 'saco'])
    cost = Sink(output / 'cost.csv', ['model', 'start', 'method', 'images', 'seconds'])
    (output / 'scores').mkdir(exist_ok=True)
    for start, ids, batch in batches(f'explain_{args.model}', dataset, indices):
        images = batch[0].to(device)
        foreground = batch[1].to(device) if has_masks else None
        valid = batch[2].to(device) if has_masks else None
        with timer() as elapsed, torch.no_grad():
            original = model(images)
        costs = [dict(model=args.model, start=start, method='forward', images=len(ids), seconds=elapsed())]
        tie = torch.Generator(device=device).manual_seed(settings['tie_seed'] * 1_000_003 + start)
        noise = torch.Generator(device=device).manual_seed(settings['random_seed'] * 1_000_003 + start)
        chosen = [n for n, index in enumerate(ids) if index in faithful]
        all_scores = {}
        for entry in methods:
            with timer() as elapsed:
                all_scores[entry['name']] = method_scores(entry, model, images, foreground, spec, noise).float()
            costs.append(dict(model=args.model, start=start, method=entry['name'], images=len(ids), seconds=elapsed()))
        cost.write(costs)
        np.savez_compressed(output / 'scores' / f'{start:06d}.npz', index=np.asarray(ids),
                            **{name: value.half().cpu().numpy() for name, value in all_scores.items()})
        probe_k = (all_scores[settings['probe_budget_from']] > 0).sum(1)
        for entry in methods:
            scores = all_scores[entry['name']]
            if has_masks:
                other = foreground.roll(1, 0) if len(ids) > 1 else torch.zeros_like(foreground)
                values = plausibility.measures(scores, foreground, valid, other,
                                               generator=torch.Generator().manual_seed(start))
                plaus.write([dict(model=args.model, index=index, method=entry['name'],
                                  **{key: float(values[key][n]) for key in PLAUSIBILITY})
                             for n, index in enumerate(ids)])
            if not chosen:
                continue
            sub = torch.tensor(chosen, device=device)
            x, logits, s = images[sub], original[sub], scores[sub]
            order = rank(s, tie)
            rows = []
            with torch.no_grad():
                for condition in settings['conditions']:
                    assert condition in CONDITIONS
                    for budget, values in curve(model, x, logits, order, settings['budgets'], condition).items():
                        rows += _rows(args.model, [ids[n] for n in chosen], entry['name'], condition, budget, values)
                k = probe_k[sub].clamp(1, order.shape[1] - 1)
                for condition, keep in [('keep_probe_k', keep_mask(order, k)),
                                        ('drop_probe_k', keep_mask(order, order.shape[1] - k, top=False))]:
                    values = measures(logits, run_kept(model, x, keep))
                    rows += _rows(args.model, [ids[n] for n in chosen], entry['name'], condition, k.tolist(), values)
                faith.write(rows)
                if settings['saco_groups']:
                    coefficient = saco(model, x, logits, s, order, settings['saco_groups']).tolist()
                    saco_rows.write([dict(model=args.model, index=ids[n], method=entry['name'], saco=c)
                                     for n, c in zip(chosen, coefficient)])
    common.finish(output, started)
    logging.info('Complete: explanations on %s', settings['dataset'])


def _rows(model, ids, method, condition, budget, values):
    budgets = budget if isinstance(budget, list) else [budget] * len(ids)
    return [dict(model=model, index=index, method=method, condition=condition, budget=budgets[n],
                 **{key: float(values[key][n]) for key in ('agree', 'p_target', 'kl')})
            for n, index in enumerate(ids)]
