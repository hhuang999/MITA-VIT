"""Neurons that make high-norm tokens and test-time registers (T2 search, T6 formal; decision B30).

Stage `search` (pilot split of ImageNet-S):
1. one unedited pass: high-norm patches per layer; L* = first layer where >= `presence_share` of the
   images have one (registers.first_high_layer). Without L* the search stops (`gate.json`: found False).
2. second pass: FFN unit statistics in layers 1..L*-1 on the patches that are high-norm at L*
   (registers.NeuronStats); `neurons.csv` ranks every unit.
3. gate (H3): for n in `counts`, shift the top n units to a register; reduction of high-norm patch-layers at
   L* and later (vs the unedited pass), top-1 agreement; the same for n random units (seeds). `gate.csv`;
   `gate.json` records L*, the smallest n passing all three criteria and the matched units (R4).

Stage `formal` (formal split of ImageNet-S, then all 50,000 ImageNet validation images), conditions:
R0 original, R1 register token only, R2 found units shifted, R3_s<seed> random units shifted, R4
activation-matched units shifted, R5 found units zeroed without a register. Per image of ImageNet-S
(`register.csv`): budget shares per layer (all queries) incl. the register, high-norm patch count per layer,
deep enrichment with R0's layer 9-12 high-norm patches excluded (and with none excluded), agreement with R0;
per ImageNet image (`accuracy.csv`): the prediction under each condition.
"""

import json
import logging
import time

import torch

from .. import plausibility
from ..budget import CATEGORIES, categories, shares
from ..data import ImageNetValidation
from ..io import write_json
from ..models import UNIFORM_PREPROCESSING, data_reference, load_model
from ..registers import (NeuronStats, as_layers, edit_neurons, first_high_layer, matched_neurons, random_neurons,
                         ranked_neurons)
from ..runtime import data_root, project_path
from . import common, datasets
from .budget import deep_scores, ledger
from .common import Sink

STAGES = ('search', 'formal')
GATE = ['n', 'condition', 'seed', 'high_before', 'high_after', 'reduction', 'agreement']
NEURONS = ['rank', 'layer', 'unit', 'score', 'high_mean', 'mean', 'std']
SHARE_FIELDS = [f'share_{name}' for name in CATEGORIES]
FORMAL = ['index', 'condition', 'layer', 'high_count'] + SHARE_FIELDS
DEEP = ['index', 'condition', 'exclude', 'mass', 'enrichment', 'agree']
ACCURACY = ['index', 'label', 'condition', 'prediction']


def run(args, config, output, commit):
    settings = config['register']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    if getattr(model, 'mita_plan', None) is not None:
        raise ValueError('Test-time registers are not compatible with MiTA routing')
    (search if args.stage == 'search' else formal)(args, config, settings, model, output, commit, device)


def pass_images(config, dataset, indices, device):
    for start in range(0, len(indices), config['batch_size']):
        batch = [dataset[index] for index in indices[start:start + config['batch_size']]]
        yield start, indices[start:start + len(batch)], [torch.stack([item[i] for item in batch]).to(device)
                                                          if torch.is_tensor(batch[0][i]) else None
                                                          for i in range(3)]


@torch.no_grad()
def high_counts(model, images, budget, neurons=None, register=False, mode='shift'):
    """Logits and high-norm masks [B, layers, patches] under an edit."""
    with edit_neurons(model, neurons, mode=mode, register=register):
        logits, _, high = ledger(model, images, None, None, budget, values=False,
                                 prefix=model.num_prefix_tokens + int(register))
    return logits, high


def search(args, config, settings, model, output, commit, device):
    budget = config['budget']
    dataset, indices = datasets.build(budget['dataset'], model, args.model, config | {'split': 'pilot'})
    indices = indices[:args.smoke] if args.smoke else indices
    started = time.perf_counter()
    write_json(output / 'metadata.json', common.provenance(commit, model=args.model, stage='search',
                                                           images=len(indices), settings=settings))
    images_cache, logits0, high0 = [], [], []
    for start, ids, (images, _, _) in pass_images(config, dataset, indices, device):
        logits, high = high_counts(model, images, budget)
        images_cache.append(images.cpu())
        logits0.append(logits)
        high0.append(high)
    high0 = torch.cat(high0)
    first = first_high_layer(high0.any(-1).cpu(), settings['presence_share'])
    result = dict(model=args.model, first_high_layer=first, presence=high0.any(-1).float().mean(0).tolist(),
                  found=False, n=None, neurons=[], matched=[])
    if first is None or first == 1:
        write_json(output / 'gate.json', result)
        common.finish(output, started)
        logging.info('No layer with high-norm patches in %s of the images: search stops', settings['presence_share'])
        return
    stats = NeuronStats()
    searched = list(range(1, first))
    offset = 0
    for images in images_cache:
        images = images.to(device)
        with torch.no_grad(), edit_neurons(model, capture=set(searched)) as acts:
            model(images)
        stats.add(acts, high0[offset:offset + len(images), first - 1], model.num_prefix_tokens)
        offset += len(images)
    table = stats.table()
    ranked = ranked_neurons(table)
    sink = Sink(output / 'neurons.csv', NEURONS)
    sink.write([dict(rank=r, layer=layer, unit=unit, score=score, high_mean=float(table[layer]['high_mean'][unit]),
                     mean=float(table[layer]['mean'][unit]), std=float(table[layer]['std'][unit]))
                for r, (layer, unit, score) in enumerate(ranked)])
    later = slice(first - 1, None)
    before = int(high0[:, later].sum())
    reference = torch.cat(logits0).argmax(-1)
    gate = Sink(output / 'gate.csv', GATE)
    passing = None
    for n in settings['counts']:
        chosen = [(layer, unit) for layer, unit, _ in ranked[:n]]
        trials = [('found', None, chosen)] + [('random', seed, random_neurons(table, n, seed, chosen))
                                              for seed in settings['seeds']]
        rows = []
        for name, seed, pairs in trials:
            after, agree, offset = 0, 0, 0
            for images in images_cache:
                images = images.to(device)
                logits, high = high_counts(model, images, budget, as_layers(pairs, device), register=True)
                after += int(high[:, later].sum())
                agree += int((logits.argmax(-1) == reference[offset:offset + len(images)]).sum())
                offset += len(images)
            rows.append(dict(n=n, condition=name, seed='' if seed is None else seed, high_before=before,
                             high_after=after, reduction=1 - after / max(before, 1), agreement=agree / len(reference)))
        gate.write(rows)
        found, randoms = rows[0], rows[1:]
        mean_random = sum(row['reduction'] for row in randoms) / len(randoms)
        if passing is None and found['reduction'] >= settings['min_reduction'] and \
                found['agreement'] >= settings['min_agreement'] and mean_random <= settings['max_random_reduction']:
            passing = (n, chosen)
    if passing:
        n, chosen = passing
        result.update(found=True, n=n, neurons=chosen, matched=matched_neurons(table, chosen))
    write_json(output / 'gate.json', result)
    common.finish(output, started)
    logging.info('Search complete for %s: L* = %s, found = %s, n = %s', args.model, first, result['found'],
                 result['n'])


def conditions(gate, seeds, device):
    found = as_layers(gate['neurons'], device)
    out = {'R0': dict(), 'R1': dict(register=True), 'R2': dict(register=True, neurons=found),
           'R4': dict(register=True, neurons=as_layers(gate['matched'], device)),
           'R5': dict(neurons=found, mode='zero')}
    for seed in seeds:
        out[f'R3_s{seed}'] = dict(register=True, neurons=as_layers(gate[f'random_s{seed}'], device))
    return dict(sorted(out.items()))


def formal(args, config, settings, model, output, commit, device):
    budget = config['budget']
    gate = json.loads(project_path(settings['gate'].format(model=args.model)).read_text())
    if not gate['found']:
        raise ValueError('The H3 gate did not pass for this model: RQ3 is descriptive and T6 does not run')
    # Random units for R3 are drawn from the same searched layers, with the gate's seeds and table.
    table = {layer: dict(score=torch.zeros(width)) for layer, width in
             ((layer, model.blocks[layer - 1].mlp.fc1.out_features) for layer in range(1, gate['first_high_layer']))}
    chosen = [tuple(pair) for pair in gate['neurons']]
    for seed in settings['seeds']:
        gate[f'random_s{seed}'] = random_neurons(table, len(chosen), seed, chosen)
    edits = conditions(gate, settings['seeds'], device)
    dataset, indices = datasets.build(budget['dataset'], model, args.model, config | {'split': 'formal'})
    indices = indices[:args.smoke] if args.smoke else indices
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, stage='formal', images=len(indices), gate=gate, conditions=list(edits),
        smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    per_layer, deep_sink = Sink(output / 'register.csv', FORMAL), Sink(output / 'deep.csv', DEEP)
    deep = budget['deep_layers']
    for start, ids, batch in batches(f'register_{args.model}', dataset, indices):
        images, foreground, valid = (tensor.to(device) for tensor in batch[:3])
        reference, exclude = None, None
        rows, deep_rows = [], []
        for name, edit in edits.items():
            register = edit.get('register', False)
            prefix = model.num_prefix_tokens + int(register)
            with torch.no_grad(), edit_neurons(model, edit.get('neurons'), mode=edit.get('mode', 'shift'),
                                               register=register):
                logits, store, high = ledger(model, images, foreground, valid, budget, values=False, prefix=prefix)
            if name == 'R0':
                reference, exclude = logits.argmax(-1), high[:, [layer - 1 for layer in deep]].any(1)
            for layer in sorted(store):
                part = shares(store[layer]['received'], categories(foreground, valid, high[:, layer - 1],
                                                                    prefix)).tolist()
                counts = high[:, layer - 1].sum(1).tolist()
                rows += [dict(index=index, condition=name, layer=layer, high_count=counts[n],
                              **dict(zip(SHARE_FIELDS, part[n]))) for n, index in enumerate(ids)]
            scores = deep_scores(store, deep, prefix)
            other = foreground.roll(1, 0) if len(ids) > 1 else torch.zeros_like(foreground)
            agree = (logits.argmax(-1) == reference).tolist()
            for label, mask in (('high', exclude), ('none', torch.zeros_like(exclude))):
                values = plausibility.measures(scores, foreground, valid & ~mask, other,
                                               generator=torch.Generator().manual_seed(start))
                deep_rows += [dict(index=index, condition=name, exclude=label, mass=float(values['mass'][n]),
                                   enrichment=float(values['enrichment'][n]), agree=int(agree[n]))
                              for n, index in enumerate(ids)]
        per_layer.write(rows)
        deep_sink.write(deep_rows)
    if not args.smoke or settings.get('smoke_accuracy', False):
        accuracy(args, config, model, edits, output, device, done, progress_path, started)
    common.finish(output, started)
    logging.info('Complete: test-time register conditions for %s', args.model)


def accuracy(args, config, model, edits, output, device, done, progress_path, started):
    dataset = ImageNetValidation(data_root() / 'imagenet_val', data_reference(model),
                                 preprocessing=UNIFORM_PREPROCESSING if config.get('preprocessing') == 'uniform'
                                 else None)
    indices = list(range(len(dataset)))[:args.smoke] if args.smoke else list(range(len(dataset)))
    batches = common.make_batches(config, done, progress_path, started)
    sink = Sink(output / 'accuracy.csv', ACCURACY)
    for start, ids, (images, labels) in batches(f'accuracy_{args.model}', dataset, indices):
        images = images.to(device)
        rows = []
        for name, edit in edits.items():
            with torch.no_grad(), edit_neurons(model, edit.get('neurons'), mode=edit.get('mode', 'shift'),
                                               register=edit.get('register', False)):
                prediction = model(images).argmax(-1).tolist()
            rows += [dict(index=index, label=int(labels[n]), condition=name, prediction=prediction[n])
                     for n, index in enumerate(ids)]
        sink.write(rows)
