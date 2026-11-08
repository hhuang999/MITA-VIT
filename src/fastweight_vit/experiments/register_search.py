"""Register-neuron search: the FFN units that make high-norm tokens (after Jiang et al. 2025).

The "+register" row of Table 1 shifts these units to a test-time register token (experiments/correct.py,
`variant: register`). The units found for DeiT-S are written into configs/register.yaml; this experiment shows how
they were found (pilot split of ImageNet-S, UNIFORM_PREPROCESSING; config `search`):
1. one unedited pass: high-norm patches per layer; L* = first layer where >= `presence_share` of the
   images have one (registers.first_high_layer). Without L* the search stops (`gate.json`: found False).
2. second pass: FFN unit statistics in layers 1..L*-1 on the patches that are high-norm at L*
   (registers.NeuronStats); `neurons.csv` ranks every unit.
3. gate: for n in `counts`, shift the top n units to a register; reduction of high-norm patch-layers at
   L* and later (vs the unedited pass), top-1 agreement; the same for n random units (seeds). `gate.csv`;
   `gate.json` records L*, the smallest n passing all three criteria (`min_reduction`, `min_agreement`,
   `max_random_reduction`) and activation-matched units.
"""

import logging
import time

import torch

from ..data.build import build
from ..execution import capture_input_norms
from ..instrument import instrument
from ..io import write_json
from ..models import load_model
from ..registers import (NeuronStats, as_layers, edit_neurons, first_high_layer, matched_neurons, random_neurons,
                         ranked_neurons)
from ..tokens import high_norm_tokens
from . import common
from .common import Sink

GATE = ['n', 'condition', 'seed', 'high_before', 'high_after', 'reduction', 'agreement']
NEURONS = ['rank', 'layer', 'unit', 'score', 'high_mean', 'mean', 'std']


def run(args, config, output, commit):
    settings = config['register']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    if getattr(model, 'mita_plan', None) is not None:
        raise ValueError('Test-time registers are not compatible with MiTA routing')
    search(args, config, settings, model, output, commit, device)


def pass_images(config, dataset, indices, device):
    for start in range(0, len(indices), config['batch_size']):
        batch = [dataset[index] for index in indices[start:start + config['batch_size']]]
        yield start, indices[start:start + len(batch)], [torch.stack([item[i] for item in batch]).to(device)
                                                          if torch.is_tensor(batch[0][i]) else None
                                                          for i in range(3)]


@torch.no_grad()
def high_counts(model, images, budget, neurons=None, register=False, mode='shift'):
    """Logits and high-norm masks [B, layers, patches] under an edit (explicit-softmax forward)."""
    prefix = model.num_prefix_tokens + int(register)
    with edit_neurons(model, neurons, mode=mode, register=register), instrument(model), \
            capture_input_norms(model, prefix) as norms:
        logits = model(images)
    high = torch.stack([high_norm_tokens(norms[layer], budget['norm_ratio']) for layer in sorted(norms)], 1)
    return logits, high


def search(args, config, settings, model, output, commit, device):
    """Steps 1-3 of the module docstring on the pilot split."""
    budget = config['search']
    dataset, indices = build(budget['dataset'], model, args.model, config | {'split': 'pilot'})
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
