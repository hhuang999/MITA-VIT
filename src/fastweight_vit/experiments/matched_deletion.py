"""Necessity test (decision A15): remove foreground or background neurons from the reading sets.

The dataset is config['matched_deletion']['dataset'] (default pet; also imagenet_s or hard_imagenet).

On the MiTA-default plan (m = 25, k = 25, dot routing, CLS routed), all 12 layers, three conditions
per category: target (remove that category, refill from the best outside tokens), random (remove the
same number of random patches per set) and centre-matched (remove the same number per set, drawn from
the same distance-to-centre rings). The conditions differ only in which tokens leave.
"""

import logging
import time

import torch
from torch.utils.data import DataLoader, Subset

from ..interventions import (centre_rings, replace_category, replace_matched_patches, replace_random_patches,
                             sink_tokens)
from ..io import write_json
from ..models import load_model
from ..execution import Pattern, Plan, base_pattern, batch_generator, capture_input_norms, capture_qk, execute, resolve
from ..execution import token_mask
from . import common, datasets
from .anatomy import MITA_SPEC
from .common import OUTCOME, Sink, outcomes, update_checks

CATEGORIES = ['foreground', 'background']
FIELDS = ['model'] + OUTCOME + ['category', 'kind', 'seed', 'eligible', 'mean_removed_per_set']
MODELS = ['deit', 'dino', 'mita']


def deletion(model, kind, qk, norms, images, fg, valid, ids, original, start, config, device, output):
    spec = resolve(MITA_SPEC, dict(scoring='query', summary_scale='mita'))
    scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
    summary = dict(summary_side=5, summary_scale='mita', landmark_mode='mita')
    base, scores = {}, {}
    for layer in range(1, 13):
        pattern, scores[layer] = base_pattern(spec, *qk[layer], scale, grid)
        base[layer] = Pattern(pattern.readsets, pattern.routes)
    plan = lambda patterns: {layer: Plan(token_mask=token_mask(patterns[layer]), **summary) for layer in patterns}
    labels = [-1] * len(ids)
    run = lambda patterns: _execute(model, plan(patterns), images)
    executed = run(base)
    if kind == 'mita':
        error = (executed - original).abs().max().item()
        assert error < 1e-5, error
        update_checks(output, max_pet_frozen_native_error=error)
    rows = outcomes(ids, labels, original, executed, model=kind, category='', kind='baseline', seed=-1,
                    eligible=1, mean_removed_per_set=0.)
    zero = torch.zeros(len(ids), 1, dtype=torch.bool, device=device)
    rings = centre_rings()
    for category in CATEGORIES:
        c = config['pet']['deletion_categories'].index(category)  # same random streams as the anatomy pet stage
        changed, doses, masks = {}, {}, {}
        eligible = torch.ones(len(ids), dtype=torch.bool, device=device)
        for layer in range(1, 13):
            sink = sink_tokens(norms[layer], config['pet']['sink_threshold'])
            mask = fg & valid if category == 'foreground' else ~fg & valid & ~sink
            masks[layer] = torch.cat([zero, mask], -1)
            changed[layer], feasible, doses[layer] = replace_category(base[layer], scores[layer], masks[layer])
            eligible &= feasible
        dose = torch.stack([d.float().mean((1, 2)) for d in doses.values()], -1).mean(-1).tolist()

        def record(output_logits, condition, seed):
            for row, ok, removed in zip(outcomes(ids, labels, original, output_logits, model=kind,
                                                 category=category, kind=condition, seed=seed),
                                        eligible.tolist(), dose):
                rows.append(row | dict(eligible=int(ok), mean_removed_per_set=removed))

        record(run(changed), 'target', -1)
        for seed in config['random_seeds']:
            randomized = {layer: replace_random_patches(base[layer], scores[layer], doses[layer],
                          batch_generator(device, seed * 10 + c, layer, 25, start)) for layer in range(1, 13)}
            record(run(randomized), 'random', seed)
            matched = {}
            for layer in range(1, 13):
                matched[layer], removed = replace_matched_patches(
                    base[layer], scores[layer], masks[layer], rings,
                    batch_generator(device, (seed + 100) * 10 + c, layer, 25, start))
                assert torch.equal(removed.sum(-1), doses[layer])
            record(run(matched), 'centre_matched', seed)
    return rows


def _execute(model, plans, images):
    with execute(model, plans):
        return model(images)


def run(args, config, output, commit):
    """args: smoke. Runs the three models over the Pet test split and writes deletion.csv."""
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    models = {name: load_model(name, device) for name in MODELS}
    name = config.get('matched_deletion', {}).get('dataset', 'pet')
    write_json(output / 'metadata.json', common.provenance(commit, decision='A15', dataset=name,
                                                           seeds=config['random_seeds'], categories=CATEGORIES))
    sink = Sink(output / 'deletion.csv', FIELDS)
    started = time.perf_counter()
    for kind, model in models.items():
        # Pet uses DeiT-S geometry for every model; ImageNet-based sets use each model's own preprocessing.
        dataset, indices = datasets.build(name, model, kind, config, deit_reference=models['deit'])
        if name == 'pet':
            assert len(dataset) == config['pet']['images']
        indices = indices[:args.smoke] if args.smoke else indices
        start = 0
        with torch.inference_mode():
            for images, fg, valid, ids in DataLoader(Subset(dataset, indices), batch_size=config['batch_size'],
                                                     num_workers=config['num_workers'], shuffle=False):
                images, fg, valid = images.to(device), fg.to(device), valid.to(device)
                with capture_qk(model) as qk, capture_input_norms(model) as norms:
                    original = model(images)
                sink.write(deletion(model, kind, qk, norms, images, fg, valid, ids.tolist(), original, start,
                                    config, device, output))
                start += len(ids)
                if start % 640 < len(ids) or start == len(indices):
                    logging.info('%s %d/%d images; %.1fs', kind, start, len(indices), time.perf_counter() - started)
    common.finish(output, started)
