"""Anatomy experiments (RQ1 faithfulness, RQ2 layer structure, RQ3 Pet composition and deletion).

Stages, each writing one folder of CSVs; every stage is resumable batch by batch:

| stage | data | what it measures |
| --- | --- | --- |
| noise | ImageNet subset | agreement after shifting the crop by 1 px (the noise floor) |
| rq1 | ImageNet subset | one layer replaced at a time: L1-L4, random controls; attention coverage |
| rq2 | ImageNet subset | all layers replaced at once over the settings in the config |
| repro | ImageNet, 50,000 | top-1 of the primary model against its published number |
| mita_repro / mita_anatomy | ImageNet | the MiTA-DeiT-S control model (DeiT-S runs only) |
| pet | Oxford-IIIT Pet | reading-set composition, union, examples and matched deletion |
"""

import csv
import logging
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

from ..data import ImageNetValidation, PetSegmentation, experiment_indices
from ..interventions import replace_category, replace_random_patches, sink_tokens
from ..models import IMAGENET_PREPROCESSING, data_reference, load_model
from ..models.deit import REVISION as DEIT_REVISION
from ..models.dino import DINO_REVISION, HEAD_SHA256
from ..models.mita import CHECKPOINT_SHA256
from ..execution import (Pattern, Plan, base_pattern, batch_generator, capture_input_norms, capture_qk, coverage,
                     execute, expand_rq2, make_plan, resolve, token_mask, union_fraction)
from ..runtime import data_root, project_path
from ..io import write_json
from . import common
from .common import OUTCOME, Sink, dense_identity, full_plans, outcomes, update_checks

STAGES = ['noise', 'rq1', 'rq2', 'repro', 'mita_repro', 'mita_anatomy', 'pet']
MODELS = ['deit', 'dino']
MITA_SPEC = dict(mode='L4', m_side=5, k=25, routing='dot', cls='routed', landmark_mode='mita')


def run(args, config, output, commit):
    """args: stage, model, smoke, resume. Loads models and data, then runs one stage."""
    assert config.get('primary', 'deit') == args.model, 'config and --model disagree'
    common.deterministic()
    assert common.DEVICE != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    device = common.DEVICE
    primary = load_model(args.model, device)
    reference = data_reference(primary)  # timm data config; Pet geometry is DeiT-S's (crop 0.9) for all
    mita = None
    if args.model == 'deit' and args.stage in {'noise', 'mita_repro', 'mita_anatomy', 'pet'}:
        mita = load_model('mita', device)
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, stage=args.stage, model=args.model, deit_revision=DEIT_REVISION, mita_sha256=CHECKPOINT_SHA256,
        dino_revision=DINO_REVISION, dino_head_sha256=HEAD_SHA256, batch_size=config['batch_size'],
        smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)

    def imagenet(model_kind, n=None, shift=0):
        dataset = ImageNetValidation(data_root() / 'imagenet_val', reference,
                                     preprocessing=IMAGENET_PREPROCESSING[model_kind], shift=shift)
        if n == 50000:
            indices = list(range(50000))
        else:
            indices = experiment_indices(len(dataset), config['n_images'], config['subset_seed'],
                                         project_path(config['indices']))
        return dataset, indices[:args.smoke] if args.smoke else indices

    def pet():
        dataset = PetSegmentation(data_root(), reference)
        assert len(dataset) == config['pet']['images']
        indices = list(range(len(dataset)))
        return dataset, indices[:args.smoke] if args.smoke else indices

    data = SimpleNamespace(imagenet=imagenet, pet=pet)
    STAGE_FUNCTIONS[args.stage](args, config, output, primary, mita, data, batches)
    common.finish(output, started)
    logging.info('Complete: %s', args.stage)


def make_batches(config, done, progress_path, started):
    return common.make_batches(config, done, progress_path, started)


def compared(args, model, mita):
    """The primary model, then MiTA-DeiT-S when the primary is DeiT-S."""
    return [(args.model, model)] + ([('mita', mita)] if mita is not None else [])


def stage_noise(args, config, output, primary, mita, data, batches):
    rows = Sink(output / 'predictions.csv', ['model'] + OUTCOME)
    models = compared(args, primary, mita)
    for kind, model in models:
        base, indices = data.imagenet(kind)
        shifted, _ = data.imagenet(kind, shift=config['noise']['shift_pixels'])
        shifted_loader = iter(DataLoader(Subset(shifted, indices), batch_size=config['batch_size'],
                                         num_workers=config['num_workers'], shuffle=False))
        position = 0
        with torch.inference_mode():
            for start, ids, batch in batches(f'noise_{kind}', base, indices):
                while position <= start:
                    moved = next(shifted_loader)
                    position += config['batch_size']
                original = model(batch[0].to(common.DEVICE))
                rows.write(outcomes(ids, batch[1].tolist(), original, model(moved[0].to(common.DEVICE)), model=kind))
    update_checks(output, models=[kind for kind, _ in models])


def stage_rq1(args, config, output, model, mita, data, batches):
    dataset, indices = data.imagenet(args.model)
    main = config['main']
    k_main = config['rq1']['k'][0]
    sweep = config['rq2']['k_sweep']
    rows = Sink(output / 'predictions.csv', OUTCOME + ['layer', 'setting', 'seed'])
    cov_fields = ['index', 'layer'] + [f'{name}_k{k}_{part}' for k in sweep
                  for name in ['L1', 'L3_spatial', 'L3_dot', 'rand'] for part in ['patch', 'cls']] + [
                  f'{name}_k{k}_union' for k in sweep for name in ['L3_spatial', 'L3_dot']]
    cover = Sink(output / 'coverage.csv', cov_fields)
    grid = model.patch_embed.grid_size
    scale = model.blocks[0].attn.scale
    settings = [('L3', dict(mode='L3')), ('L4', dict(mode='L4')), ('L2', dict(mode='L2')),
                ('const', dict(mode='L2', m_side=1)), ('drop', dict(mode='drop'))]
    settings += [(f'L1_k{k}', dict(mode='L1', k=k)) for k in config['rq1']['per_query_k']]
    settings += [(f'{mode}_s{seed}', dict(mode=mode, seed=seed))
                 for seed in config['random_seeds'] for mode in ['rand', 'randC']]
    with torch.inference_mode():
        for start, ids, batch in batches('rq1', dataset, indices):
            images, labels = batch[0].to(common.DEVICE), batch[1].tolist()
            original = model(images)
            qk, error = dense_identity(model, images, original)
            update_checks(output, max_identity_error=error)
            out, cov_rows = [], [dict(index=index) for index in ids]
            for layer in range(1, 13):
                q, k = qk[layer]
                probs = ((q * scale) @ k.transpose(-2, -1)).softmax(-1)
                layer_rows = [dict(row, layer=layer) for row in cov_rows]
                for kk in sweep:
                    masks = {'L1': make_plan(resolve(dict(mode='L1', k=kk), main), q, k, scale, grid).token_mask}
                    for name, routing in [('L3_spatial', 'spatial'), ('L3_dot', 'dot')]:
                        spec = resolve(dict(mode='L3', k=kk, routing=routing), main)
                        pattern, _ = base_pattern(spec, q, k, scale, grid)
                        masks[name] = token_mask(pattern)
                        for n, value in enumerate(union_fraction(pattern).tolist()):
                            layer_rows[n][f'{name}_k{kk}_union'] = value
                    masks['rand'] = make_plan(resolve(dict(mode='rand', k=kk, seed=0), main), q, k, scale,
                                              grid, layer=layer, batch_start=start).token_mask
                    for name, mask in masks.items():
                        patch, cls = (part.tolist() for part in coverage(probs, mask))
                        for n in range(len(ids)):
                            layer_rows[n][f'{name}_k{kk}_patch'] = patch[n]
                            layer_rows[n][f'{name}_k{kk}_cls'] = cls[n]
                cover.write(layer_rows)
                extra = []
                if layer == config['rq1']['cls_check']['layer']:
                    kc = config['rq1']['cls_check']['k']
                    extra = [(f'L3_k{kc}_v1_cls_routed', dict(mode='L3', k=kc, routing='dot', cls='routed')),
                             (f'L3_k{kc}_cls_own', dict(mode='L3', k=kc, routing='dot', cls='own'))]
                for name, spec in settings + extra:
                    spec = resolve(dict(k=k_main) | spec, main)
                    plan = make_plan(spec, q, k, scale, grid, layer=layer, batch_start=start)
                    with execute(model, {layer: plan}):
                        out += outcomes(ids, labels, original, model(images), layer=layer, setting=name,
                                        seed=spec.get('seed', -1))
            rows.write(out)
    update_checks(output, settings_per_layer=len(settings))


def stage_rq2(args, config, output, model, mita, data, batches):
    dataset, indices = data.imagenet(args.model)
    specs = expand_rq2(config)
    with (output / 'settings.csv').open('w', newline='') as stream:
        fields = sorted({key for spec in specs for key in spec})
        writer = csv.DictWriter(stream, fieldnames=['setting_id'] + fields)
        writer.writeheader()
        writer.writerows(dict(setting_id=i, **spec) for i, spec in enumerate(specs))
    rows = Sink(output / 'predictions.csv', OUTCOME + ['setting_id'])
    grid, scale = model.patch_embed.grid_size, model.blocks[0].attn.scale
    with torch.inference_mode():
        for start, ids, batch in batches('rq2', dataset, indices):
            images, labels = batch[0].to(common.DEVICE), batch[1].tolist()
            original = model(images)
            qk, error = dense_identity(model, images, original)
            update_checks(output, max_identity_error=error)
            out = []
            for setting_id, spec in enumerate(specs):
                plans = full_plans(12, lambda layer: make_plan(spec, *qk[layer], scale, grid,
                                                               layer=layer, batch_start=start))
                with execute(model, plans):
                    out += outcomes(ids, labels, original, model(images), setting_id=setting_id)
            rows.write(out)
    update_checks(output, settings=len(specs))


def mita_frozen(model, qk, start, spec_overrides=None):
    spec = resolve(MITA_SPEC | (spec_overrides or {}), dict(scoring='query', summary_scale='mita'))
    scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
    return {layer: make_plan(spec, *qk[layer], scale, grid, layer=layer, batch_start=start)
            for layer in range(1, 13)}


def reproduce(args, output, model, kind, data, batches, reported, tolerance):
    """Top-1 on all 50,000 validation images against the published number."""
    dataset, indices = data.imagenet(kind, n=50000)
    rows = Sink(output / 'predictions.csv', ['index', 'target', 'predicted', 'correct'])
    with torch.inference_mode():
        for start, ids, batch in batches(f'{kind}_repro', dataset, indices):
            predicted = model(batch[0].to(common.DEVICE)).argmax(-1).tolist()
            labels = batch[1].tolist()
            rows.write([dict(index=i, target=labels[n], predicted=predicted[n],
                             correct=int(predicted[n] == labels[n])) for n, i in enumerate(ids)])
    import pandas as pd
    accuracy = pd.read_csv(output / 'predictions.csv').drop_duplicates('index')['correct'].mean() * 100
    update_checks(output, top1=accuracy, reported_top1=reported)
    if not args.smoke:
        assert abs(accuracy - reported) <= tolerance, accuracy


def stage_mita_repro(args, config, output, model, mita, data, batches):
    reproduce(args, output, mita, 'mita', data, batches, config['mita']['reported_top1'],
              config['mita']['reproduce_tolerance_pp'])


def stage_repro(args, config, output, model, mita, data, batches):
    reproduce(args, output, model, args.model, data, batches, config['repro']['reported_top1'],
              config['repro']['tolerance_pp'])


def stage_mita_anatomy(args, config, output, model, mita, data, batches):
    dataset, indices = data.imagenet('mita')
    rows = Sink(output / 'predictions.csv', OUTCOME + ['setting', 'seed'])
    settings = [('L4_frozen', {}), ('L3', dict(mode='L3')), ('L2', dict(mode='L2')),
                ('L1_k25', dict(mode='L1'))]
    settings += [(f'{mode}_s{seed}', dict(mode=mode, seed=seed))
                 for seed in config['random_seeds'] for mode in ['rand', 'randC']]
    with torch.inference_mode():
        for start, ids, batch in batches('mita_anatomy', dataset, indices):
            images, labels = batch[0].to(common.DEVICE), batch[1].tolist()
            with capture_qk(mita) as qk:
                native = mita(images)
            out = []
            for name, overrides in settings:
                with execute(mita, mita_frozen(mita, qk, start, overrides)):
                    executed = mita(images)
                if name == 'L4_frozen':
                    error = (executed - native).abs().max().item()
                    assert error < 1e-5 and torch.equal(executed.argmax(-1), native.argmax(-1)), error
                    update_checks(output, max_frozen_native_error=error)
                out += outcomes(ids, labels, native, executed, setting=name, seed=overrides.get('seed', -1))
            rows.write(out)


def stage_pet(args, config, output, primary, mita, data, batches):
    dataset, indices = data.pet()
    comp_fields = ['model', 'index', 'spec', 'layer', 'cls', 'sink', 'foreground', 'background', 'unknown',
                   'fg_area']
    composition = Sink(output / 'composition.csv', comp_fields)
    # A12 union view: each spatial token counted once per layer over all regions and heads.
    union = Sink(output / 'union_composition.csv', ['model', 'index', 'spec', 'layer', 'union_size', 'sink',
                                                    'foreground', 'background', 'unknown', 'fg_area'])
    examples = Sink(output / 'examples.csv', ['model', 'index', 'spec', 'layer', 'tokens'])
    deletion = Sink(output / 'deletion.csv', ['model'] + OUTCOME + ['category', 'kind', 'seed',
                                                                    'eligible', 'mean_removed_per_set'])
    categories = config['pet']['deletion_categories']
    models = compared(args, primary, mita)
    for kind, model in models:
        scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
        with torch.inference_mode():
            for start, ids, batch in batches(f'pet_{kind}', dataset, indices):
                images, fg, valid = (part.to(common.DEVICE) for part in batch[:3])
                labels = [-1] * len(ids)
                with capture_qk(model) as qk, capture_input_norms(model) as norms:
                    original = model(images)
                fg_area = ((fg & valid).sum(-1) / valid.sum(-1).clamp_min(1)).tolist()
                comp_rows, example_rows, union_rows = [], [], []
                for layer in range(1, 13):
                    sink = sink_tokens(norms[layer], config['pet']['sink_threshold'])
                    labels_tok = torch.full((len(ids), 197), 4, device=common.DEVICE)
                    patches = labels_tok[:, 1:]
                    patches[valid & fg] = 2
                    patches[valid & ~fg] = 3
                    patches[sink] = 1
                    labels_tok[:, 0] = 0
                    for comp in config['pet']['composition']:
                        spec = resolve(dict(mode='L3', routing='dot', cls='routed') | comp,
                                       dict(scoring='query', summary_scale='mita'))
                        pattern, _ = base_pattern(spec, *qk[layer], scale, grid)
                        chosen = labels_tok.gather(1, pattern.readsets.flatten(1))
                        counts = torch.nn.functional.one_hot(chosen, 5).float().mean(1).tolist()
                        name = f"m{comp['m_side'] ** 2}_k{comp['k']}_{comp['landmark_mode']}"
                        used = torch.zeros(len(ids), 197, dtype=torch.bool, device=common.DEVICE)
                        used.scatter_(1, pattern.readsets.flatten(1), True)
                        used = used[:, 1:]
                        size = used.sum(-1).clamp_min(1)
                        parts = dict(sink=used & sink, foreground=used & fg & valid & ~sink,
                                     background=used & ~fg & valid & ~sink, unknown=used & ~valid & ~sink)
                        fractions = {key: (value.sum(-1) / size).tolist() for key, value in parts.items()}
                        for n, index in enumerate(ids):
                            union_rows.append(dict(model=kind, index=index, spec=name, layer=layer,
                                                   union_size=int(used[n].sum()), fg_area=fg_area[n],
                                                   **{key: value[n] for key, value in fractions.items()}))
                            comp_rows.append(dict(zip(comp_fields, [kind, index, name, layer, *counts[n],
                                                                    fg_area[n]])))
                            if index in config['pet']['example_indices']:
                                example_rows.append(dict(model=kind, index=index, spec=name, layer=layer,
                                    tokens=''.join('1' if t else '0' for t in used[n].tolist())))
                composition.write(comp_rows)
                union.write(union_rows)
                if example_rows:
                    examples.write(example_rows)
                deletion.write(pet_deletion(model, kind, qk, norms, images, fg, valid, ids, labels,
                                            original, start, config, output, categories))
    update_checks(output, models=[kind for kind, _ in models])


def pet_deletion(model, kind, qk, norms, images, fg, valid, ids, labels, original, start, config,
                 output, categories):
    """V01 matched deletion, all 12 layers, on the MiTA-default plan of each model."""
    spec = resolve(MITA_SPEC, dict(scoring='query', summary_scale='mita'))
    scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
    summary = dict(summary_side=5, summary_scale='mita', landmark_mode='mita')
    base, scores = {}, {}
    for layer in range(1, 13):
        pattern, scores[layer] = base_pattern(spec, *qk[layer], scale, grid)
        base[layer] = Pattern(pattern.readsets, pattern.routes)
    plan = lambda patterns: {layer: Plan(token_mask=token_mask(patterns[layer]), **summary) for layer in patterns}
    rows = []
    with execute(model, plan(base)):
        executed = model(images)
    if kind == 'mita':
        error = (executed - original).abs().max().item()
        assert error < 1e-5, error
        update_checks(output, max_pet_frozen_native_error=error)
    rows += outcomes(ids, labels, original, executed, model=kind, category='', kind='baseline', seed=-1,
                     eligible=1, mean_removed_per_set=0.)
    zero = torch.zeros(len(ids), 1, dtype=torch.bool, device=common.DEVICE)
    for c, category in enumerate(categories):
        changed, doses = {}, {}
        eligible = torch.ones(len(ids), dtype=torch.bool, device=common.DEVICE)
        for layer in range(1, 13):
            sink = sink_tokens(norms[layer], config['pet']['sink_threshold'])
            mask = sink if category == 'sink' else (fg & valid if category == 'foreground' else ~fg & valid & ~sink)
            changed[layer], feasible, doses[layer] = replace_category(base[layer], scores[layer],
                                                                      torch.cat([zero, mask], -1))
            eligible &= feasible
        dose = torch.stack([d.float().mean((1, 2)) for d in doses.values()], -1).mean(-1).tolist()
        with execute(model, plan(changed)):
            target = model(images)
        for row, ok, removed in zip(outcomes(ids, labels, original, target, model=kind, category=category,
                                             kind='target', seed=-1), eligible.tolist(), dose):
            rows.append(row | dict(eligible=int(ok), mean_removed_per_set=removed))
        for seed in config['random_seeds']:
            randomized = {layer: replace_random_patches(base[layer], scores[layer], doses[layer],
                          batch_generator(common.DEVICE, seed * 10 + c, layer, 25, start))
                          for layer in range(1, 13)}
            with execute(model, plan(randomized)):
                random_out = model(images)
            for row, ok, removed in zip(outcomes(ids, labels, original, random_out, model=kind,
                                                 category=category, kind='random', seed=seed),
                                        eligible.tolist(), dose):
                rows.append(row | dict(eligible=int(ok), mean_removed_per_set=removed))
    return rows


STAGE_FUNCTIONS = dict(noise=stage_noise, rq1=stage_rq1, rq2=stage_rq2, repro=stage_repro,
                       mita_repro=stage_mita_repro, mita_anatomy=stage_mita_anatomy, pet=stage_pet)

