"""Run one V2 anatomy stage (design v2 section 7) on the fixed data; resumable by batch."""

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root, project_path
ROOT = configure_runtime()

import numpy as np
import timm
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mita_probe.anatomy import Plan, capture_qk, coverage, execute, token_mask, union_fraction
from mita_probe.assets import MODEL_REVISION
from mita_probe.data import ImageNetValidation, experiment_indices
from mita_probe.io import load_config, source_commit, write_json
from mita_probe.metrics import kl_divergence
from mita_probe.mita_model import CHECKPOINT_SHA256, PREPROCESSING, load_mita_model
from mita_probe.models import load_model
from mita_probe.pet import PetSegmentation
from mita_probe.probe import ReadingPattern
from mita_probe.rq3_extension import capture_input_norms, replace_category, sink_tokens
from mita_probe.settings import base_pattern, batch_generator, expand_rq2, make_plan, resolve
from mita_probe.validation import replace_random_patches

STAGES = ['noise', 'rq1', 'rq2', 'mita_repro', 'mita_anatomy', 'pet']
OUTCOME = ['index', 'target', 'original', 'executed', 'agree', 'kl']
DEVICE = 'cuda'  # tests set 'cpu' to exercise the stages without a GPU
MITA_SPEC = dict(mode='L4', m_side=5, k=25, routing='dot', cls='routed', landmark_mode='mita')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = load_config(ROOT / 'configs/anatomy_v2.yaml')
    commit = source_commit(ROOT)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume:
        (output / 'config.yaml').write_text(yaml.safe_dump(config | dict(stage=args.stage,
                                            smoke_images=args.smoke)), encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.FileHandler(output / 'log.txt'), logging.StreamHandler()])
    try:
        run(args, config, output, commit)
    except Exception:
        logging.exception('Stage failed; preserve this directory and resume with --resume')
        raise


class Sink:
    """Append-only CSV written one whole batch at a time."""

    def __init__(self, path, fields):
        fresh = not path.exists()
        self.stream = path.open('a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        if fresh:
            self.writer.writeheader()

    def write(self, rows):
        self.writer.writerows(rows)
        self.stream.flush()


def outcomes(ids, labels, original, executed, **fields):
    assert torch.isfinite(executed).all()
    kl = kl_divergence(original.double(), executed.double()).tolist()
    predicted, reference = executed.argmax(-1).tolist(), original.argmax(-1).tolist()
    return [dict(index=index, target=labels[n], original=reference[n], executed=predicted[n],
                 agree=int(predicted[n] == reference[n]), kl=kl[n], **fields)
            for n, index in enumerate(ids)]


def full_plans(n_layers, plan):
    return {layer: plan(layer) for layer in range(1, n_layers + 1)}


def run(args, config, output, commit):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available(), 'Formal inference requires the authorized remote GPU'
    device = DEVICE
    deit = load_model(device)
    mita = None
    if args.stage in {'noise', 'mita_repro', 'mita_anatomy', 'pet'}:
        path = project_path(config['mita']['checkpoint'])
        import hashlib
        assert hashlib.sha256(path.read_bytes()).hexdigest() == CHECKPOINT_SHA256
        mita = load_mita_model(path, device)
    progress_path = output / 'progress.json'
    done = json.loads(progress_path.read_text())['done'] if args.resume and progress_path.exists() else {}
    write_json(output / f'metadata{"_resume" if done else ""}.json', dict(
        git_commit=commit, stage=args.stage, timestamp=datetime.now().astimezone().isoformat(),
        torch=torch.__version__, timm=timm.__version__, device=torch.cuda.get_device_name(),
        dtype='float32, no autocast', deit_revision=MODEL_REVISION, mita_sha256=CHECKPOINT_SHA256,
        batch_size=config['batch_size'], smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = make_batches(config, done, progress_path, started)

    def imagenet(model_kind, n=None, shift=0):
        pre = PREPROCESSING if model_kind == 'mita' else None
        dataset = ImageNetValidation(data_root() / 'imagenet_val', deit, preprocessing=pre, shift=shift)
        if n == 50000:
            indices = list(range(50000))
        else:
            indices = experiment_indices(len(dataset), config['n_images'], config['subset_seed'],
                                         project_path(config['indices']))
        indices = indices[:args.smoke] if args.smoke else indices
        return dataset, indices

    STAGE_FUNCTIONS[args.stage](args, config, output, deit, mita, data, batches)
    write_json(output / 'checks.json', dict(complete=True, elapsed_seconds=time.perf_counter() - started,
                                            **json.loads((output / 'running_checks.json').read_text())))
    logging.info('Complete: %s', args.stage)


def make_batches(config, done, progress_path, started):
    def batches(name, dataset, indices):
        """Yield (start, ids, batch) for batches not yet completed; record completion after use."""
        start = 0
        for batch in DataLoader(Subset(dataset, indices), batch_size=config['batch_size'],
                                num_workers=config['num_workers'], shuffle=False):
            ids = indices[start:start + len(batch[0])]
            if start >= done.get(name, 0):
                yield start, ids, batch
                done[name] = start + len(ids)
                write_json(progress_path, dict(done=done))
                if done[name] % 640 < len(ids) or done[name] == len(indices):
                    logging.info('%s %d/%d images; %.1fs', name, done[name], len(indices),
                                 time.perf_counter() - started)
            start += len(ids)
    return batches


def update_checks(output, **values):
    path = output / 'running_checks.json'
    checks = json.loads(path.read_text()) if path.exists() else {}
    for key, value in values.items():
        checks[key] = max(checks.get(key, value), value) if key.startswith('max_') else value
    write_json(path, checks)


def dense_identity(model, images, original):
    """Captured Q/K leave logits unchanged; dense attention over all keys reproduces them."""
    with capture_qk(model) as qk:
        observed = model(images)
    assert torch.equal(observed, original)
    q = qk[1][0]
    full = torch.ones(*q.shape[:3], q.shape[2], dtype=torch.bool, device=q.device)
    with execute(model, full_plans(len(model.blocks), lambda layer: Plan(token_mask=full))):
        dense = model(images)
    error = (dense - original).abs().max().item()
    assert error < 1e-4 and torch.equal(dense.argmax(-1), original.argmax(-1)), error
    return qk, error


def stage_noise(args, config, output, deit, mita, data, batches):
    rows = Sink(output / 'predictions.csv', ['model'] + OUTCOME)
    for kind, model in [('deit', deit), ('mita', mita)]:
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
                original = model(batch[0].to(DEVICE))
                rows.write(outcomes(ids, batch[1].tolist(), original, model(moved[0].to(DEVICE)), model=kind))
    update_checks(output, models=['deit', 'mita'])


def stage_rq1(args, config, output, deit, mita, data, batches):
    dataset, indices = data.imagenet('deit')
    main = config['main']
    k_main = config['rq1']['k'][0]
    sweep = config['rq2']['k_sweep']
    rows = Sink(output / 'predictions.csv', OUTCOME + ['layer', 'setting', 'seed'])
    cov_fields = ['index', 'layer'] + [f'{name}_k{k}_{part}' for k in sweep
                  for name in ['L1', 'L3_spatial', 'L3_dot', 'rand'] for part in ['patch', 'cls']] + [
                  f'{name}_k{k}_union' for k in sweep for name in ['L3_spatial', 'L3_dot']]
    cover = Sink(output / 'coverage.csv', cov_fields)
    grid = deit.patch_embed.grid_size
    scale = deit.blocks[0].attn.scale
    settings = [('L3', dict(mode='L3')), ('L4', dict(mode='L4')), ('L2', dict(mode='L2')),
                ('const', dict(mode='L2', m_side=1)), ('drop', dict(mode='drop'))]
    settings += [(f'L1_k{k}', dict(mode='L1', k=k)) for k in config['rq1']['per_query_k']]
    settings += [(f'{mode}_s{seed}', dict(mode=mode, seed=seed))
                 for seed in config['random_seeds'] for mode in ['rand', 'randC']]
    with torch.inference_mode():
        for start, ids, batch in batches('rq1', dataset, indices):
            images, labels = batch[0].to(DEVICE), batch[1].tolist()
            original = deit(images)
            qk, error = dense_identity(deit, images, original)
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
                for name, spec in settings:
                    spec = resolve(dict(k=k_main) | spec, main)
                    plan = make_plan(spec, q, k, scale, grid, layer=layer, batch_start=start)
                    with execute(deit, {layer: plan}):
                        out += outcomes(ids, labels, original, deit(images), layer=layer, setting=name,
                                        seed=spec.get('seed', -1))
            rows.write(out)
    update_checks(output, settings_per_layer=len(settings))


def stage_rq2(args, config, output, deit, mita, data, batches):
    dataset, indices = data.imagenet('deit')
    specs = expand_rq2(config)
    with (output / 'settings.csv').open('w', newline='') as stream:
        fields = sorted({key for spec in specs for key in spec})
        writer = csv.DictWriter(stream, fieldnames=['setting_id'] + fields)
        writer.writeheader()
        writer.writerows(dict(setting_id=i, **spec) for i, spec in enumerate(specs))
    rows = Sink(output / 'predictions.csv', OUTCOME + ['setting_id'])
    grid, scale = deit.patch_embed.grid_size, deit.blocks[0].attn.scale
    with torch.inference_mode():
        for start, ids, batch in batches('rq2', dataset, indices):
            images, labels = batch[0].to(DEVICE), batch[1].tolist()
            original = deit(images)
            qk, error = dense_identity(deit, images, original)
            update_checks(output, max_identity_error=error)
            out = []
            for setting_id, spec in enumerate(specs):
                plans = full_plans(12, lambda layer: make_plan(spec, *qk[layer], scale, grid,
                                                               layer=layer, batch_start=start))
                with execute(deit, plans):
                    out += outcomes(ids, labels, original, deit(images), setting_id=setting_id)
            rows.write(out)
    update_checks(output, settings=len(specs))


def mita_frozen(model, qk, start, spec_overrides=None):
    spec = resolve(MITA_SPEC | (spec_overrides or {}), dict(scoring='query', summary_scale='mita'))
    scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
    return {layer: make_plan(spec, *qk[layer], scale, grid, layer=layer, batch_start=start)
            for layer in range(1, 13)}


def stage_mita_repro(args, config, output, deit, mita, data, batches):
    dataset, indices = data.imagenet('mita', n=50000)
    rows = Sink(output / 'predictions.csv', ['index', 'target', 'predicted', 'correct'])
    with torch.inference_mode():
        for start, ids, batch in batches('mita_repro', dataset, indices):
            predicted = mita(batch[0].to(DEVICE)).argmax(-1).tolist()
            labels = batch[1].tolist()
            rows.write([dict(index=i, target=labels[n], predicted=predicted[n],
                             correct=int(predicted[n] == labels[n])) for n, i in enumerate(ids)])
    import pandas as pd
    accuracy = pd.read_csv(output / 'predictions.csv').drop_duplicates('index')['correct'].mean() * 100
    update_checks(output, top1=accuracy, reported_top1=config['mita']['reported_top1'])
    if not args.smoke:
        assert abs(accuracy - config['mita']['reported_top1']) <= config['mita']['reproduce_tolerance_pp'], accuracy


def stage_mita_anatomy(args, config, output, deit, mita, data, batches):
    dataset, indices = data.imagenet('mita')
    rows = Sink(output / 'predictions.csv', OUTCOME + ['setting', 'seed'])
    settings = [('L4_frozen', {}), ('L3', dict(mode='L3')), ('L2', dict(mode='L2')),
                ('L1_k25', dict(mode='L1'))]
    settings += [(f'{mode}_s{seed}', dict(mode=mode, seed=seed))
                 for seed in config['random_seeds'] for mode in ['rand', 'randC']]
    with torch.inference_mode():
        for start, ids, batch in batches('mita_anatomy', dataset, indices):
            images, labels = batch[0].to(DEVICE), batch[1].tolist()
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


def stage_pet(args, config, output, deit, mita, data, batches):
    dataset, indices = data.pet()
    comp_fields = ['model', 'index', 'spec', 'layer', 'cls', 'sink', 'foreground', 'background', 'unknown',
                   'fg_area']
    composition = Sink(output / 'composition.csv', comp_fields)
    examples = Sink(output / 'examples.csv', ['model', 'index', 'spec', 'layer', 'tokens'])
    deletion = Sink(output / 'deletion.csv', ['model'] + OUTCOME + ['category', 'kind', 'seed',
                                                                    'eligible', 'mean_removed_per_set'])
    categories = config['pet']['deletion_categories']
    for kind, model in [('deit', deit), ('mita', mita)]:
        scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
        with torch.inference_mode():
            for start, ids, batch in batches(f'pet_{kind}', dataset, indices):
                images, fg, valid = (part.to(DEVICE) for part in batch[:3])
                labels = [-1] * len(ids)
                with capture_qk(model) as qk, capture_input_norms(model) as norms:
                    original = model(images)
                fg_area = ((fg & valid).sum(-1) / valid.sum(-1).clamp_min(1)).tolist()
                comp_rows, example_rows = [], []
                for layer in range(1, 13):
                    sink = sink_tokens(norms[layer], config['pet']['sink_threshold'])
                    labels_tok = torch.full((len(ids), 197), 4, device=DEVICE)
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
                        for n, index in enumerate(ids):
                            comp_rows.append(dict(zip(comp_fields, [kind, index, name, layer, *counts[n],
                                                                    fg_area[n]])))
                            if index in config['pet']['example_indices']:
                                used = torch.zeros(197, dtype=torch.bool, device=DEVICE)
                                used[pattern.readsets[n].flatten()] = True
                                example_rows.append(dict(model=kind, index=index, spec=name, layer=layer,
                                    tokens=''.join('1' if t else '0' for t in used[1:].tolist())))
                composition.write(comp_rows)
                if example_rows:
                    examples.write(example_rows)
                deletion.write(pet_deletion(model, kind, qk, norms, images, fg, valid, ids, labels,
                                            original, start, config, output, categories))
    update_checks(output, models=['deit', 'mita'])


def pet_deletion(model, kind, qk, norms, images, fg, valid, ids, labels, original, start, config,
                 output, categories):
    """V01 matched deletion, all 12 layers, on the MiTA-default plan of each model."""
    spec = resolve(MITA_SPEC, dict(scoring='query', summary_scale='mita'))
    scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
    summary = dict(summary_side=5, summary_scale='mita', landmark_mode='mita')
    base, scores = {}, {}
    for layer in range(1, 13):
        pattern, scores[layer] = base_pattern(spec, *qk[layer], scale, grid)
        base[layer] = ReadingPattern(pattern.readsets, pattern.routes)
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
    zero = torch.zeros(len(ids), 1, dtype=torch.bool, device=DEVICE)
    for c, category in enumerate(categories):
        changed, doses = {}, {}
        eligible = torch.ones(len(ids), dtype=torch.bool, device=DEVICE)
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
                          batch_generator(DEVICE, seed * 10 + c, layer, 25, start))
                          for layer in range(1, 13)}
            with execute(model, plan(randomized)):
                random_out = model(images)
            for row, ok, removed in zip(outcomes(ids, labels, original, random_out, model=kind,
                                                 category=category, kind='random', seed=seed),
                                        eligible.tolist(), dose):
                rows.append(row | dict(eligible=int(ok), mean_removed_per_set=removed))
    return rows


STAGE_FUNCTIONS = dict(noise=stage_noise, rq1=stage_rq1, rq2=stage_rq2, mita_repro=stage_mita_repro,
                       mita_anatomy=stage_mita_anatomy, pet=stage_pet)

if __name__ == '__main__':
    main()
