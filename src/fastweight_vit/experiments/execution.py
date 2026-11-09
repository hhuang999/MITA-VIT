"""Execution checks: does the model still predict the same class when it may read only the units the attention
map picks? (the execution test with its k sweep, and the single-layer checks)

On the fixed 5,000-image ImageNet validation subset (data/splits), per image and setting: top-1 agreement with the
intact model and KL(intact || restricted).

| stage | what is restricted | output |
| --- | --- | --- |
| k_sweep | all 12 layers at once read only their reading sets (m regions x k units), for every k of `sweep.k_sweep`; random sets of the same size (seeds `random_seeds`) as the control | predictions.csv + settings.csv |
| single_layer | one layer at a time reads only its reading sets (k = `single_layer.k`) | predictions.csv |
| attention_off | the attention branch of every layer (`all`) or of one layer at a time outputs zero; residual stream and FFNs unchanged | attention_off.csv |

The reading sets of a layer come from the model's own Q/K on the same image (execution.build_pattern): each
region's landmark (mean query) keeps its top-k keys, every patch query reads the set of its region (spatial
routing), and CLS reads its own top-k set. Before any restriction, `dense_identity` checks that the captured
Q/K and the explicit-softmax forward reproduce the model's logits.
"""

import csv
import logging
import time

import torch

from ..data.build import build
from ..execution import Plan, execute, expand_settings, make_plan, resolve
from ..io import write_json
from ..models import load_model
from . import common
from .common import OUTCOME, Sink, dense_identity, full_plans, outcomes, update_checks

STAGES = ('k_sweep', 'single_layer', 'attention_off')


def run(args, config, output, commit):
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    dataset, indices = build('imagenet', model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, stage=args.stage, model=args.model, images=len(indices), batch_size=config['batch_size'],
        preprocessing=config.get('preprocessing', 'model'), smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    STAGE_FUNCTIONS[args.stage](config, output, model, dataset, indices, batches)
    common.finish(output, started)
    logging.info('Complete: execution %s for %s', args.stage, args.model)


def stage_k_sweep(config, output, model, dataset, indices, batches):
    specs = expand_settings(config)
    with (output / 'settings.csv').open('w', newline='') as stream:
        fields = sorted({key for spec in specs for key in spec})
        writer = csv.DictWriter(stream, fieldnames=['setting_id'] + fields)
        writer.writeheader()
        writer.writerows(dict(setting_id=i, **spec) for i, spec in enumerate(specs))
    rows = Sink(output / 'predictions.csv', OUTCOME + ['setting_id'])
    grid, scale = model.patch_embed.grid_size, model.blocks[0].attn.scale
    with torch.inference_mode():
        for start, ids, batch in batches('k_sweep', dataset, indices):
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


def stage_single_layer(config, output, model, dataset, indices, batches):
    spec = resolve(dict(mode='L3', k=config['single_layer']['k']), config['main'])
    rows = Sink(output / 'predictions.csv', OUTCOME + ['layer', 'setting', 'seed'])
    grid, scale = model.patch_embed.grid_size, model.blocks[0].attn.scale
    with torch.inference_mode():
        for start, ids, batch in batches('single_layer', dataset, indices):
            images, labels = batch[0].to(common.DEVICE), batch[1].tolist()
            original = model(images)
            qk, error = dense_identity(model, images, original)
            update_checks(output, max_identity_error=error)
            out = []
            for layer in range(1, len(model.blocks) + 1):
                plan = make_plan(spec, *qk[layer], scale, grid, layer=layer, batch_start=start)
                with execute(model, {layer: plan}):
                    out += outcomes(ids, labels, original, model(images), layer=layer, setting='L3', seed=-1)
            rows.write(out)


def stage_attention_off(config, output, model, dataset, indices, batches):
    depth = len(model.blocks)
    settings = {'all': list(range(1, depth + 1))} | {str(layer): [layer] for layer in range(1, depth + 1)}
    sink = Sink(output / 'attention_off.csv', ['index', 'layers', 'target', 'original', 'executed', 'agree', 'kl'])
    for start, ids, (images, labels) in batches('attention_off', dataset, indices):
        images = images.to(common.DEVICE)
        labels = labels.tolist()
        rows = []
        with torch.no_grad():
            original = model(images)
            for name, layers in settings.items():
                with execute(model, {layer: Plan(drop=True) for layer in layers}):
                    edited = model(images)
                rows += common.outcomes(ids, labels, original, edited, layers=name)
        sink.write(rows)


STAGE_FUNCTIONS = dict(k_sweep=stage_k_sweep, single_layer=stage_single_layer, attention_off=stage_attention_off)
