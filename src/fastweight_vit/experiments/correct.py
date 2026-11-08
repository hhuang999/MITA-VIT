"""Correct (RQ3; Table 1): the lens readings, their stand-ins and the references on one dataset.

One run scores a list of maps on one dataset:
- `imagenet_s`: plausibility.csv (same-distance-ring enrichment, with the other measures of
  fastweight_vit.enrichment) on every image, and faithfulness.csv (patches removed most- and least-important
  first, at every budget; SRG in scripts/make_tables.py) on a fixed random subset (`faithfulness_images`,
  `subset_seed`; data/splits/imagenet_s_srg_2000.json lists it);
- `clickme`: alignment.csv, Spearman with the human map and with another image's human map (image-specific
  part = own - shuffled; fastweight_vit.metrics.alignment).

Map entries in `correct.methods`:
- dict(name, reading, layers): a lens reading (fastweight_vit.readings: R0, C1, C2, C3, R_all, R_all_alt and the
  stand-ins stand_in_blur, stand_in_no_high, stand_in_centre; mean_map), computed together from one forward;
- dict(name, method): a reference of fastweight_vit.references (gae, centre, random).
The lens (probes, blur widths) comes from the diagnose `probe` run and the mean maps from the diagnose
`mean_maps` run of the same model (`lens_runs`, with {model}); ClickMe uses the ClickMe-geometry mean maps.

`variant` changes the model instead of the reading (Table 1, "change the model" rows), with the R0 reading only:
- `register`: DeiT-S with a test-time register (the FFN units in `gate`, found by the register search, shifted to
  an added register token), prefix = CLS + register;
- `mita`: MiTA-DeiT-S (top-k training), attention with summary-slot weight passed back to tokens
  (fastweight_vit.instrument).
The register variant has the faithfulness part too: the register token is inserted at norm_pre, before
tokens are dropped, so it is kept like CLS (prefix 2). MiTA has none: its summary slots pool fixed regions of
the full 14 x 14 grid (5 x 5 landmarks), which are undefined once patch tokens are removed.
Maps listed in `no_srg` skip the faithfulness part (the early-layer negative control is judged on enrichment
and ClickMe only; docs/preregistration.md).

A seeded gallery of `gallery_images` images keeps every map, and for ClickMe the image and the human map, in
gallery_<start>.npz (figure 2(b): the two lowest-indexed of the 24 drawn images).
"""

import logging
import time
from contextlib import nullcontext

import numpy as np
import torch
from timm.data import resolve_model_data_config

from .. import enrichment, faithfulness, readings, references
from ..data.build import build
from ..data.clickme import ClickMe
from ..faithfulness import curve, rank
from ..instrument import instrument
from ..io import write_json
from ..metrics import ALIGNMENT, alignment
from ..models import IMAGENET_PREPROCESSING, data_reference, load_model
from ..registers import as_layers, edit_neurons
from ..runtime import data_root, project_path
from . import common
from .common import Sink

VARIANTS = (None, 'register', 'mita')


def load_lens(settings, model_name, geometry, device):
    probe_run = project_path(settings['lens_runs']['probe'].format(model=model_name))
    mean_run = project_path(settings['lens_runs']['mean_maps'].format(model=model_name))
    probes, sigmas = readings.load_probes(probe_run / 'probes.npz')
    means = readings.load_means(mean_run / f'means_{geometry}.npz')
    return readings.Lens(probes, sigmas, means, settings['norm_ratio']).to(device)


def all_maps(model, images, entries, lens, generator=None, target=None, prefix=None, start=0):
    """{name: [B, 196] float} for lens readings (one forward) and references."""
    lens_entries = [entry for entry in entries if 'reading' in entry]
    out = {}
    if lens_entries:
        with torch.no_grad(), readings.capture(model, readings.needed_layers(lens_entries), prefix) as store:
            model(images)
            out.update(readings.compute(store, lens_entries, lens, start=start))
    for entry in entries:
        if 'reading' not in entry:
            out[entry['name']] = references.compute(entry['method'], model, images, layers=entry.get('layers'),
                                                    generator=generator, target=target)
    return {name: value.float() for name, value in out.items()}


def variant_maps(model, images, entries, variant, gate, device):
    """R0 readings of a changed model (register, MiTA) from the attention instrument."""
    assert all(entry.get('reading') == 'R0' for entry in entries), 'Variant runs take the R0 reading only'
    layers = readings.needed_layers(entries)
    if variant == 'register':
        context, prefix = edit_neurons(model, as_layers(gate['neurons'], device), mode='shift', register=True), 2
    else:
        context, prefix = nullcontext(), model.num_prefix_tokens
    with torch.no_grad(), context, instrument(model, layers) as store:
        logits = model(images)
    out = {entry['name']: sum(store[layer]['received'].sum(1)[:, prefix:] for layer in entry['layers']).float()
           for entry in entries}
    return out, logits


def gallery_rows(indices, n, seed):
    rng = np.random.default_rng(seed)
    return set(np.asarray(indices)[rng.choice(len(indices), min(n, len(indices)), replace=False)].tolist())


def run(args, config, output, commit):
    settings = config['correct']
    variant = settings.get('variant')
    assert variant in VARIANTS, variant
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    entries = settings['methods']
    gate = dict(neurons=settings['register_units']) if variant == 'register' else None  # (layer, unit) pairs
    clickme = settings['dataset'] == 'clickme'
    lens = None if variant else load_lens(settings, args.model, 'clickme' if clickme else 'crop', device)
    if clickme:
        pre = IMAGENET_PREPROCESSING[args.model] or resolve_model_data_config(data_reference(model))
        dataset = ClickMe(data_root() / 'clickme', pre['mean'], pre['std'], expected_shards=settings['shards'])
        indices = list(range(len(dataset)))
    else:
        dataset, indices = build(settings['dataset'], model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    rng = np.random.default_rng(settings['subset_seed'])
    n_faith = 0 if clickme or variant == 'mita' else min(settings['faithfulness_images'], len(indices))
    faithful = set(np.asarray(indices)[rng.choice(len(indices), n_faith, replace=False)].tolist()) if n_faith else set()
    shown = gallery_rows(indices, settings['gallery_images'], settings['subset_seed'])
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, variant=variant, dataset=settings['dataset'], images=len(indices),
        faithfulness_images=n_faith, methods=[entry['name'] for entry in entries], lens_runs=settings.get('lens_runs'),
        smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    if clickme:
        scores_sink = Sink(output / 'alignment.csv', ['model', 'index', 'label', 'predicted', 'method'] + ALIGNMENT)
    else:
        scores_sink = Sink(output / 'plausibility.csv', ['model', 'index', 'method'] + enrichment.FIELDS)
        faith = Sink(output / 'faithfulness.csv', faithfulness.FIELDS)
    gallery = {}
    for start, ids, batch in batches(f'correct_{args.model}', dataset, indices):
        images = batch[0].to(device)
        noise = torch.Generator(device=device).manual_seed(settings['random_seed'] * 1_000_003 + start)
        if variant:
            maps, original = variant_maps(model, images, entries, variant, gate, device)
        else:
            with torch.no_grad():
                original = model(images)
            maps = all_maps(model, images, entries, lens, generator=noise, start=start)
        rows = []
        if clickme:
            human = batch[1].to(device)
            predicted = original.argmax(-1).tolist()
            for name, scores in maps.items():
                values = {key: value.tolist() for key, value in alignment(scores, human).items()}
                rows += [dict(model=args.model, index=index, label=int(batch[2][n]), predicted=predicted[n],
                              method=name, **{key: values[key][n] for key in ALIGNMENT}) for n, index in enumerate(ids)]
        else:
            foreground, valid = batch[1].to(device), batch[2].to(device)
            other = foreground.roll(1, 0) if len(ids) > 1 else torch.zeros_like(foreground)
            for name, scores in maps.items():
                values = enrichment.measures(scores, foreground, valid, other,
                                               generator=torch.Generator().manual_seed(start))
                rows += [dict(model=args.model, index=index, method=name,
                              **{key: float(values[key][n]) for key in enrichment.FIELDS}) for n, index in enumerate(ids)]
            chosen = [n for n, index in enumerate(ids) if index in faithful]
            if chosen:
                sub = torch.tensor(chosen, device=device)
                tie = torch.Generator(device=device).manual_seed(settings['tie_seed'] * 1_000_003 + start)
                out = []
                context = edit_neurons(model, as_layers(gate['neurons'], device), mode='shift', register=True) \
                    if variant == 'register' else nullcontext()
                prefix = 2 if variant == 'register' else None
                with torch.no_grad(), context:
                    for name, scores in maps.items():
                        if name in settings.get('no_srg', []):
                            continue
                        order = rank(scores[sub], tie)
                        for condition in settings['conditions']:
                            for budget, values in curve(model, images[sub], original[sub], order, settings['budgets'],
                                                        condition, prefix).items():
                                out += faithfulness.rows(args.model, [ids[n] for n in chosen], name, condition, budget,
                                                         values)
                faith.write(out)
        scores_sink.write(rows)
        for n, index in enumerate(ids):
            if index in shown:
                gallery.setdefault('index', []).append(index)
                for name, scores in maps.items():
                    gallery.setdefault(f'map_{name}', []).append(scores[n].cpu().numpy())
                if clickme:
                    image, heatmap, _ = dataset.record(index)
                    gallery.setdefault('image', []).append(np.asarray(image.resize((224, 224))))
                    gallery.setdefault('human', []).append(batch[1][n].numpy())
                else:
                    gallery.setdefault('object', []).append(batch[1][n].numpy())
                    gallery.setdefault('valid', []).append(batch[2][n].numpy())
        if gallery:
            np.savez_compressed(output / f'gallery_{start:06d}.npz',
                                **{key: np.stack(value) for key, value in gallery.items()})
            gallery = {}
    common.finish(output, started)
    logging.info('Complete: correct on %s for %s (variant %s)', settings['dataset'], args.model, variant)
