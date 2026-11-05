"""v5 step 2, diagnose (decisions B36-B38): held-out images, position probes, mean maps and the M1 share.

Stage `probe` (diagnose_probe):
- selects the held-out images (data.heldout; the same for every model): the first `probe_images` train the
  position probes, the next `mean_images` are used for everything else (probe test accuracy, C2's
  displacement and the matched blur width, the mean maps of stage m1); heldout.json;
- one linear position probe per layer on the normalised block input (the E3 probe and training, fitted on
  the larger held-out set and kept for C2); probes.npz holds the probes and, per layer, the mean squared
  displacement of C2 on the test images and the matching Gaussian blur width sigma;
- identity.csv: per layer test accuracy (chance 1/196), mean grid distance of the errors, mean top
  probability, displacement and sigma.

Stage `m1` (diagnose_m1), after `probe`:
- mean maps of R0, C3, C2 (relocated R0) and C2C3 (relocated C3) at every layer, averaged over the
  held-out test images, under two geometries: the model's ImageNet evaluation crop (for ImageNet-S) and the
  ClickMe storage format (for ClickMe); means_crop.npz, means_clickme.npz; radial.csv (ring-explained share
  of each mean map);
- m1.csv.gz on the evaluation images (ImageNet-S): per image, layer and variant the residual and total sum
  of squares around each map's spatial mean; M1 share = 1 - sum(residual) / sum(total) (readings.m1_sums).
"""

import json
import logging
import time

import numpy as np
import torch
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset, Subset

from .. import readings
from ..data.clickme import ClickMe
from ..data.heldout import GEOMETRIES, clickme_like, select, thumbnail
from ..data.imagenet_s import ImageNetS
from ..io import write_csv, write_json
from ..models import IMAGENET_PREPROCESSING, data_reference, load_model
from ..runtime import data_root, project_path
from . import common, datasets
from .common import Sink

STAGES = ('probe', 'm1')


class HeldoutImages(Dataset):
    """ImageNet validation rows under the model's evaluation crop or the ClickMe storage format."""

    def __init__(self, files, rows, preprocessing, geometry=None):
        self.files, self.rows, self.geometry = files, rows, geometry
        self.transform = create_transform(**preprocessing, is_training=False)
        self.mean = torch.tensor(preprocessing['mean'], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(preprocessing['std'], dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, n):
        image = self.files.image_at(self.rows[n])
        if self.geometry is None:
            return self.transform(image)
        stored = clickme_like(image, self.geometry)
        x = torch.from_numpy(np.array(stored, dtype=np.float32)).permute(2, 0, 1)[None]
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)[0] / 255
        return (x - self.mean) / self.std


def preprocessing_of(model, name):
    from timm.data import resolve_model_data_config
    return IMAGENET_PREPROCESSING[name] or resolve_model_data_config(data_reference(model))


def heldout(settings, n_total, smoke):
    """Select (or, in a smoke run, select a few) held-out rows; returns (rows, info)."""
    files = datasets.imagenet_files()
    imagenet_s = ImageNetS(data_root() / 'imagenet_s' / 'ImageNetS919', files, None,
                           preprocessing=IMAGENET_PREPROCESSING['dino'])
    excluded = {imagenet_s.name(index) for index in range(len(imagenet_s))}
    clickme = ClickMe(data_root() / 'clickme', (0, 0, 0), (1, 1, 1))
    references = {}
    for index in range(len(clickme)):
        image, _, label = clickme.record(index)
        references.setdefault(label, []).append(thumbnail(image, 'squash'))
    references = {label: np.stack(value) for label, value in references.items()}
    order = np.random.default_rng(settings['heldout_seed']).permutation(len(files.names))
    rows, info = select(order, files.names, files.labels(), excluded, references, files.image_at, n_total,
                        settings['duplicate_threshold'])
    geometries = [entry['geometry'] for entry in info['clickme_duplicates']]
    info['clickme_geometry'] = max(GEOMETRIES, key=geometries.count) if geometries else 'squash'
    info['clickme_images'], info['imagenet_s_images'] = len(clickme), len(excluded)
    return files, rows, info


def collect_tokens(model, loader, layers, device):
    """{layer: [images, patches, dim] float16 on the CPU}: the normalised block inputs."""
    from ..identity import collect
    out = {layer: [] for layer in layers}
    for images in loader:
        tokens = collect(model, images.to(device), layers)
        for layer in layers:
            out[layer].append(tokens[layer].half().cpu())
    return {layer: torch.cat(value) for layer, value in out.items()}


def stage_probe(args, config, output, model, device):
    settings = config['diagnose']
    n_probe, n_mean = settings['probe_images'], settings['mean_images']
    if args.smoke:
        n_probe, n_mean = max(args.smoke - 8, 2), 8
    files, rows, info = heldout(settings, n_probe + n_mean, args.smoke)
    write_json(output / 'heldout.json', dict(rows=rows, names=[files.names[row] for row in rows],
                                             probe_images=n_probe, mean_images=n_mean, **info))
    logging.info('held-out: %d rows; %d ClickMe duplicates skipped; geometry %s', len(rows),
                 len(info['clickme_duplicates']), info['clickme_geometry'])
    layers = list(range(1, len(model.blocks) + 1))
    loader = DataLoader(HeldoutImages(files, rows, preprocessing_of(model, args.model)),
                        batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
    tokens = collect_tokens(model, loader, layers, device)
    probes, sigmas, displacements, table = {}, {}, {}, []
    for layer in layers:
        train, test = tokens[layer][:n_probe].to(device), tokens[layer][n_probe:].to(device)
        probe = readings.fit_position_probe(train, epochs=settings['epochs'], seed=settings['seed'])
        with torch.no_grad():
            probs = probe.probs(test)
        target = torch.arange(probs.shape[1], device=device).expand(probs.shape[0], -1)
        predicted = probs.argmax(-1)
        positions = readings.grid_positions(device=device)
        error = (positions[predicted] - positions[target]).norm(dim=-1)
        displacements[layer] = readings.displacement(probs)
        sigmas[layer] = readings.match_sigma(displacements[layer])
        probes[layer] = probe.to('cpu')
        table.append(dict(model=args.model, layer=layer, position_accuracy=float((predicted == target).float().mean()),
                          error_distance=float(error.mean()), top_probability=float(probs.max(-1).values.mean()),
                          displacement=displacements[layer], sigma=sigmas[layer]))
        logging.info('layer %d: accuracy %.3f, displacement %.3f, sigma %.3f', layer, table[-1]['position_accuracy'],
                     displacements[layer], sigmas[layer])
    readings.save_probes(output / 'probes.npz', probes, sigmas, displacements)
    write_csv(output / 'identity.csv', table)


def lens_folder(config, model):
    return project_path(config['diagnose']['probe_run'].format(model=model))


def stage_m1(args, config, output, model, device):
    settings = config['diagnose']
    folder = lens_folder(config, args.model)
    probes, sigmas = readings.load_probes(folder / 'probes.npz')
    held = json.loads((folder / 'heldout.json').read_text())
    rows = held['rows'][held['probe_images']:]
    rows = rows[:args.smoke] if args.smoke else rows
    lens = readings.Lens(probes, sigmas, norm_ratio=settings['norm_ratio']).to(device)
    layers = list(range(1, len(model.blocks) + 1))
    files = datasets.imagenet_files()
    pre = preprocessing_of(model, args.model)
    radial = []
    for geometry, name in ((None, 'crop'), (held['clickme_geometry'], 'clickme')):
        loader = DataLoader(HeldoutImages(files, rows, pre, geometry), batch_size=config['batch_size'],
                            num_workers=config['num_workers'], shuffle=False)
        means = readings.accumulate_means(model, loader, lens, layers, device)
        means = {variant: {layer: value.cpu() for layer, value in maps.items()} for variant, maps in means.items()}
        readings.save_means(output / f'means_{name}.npz', means)
        radial += [dict(model=args.model, geometry=name, variant=variant, layer=layer,
                        radial_share=readings.radial_share(means[variant][layer]))
                   for variant in readings.MEAN_VARIANTS for layer in layers]
        if name == 'crop':
            lens.means = {variant: {layer: value.to(device) for layer, value in maps.items()}
                          for variant, maps in means.items()}
    write_csv(output / 'radial.csv', radial)
    dataset, indices = datasets.build(settings['dataset'], model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    sink = Sink(output / 'm1.csv', ['model', 'index', 'layer', 'variant', 'residual', 'total'])
    progress_path, done = common.progress(output, args.resume)
    batches = common.make_batches(config, done, progress_path, time.perf_counter())
    for start, ids, batch in batches(f'm1_{args.model}', dataset, indices):
        out = []
        with torch.no_grad(), readings.capture(model, layers) as store:
            model(batch[0].to(device))
            for layer in layers:
                pieces = readings.layer_variants(store, layer, lens, set(readings.MEAN_VARIANTS))
                for variant in readings.MEAN_VARIANTS:
                    residual, total = readings.m1_sums(pieces[variant].double(), lens.means[variant][layer].double())
                    out += [dict(model=args.model, index=index, layer=layer, variant=variant,
                                 residual=float(residual[n]), total=float(total[n])) for n, index in enumerate(ids)]
        sink.write(out)


def run(args, config, output, commit):
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    write_json(output / 'metadata.json', common.provenance(commit, stage=args.stage, model=args.model,
                                                           settings=config['diagnose'], smoke_images=args.smoke))
    started = time.perf_counter()
    (stage_probe if args.stage == 'probe' else stage_m1)(args, config, output, model, device)
    common.finish(output, started)
    logging.info('Complete: diagnose %s for %s', args.stage, args.model)
