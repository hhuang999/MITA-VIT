"""E-C2 (decision B45; exploratory, post hoc): why does relocation (C2) make the reading worse?

Uses only saved material: the v5 position probes and blur widths (diagnose_probe run), the 2,000 test held-out
images of that run (rows 2,001-4,000 of heldout.json, not used to train the probes), the R0 mean maps
(diagnose_m1 run), and ImageNet-S. Same models, preprocessing and code as the v5 formal runs.

1. probe_errors.csv: per test image and layer 6-12 (mean over its 196 patch tokens), for the probe and for the
   matched Gaussian blur: exact accuracy, top-5 accuracy, expected distance error sum_p P(p) ||p - j|| (patches),
   mass in the 3 x 3 neighbourhood of j, mass farther than 3 patches, and the far share of the errors
   (mass farther than 3 / (1 - P(j))).
2. mean_prediction.npz: per layer, each test image's mean predicted distribution over the 196 positions
   (float32 [images, 196]); averaged over images it is the probe's image-independent position preference.
3. shift.csv: per ImageNet-S image, for layers 9-12 summed: object share of R0, C2 and the matched blur; the
   share of the total score in each distance bin (floor of the distance to the grid centre, 0-9 patches); and
   Pearson and Spearman correlations of (C2 - R0) and (blur - R0) with the model mean map and with closeness to
   the centre.
"""

import json
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import readings
from ..identity import collect
from ..io import write_json
from ..metrics import spearman
from ..models import load_model
from ..runtime import project_path
from . import common, datasets
from .common import Sink
from .diagnose import HeldoutImages, preprocessing_of

PROBE_LAYERS = list(range(6, 13))
DEEP = [9, 10, 11, 12]
N_BINS = 10
ERRORS = ['exact', 'top5', 'distance', 'near3x3', 'far3']


def distances(grid=14, device=None):
    """[patches, positions] Euclidean grid distance in patches."""
    return readings.squared_distances(grid, device).sqrt()


def closeness(grid=14, device=None):
    """[positions] minus the distance of each patch centre to the grid centre."""
    xy = readings.grid_positions(grid, device)
    return -((xy - (grid - 1) / 2) ** 2).sum(-1).sqrt()


def distance_bins(grid=14, device=None):
    """[positions] floor of the distance to the grid centre (0-9 for a 14 x 14 grid)."""
    return (-closeness(grid, device)).floor().long()


def error_stats(probs, grid=14):
    """probs [B, patches, positions] (row j: P(position | token j)) -> dict of [B] means over the tokens."""
    dist = distances(grid, probs.device)
    patches = probs.shape[-2]
    own = torch.arange(patches, device=probs.device)
    correct = probs[..., own, own]                      # P(j | token j), [B, patches]
    top5 = probs.topk(5, dim=-1).indices
    near = (dist <= 1.5).float()                        # 3 x 3 neighbourhood (diagonal distance sqrt 2)
    far_mass = (probs * (dist > 3).float()).sum(-1)
    # Error mass as the sum of the off-position probabilities in float64 (B47): 1 - P(j) in float32 rounds to
    # ~0 for well-located tokens and made the ratio blow up (> 1). Tokens with no error mass are left out.
    off = probs.double() * (1 - torch.eye(patches, device=probs.device, dtype=torch.float64))
    errors = off.sum(-1)
    far_double = (off * (dist > 3).double()).sum(-1)
    share = torch.where(errors > 0, far_double / errors.clamp_min(1e-300), torch.full_like(errors, float('nan')))
    return dict(
        exact=(probs.argmax(-1) == own).float().mean(-1),
        top5=(top5 == own[:, None]).any(-1).float().mean(-1),
        distance=(probs * dist).sum(-1).mean(-1),
        near3x3=(probs * near).sum(-1).mean(-1),
        far3=far_mass.mean(-1),
        far_share=share.nanmean(-1).float(),
    )


def pearson(a, b):
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def shift_stats(r0, moved, mean_map, foreground, valid, grid=14):
    """r0, moved [B, positions] (moved = C2 or blur); per-image object shares, distance-bin shares, correlations."""
    fg = (foreground & valid).to(r0.dtype)
    v = valid.to(r0.dtype)

    def object_share(scores):
        return (scores * fg).sum(-1) / (scores * v).sum(-1).clamp_min(1e-12)

    bins = distance_bins(grid, r0.device)
    one_hot = torch.nn.functional.one_hot(bins, N_BINS).to(r0.dtype)    # [positions, bins]

    def bin_share(scores):
        return (scores @ one_hot) / scores.sum(-1, keepdim=True).clamp_min(1e-12)

    delta = moved - r0
    near = closeness(grid, r0.device).to(r0.dtype).expand_as(delta)
    base = mean_map.expand_as(delta)
    return dict(object_r0=object_share(r0), object_moved=object_share(moved),
                bins_r0=bin_share(r0), bins_moved=bin_share(moved),
                pearson_mean=pearson(delta, base), spearman_mean=spearman(delta, base),
                pearson_centre=pearson(delta, near), spearman_centre=spearman(delta, near))


def probe_part(args, config, output, model, device, probes, sigmas):
    folder = project_path(config['diagnose']['probe_run'].format(model=args.model))
    held = json.loads((folder / 'heldout.json').read_text())
    rows = held['rows'][held['probe_images']:]
    rows = rows[:args.smoke] if args.smoke else rows
    loader = DataLoader(HeldoutImages(datasets.imagenet_files(), rows, preprocessing_of(model, args.model)),
                        batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
    sink = Sink(output / 'probe_errors.csv', ['model', 'row', 'layer', 'source'] + ERRORS + ['far_share'])
    blur_rows = {layer: readings.gaussian_relocation(sigmas[layer], device=device) for layer in PROBE_LAYERS}
    mean_prediction = {layer: [] for layer in PROBE_LAYERS}
    start = 0
    for images in loader:
        ids = rows[start:start + len(images)]
        start += len(images)
        tokens = collect(model, images.to(device), PROBE_LAYERS)
        out = []
        for layer in PROBE_LAYERS:
            with torch.no_grad():
                probs = probes[layer].probs(tokens[layer])
            mean_prediction[layer].append(probs.mean(1).cpu().numpy())
            for source, values in (('probe', error_stats(probs)),
                                   ('blur', error_stats(blur_rows[layer].expand(len(ids), -1, -1)))):
                values = {key: value.tolist() for key, value in values.items()}
                out += [dict(model=args.model, row=row, layer=layer, source=source,
                             **{key: values[key][n] for key in values}) for n, row in enumerate(ids)]
        sink.write(out)
    np.savez_compressed(output / 'mean_prediction.npz',
                        **{f'layer{layer}': np.concatenate(value) for layer, value in mean_prediction.items()})


def shift_part(args, config, output, model, device, lens):
    dataset, indices = datasets.build('imagenet_s', model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    mean_map = lens.mean('R0', DEEP)
    fields = ['model', 'index', 'source', 'object_r0', 'object_moved', 'pearson_mean', 'spearman_mean',
              'pearson_centre', 'spearman_centre'] + [f'bin{b}_r0' for b in range(N_BINS)] + \
             [f'bin{b}_moved' for b in range(N_BINS)]
    sink = Sink(output / 'shift.csv', fields)
    entries = [dict(name='R0', reading='R0', layers=DEEP), dict(name='C2', reading='C2', layers=DEEP),
               dict(name='blur', reading='blur', layers=DEEP)]
    progress_path, done = common.progress(output, args.resume)
    batches = common.make_batches(config, done, progress_path, time.perf_counter())
    for start, ids, batch in batches(f'shift_{args.model}', dataset, indices):
        images, foreground, valid = (tensor.to(device) for tensor in batch[:3])
        with torch.no_grad(), readings.capture(model, DEEP) as store:
            model(images)
            maps = readings.compute(store, entries, lens)
        out = []
        for source in ('C2', 'blur'):
            stats = shift_stats(maps['R0'].double(), maps[source].double(), mean_map.double(), foreground, valid)
            stats = {key: value.cpu().tolist() for key, value in stats.items()}
            for n, index in enumerate(ids):
                row = dict(model=args.model, index=index, source=source,
                           **{key: stats[key][n] for key in ('object_r0', 'object_moved', 'pearson_mean',
                                                              'spearman_mean', 'pearson_centre', 'spearman_centre')})
                row.update({f'bin{b}_r0': stats['bins_r0'][n][b] for b in range(N_BINS)})
                row.update({f'bin{b}_moved': stats['bins_moved'][n][b] for b in range(N_BINS)})
                out.append(row)
        sink.write(out)


def run(args, config, output, commit):
    """args: model, smoke, resume. Lens paths come from config['correct']['lens_runs'] (the v5 formal runs).
    config['explore_c2']['parts'] (default both) selects 'probe' and / or 'shift' (B47: probe part rerun only)."""
    from .correct import load_lens
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    lens = load_lens(config['correct'], args.model, 'crop', device)
    write_json(output / 'metadata.json', common.provenance(commit, model=args.model, decision='B45 (exploratory)',
                                                           lens_runs=config['correct']['lens_runs'],
                                                           smoke_images=args.smoke))
    started = time.perf_counter()
    parts = config.get('explore_c2', {}).get('parts', ['probe', 'shift'])
    if 'probe' in parts:
        probe_part(args, config, output, model, device, lens.probes, lens.sigmas)
        logging.info('probe part done; %.1fs', time.perf_counter() - started)
    if 'shift' in parts:
        shift_part(args, config, output, model, device, lens)
    common.finish(output, started)
    logging.info('Complete: E-C2 for %s', args.model)
