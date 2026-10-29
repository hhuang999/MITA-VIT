"""Centre-prior control for the probe's union reading sets on Pet (decision A14); runs on CPU.

For every model, image, layer and composition spec, take the probe's union size U (from an earlier
anatomy pet run) and read the U patches nearest the image centre instead (ties broken uniformly at
random, in expectation). Both are scored the same way: foreground / (foreground + background +
high-norm), boundary-only patches excluded. High-norm tokens come from a local forward pass.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from ..data import PetSegmentation
from ..interventions import sink_tokens
from ..io import write_json
from ..models import data_reference, load_model
from ..probe import capture_input_norms
from ..runtime import ROOT, data_root
from ..stats import interval, resample_indices

GRID = 14
PARTS = ['foreground', 'background', 'sink', 'unknown']


def centre_weights(grid=GRID):
    """Row U: expected membership of each patch among the U patches nearest the centre."""
    rows, cols = np.divmod(np.arange(grid * grid), grid)
    distance = (rows - (grid - 1) / 2) ** 2 + (cols - (grid - 1) / 2) ** 2
    rings = [np.flatnonzero(distance == d) for d in np.unique(distance)]
    weights = np.zeros((grid * grid + 1, grid * grid))
    for size in range(1, grid * grid + 1):
        left = size
        for ring in rings:
            take = min(left, len(ring))
            weights[size, ring] = take / len(ring)
            left -= take
            if not left:
                break
    return weights


def share(counts):
    total = counts['foreground'] + counts['background'] + counts['sink']
    return np.where(total > 0, counts['foreground'] / np.where(total > 0, total, 1), 0.)


def token_parts(models, dataset, n, threshold):
    """masks[model][part]: bool [image, layer, patch] from block-input norms of a local forward."""
    masks = {kind: {part: [] for part in PARTS} for kind in models}
    with torch.inference_mode():
        for images, fg, valid, _ in DataLoader(Subset(dataset, range(n)), batch_size=32, num_workers=2):
            for kind, model in models.items():
                with capture_input_norms(model) as norms:
                    model(images)
                sink = torch.stack([sink_tokens(norms[layer], threshold) for layer in range(1, 13)], 1)
                fg_l, valid_l = fg[:, None].expand_as(sink), valid[:, None].expand_as(sink)
                parts = dict(foreground=fg_l & valid_l & ~sink, background=~fg_l & valid_l & ~sink,
                             sink=sink, unknown=~valid_l & ~sink)
                for part, value in parts.items():
                    masks[kind][part].append(value.numpy())
    return {kind: {part: np.concatenate(value) for part, value in parts.items()} for kind, parts in masks.items()}


def run(args, config, output, commit):
    """args: smoke. Reads the union CSVs named in config['center_control']['union_runs']."""
    torch.set_num_threads(4)
    models = {name: load_model(name, 'cpu') for name in ('deit', 'mita', 'dino')}
    dataset = PetSegmentation(data_root(), data_reference(models['deit']))
    assert len(dataset) == config['pet']['images']
    n = args.smoke or len(dataset)
    masks = token_parts(models, dataset, n, config['pet']['sink_threshold'])
    weights = centre_weights()
    frames, checks = [], dict(git_commit=commit, images=n, device='cpu', torch=torch.__version__)
    for kind in models:
        path = ROOT / 'experiments' / config['center_control']['union_runs'][kind] / 'union_composition.csv.gz'
        probe = pd.read_csv(path).query('model == @kind and index < @n').drop_duplicates(['index', 'spec', 'layer'])
        image, layer = probe['index'].to_numpy(), probe['layer'].to_numpy() - 1
        size = probe['union_size'].to_numpy()
        centre = {part: (weights[size] * masks[kind][part][image, layer]).sum(-1) for part in PARTS}
        frame = probe[['index', 'spec', 'layer', 'union_size']].assign(model=kind)
        frame['probe_fg_share'] = share({part: probe[part].to_numpy() * size for part in PARTS})
        frame['centre_fg_share'] = share(centre)
        for part in PARTS:
            frame[f'centre_{part}'] = centre[part] / np.maximum(size, 1)
        frames.append(frame)
        # Where the probe read every patch, its high-norm count must equal the local one.
        full = size == GRID * GRID
        local_sink = masks[kind]['sink'][image, layer].sum(-1)
        checks[f'{kind}_full_union_rows'] = int(full.sum())
        checks[f'{kind}_full_union_sink_mismatch_rows'] = int(
            (np.rint(probe['sink'].to_numpy()[full] * GRID * GRID) != local_sink[full]).sum())
    table = pd.concat(frames)
    table.to_csv(output / 'center_control.csv.gz', index=False)
    summarize(table, output)
    write_json(output / 'checks.json', checks)


def summarize(table, output):
    """Per-layer and layers 9-12 probe-minus-centre differences with paired image bootstrap intervals."""
    ids = np.sort(table['index'].unique())
    draws = resample_indices(len(ids))
    rows = []
    for (kind, spec, layer), frame in table.groupby(['model', 'spec', 'layer']):
        frame = frame.set_index('index').loc[ids]
        diff = frame['probe_fg_share'] - frame['centre_fg_share']
        rows.append(dict(model=kind, spec=spec, layer=layer, union_size=frame['union_size'].mean(),
                         probe=frame['probe_fg_share'].mean(), centre=frame['centre_fg_share'].mean(),
                         **dict(zip(['probe_minus_centre', 'low', 'high'], interval(diff, draws)))))
    pd.DataFrame(rows).to_csv(output / 'center_control_by_layer.csv', index=False)
    verdicts = []
    for (kind, spec), frame in table.query('layer >= 9').groupby(['model', 'spec']):
        deep = frame.groupby('index')[['probe_fg_share', 'centre_fg_share']].mean().loc[ids]
        mean, low, high = interval(deep['probe_fg_share'] - deep['centre_fg_share'], draws)
        primary = spec == 'm25_k25_mita' and kind in {'deit', 'dino'}
        verdict = ('supported' if low > 0 else 'not supported' if high < 0 else 'uncertain') if primary else ''
        verdicts.append(dict(model=kind, spec=spec, probe=deep['probe_fg_share'].mean(),
                             centre=deep['centre_fg_share'].mean(), probe_minus_centre=mean, low=low, high=high,
                             primary=primary, verdict=verdict))
    pd.DataFrame(verdicts).to_csv(output / 'center_control_layers_9_12.csv', index=False)
