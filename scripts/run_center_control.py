"""Centre-prior control for the probe's union reading sets on Pet (decision A14), CPU only.

For every model, image, layer and composition spec, take the probe's union size U and read the U patches
closest to the image centre instead (ties broken uniformly at random, in expectation). Both are scored the
same way: foreground / (foreground + background + high-norm), boundary-only patches excluded.
"""

import argparse
import hashlib
import json

from mita_probe.runtime import configure_runtime, data_root, project_path
ROOT = configure_runtime()

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mita_probe.anatomy_analysis import interval, resample_indices
from mita_probe.dino_model import HEAD, HEAD_SHA256, load_dino_model
from mita_probe.io import load_config, source_commit, write_json
from mita_probe.mita_model import CHECKPOINT_SHA256, load_mita_model
from mita_probe.models import load_model
from mita_probe.pet import PetSegmentation
from mita_probe.rq3_extension import capture_input_norms, sink_tokens

GRID = 14
UNION = {'deit': '2026-10-28_v2_pet_union', 'mita': '2026-10-28_v2_pet_union', 'dino': '2026-10-28_v2_dino_pet'}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='experiments/2026-10-28_center_control')
    parser.add_argument('--smoke', type=int, default=0)
    args = parser.parse_args()
    commit = source_commit(ROOT)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / 'configs/anatomy_v2.yaml')
    torch.set_num_threads(4)
    deit = load_model('cpu')
    mita_path = project_path(config['mita']['checkpoint'])
    assert hashlib.sha256(mita_path.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    head = project_path(HEAD)
    assert hashlib.sha256(head.read_bytes()).hexdigest() == HEAD_SHA256
    models = dict(deit=deit, mita=load_mita_model(mita_path, 'cpu'), dino=load_dino_model(head, 'cpu'))
    dataset = PetSegmentation(data_root(), deit)
    assert len(dataset) == config['pet']['images']
    n = args.smoke or len(dataset)
    loader = DataLoader(torch.utils.data.Subset(dataset, range(n)), batch_size=32, num_workers=2)

    # masks[model][part]: bool [image, layer, patch]
    masks = {kind: {part: [] for part in PARTS} for kind in models}
    with torch.inference_mode():
        for images, fg, valid, _ in loader:
            for kind, model in models.items():
                with capture_input_norms(model) as norms:
                    model(images)
                sink = torch.stack([sink_tokens(norms[layer], config['pet']['sink_threshold'])
                                    for layer in range(1, 13)], 1)
                fg_l, valid_l = fg[:, None].expand_as(sink), valid[:, None].expand_as(sink)
                parts = dict(foreground=fg_l & valid_l & ~sink, background=~fg_l & valid_l & ~sink,
                             sink=sink, unknown=~valid_l & ~sink)
                for part, value in parts.items():
                    masks[kind][part].append(value.numpy())
    masks = {kind: {part: np.concatenate(value) for part, value in parts.items()} for kind, parts in masks.items()}

    weights = centre_weights()
    frames, checks = [], dict(git_commit=commit, images=n, device='cpu', torch=torch.__version__)
    for kind in models:
        path = ROOT / 'experiments' / UNION[kind] / 'union_composition.csv.gz'
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
    write_json(output / 'checks.json', checks)
    print(json.dumps(checks, indent=2))
    print(pd.DataFrame(verdicts).round(4).to_string())


if __name__ == '__main__':
    main()
