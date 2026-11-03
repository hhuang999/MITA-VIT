"""Neurons that make high-norm tokens, and test-time registers (decision B30; after Jiang et al. 2025).

`edit_neurons` changes a forward pass in two optional ways:
- register=True inserts one all-zero token right after CLS (no positional embedding), so the prefix grows
  from 1 to 2 tokens;
- neurons {one-based layer: LongTensor of FFN hidden units} with mode
  'shift': the units' largest activation over the image patches is written to the register token and their
    activations on all patches are set to 0 (Jiang et al., Algorithm 2);
  'zero': their activations on all patches are set to 0 and nothing is moved.
CLS keeps its activations in both modes.

`NeuronStats` accumulates, per FFN unit of the layers before the first high-norm layer L*, the mean
activation on patches that will be high-norm at L* and on all patches, and the spread over all patches;
`scores` turns them into the search score (high mean - overall mean) / standard deviation.
"""

from contextlib import contextmanager

import torch


def first_high_layer(presence, share=0.5):
    """One-based first layer where at least `share` of the images have a high-norm patch, else None.

    presence: [images, layers] bool (any high-norm patch in that image at that layer).
    """
    rate = presence.float().mean(0)
    hits = (rate >= share).nonzero().flatten()
    return int(hits[0]) + 1 if len(hits) else None


@contextmanager
def edit_neurons(model, neurons=None, mode='shift', register=False, capture=None):
    """Run the model with the edits above; yields {layer: activations [B, N, hidden]} for `capture` layers."""
    if mode not in ('shift', 'zero'):
        raise ValueError(f'Unknown mode {mode}')
    if mode == 'shift' and neurons and not register:
        raise ValueError('Shifting needs a register token to move the activations to')
    neurons = neurons or {}
    prefix = model.num_prefix_tokens + int(register)
    captured, handles = {}, []

    def insert(module, inputs, output):
        token = output.new_zeros(output.shape[0], 1, output.shape[2])
        return torch.cat([output[:, :model.num_prefix_tokens], token, output[:, model.num_prefix_tokens:]], 1)

    def make_hook(layer):
        units = neurons.get(layer)

        def hook(module, inputs, output):
            if units is not None and len(units):
                output = output.clone()
                patches = output[:, prefix:, units]
                if mode == 'shift':
                    output[:, prefix - 1, units] = patches.max(1).values
                output[:, prefix:, units] = 0.
            if capture is not None and layer in capture:
                captured[layer] = output
            return output
        return hook

    try:
        if register:
            handles.append(model.norm_pre.register_forward_hook(insert))
        for layer, block in enumerate(model.blocks, 1):
            if layer in neurons or (capture is not None and layer in capture):
                handles.append(block.mlp.act.register_forward_hook(make_hook(layer)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class NeuronStats:
    """Running sums per (layer, unit) over patch positions; `future_high` marks the high-norm patches at L*."""

    def __init__(self):
        self.sums = {}

    def add(self, activations, future_high, prefix=1):
        for layer, value in activations.items():
            patches = value[:, prefix:].double()  # [B, P, units]
            high = future_high[..., None].double()
            entry = self.sums.setdefault(layer, dict(high=0., n_high=0., all=0., squares=0., n_all=0.))
            entry['high'] = entry['high'] + (patches * high).sum((0, 1))
            entry['n_high'] += float(future_high.sum())
            entry['all'] = entry['all'] + patches.sum((0, 1))
            entry['squares'] = entry['squares'] + (patches ** 2).sum((0, 1))
            entry['n_all'] += float(patches.shape[0] * patches.shape[1])

    def table(self):
        """{layer: dict(score, high_mean, mean, std)} tensors over units."""
        out = {}
        for layer, entry in sorted(self.sums.items()):
            mean = entry['all'] / entry['n_all']
            std = (entry['squares'] / entry['n_all'] - mean ** 2).clamp_min(0).sqrt()
            high_mean = entry['high'] / max(entry['n_high'], 1.)
            out[layer] = dict(score=(high_mean - mean) / std.clamp_min(1e-12), high_mean=high_mean, mean=mean,
                              std=std)
        return out


def ranked_neurons(table):
    """[(layer, unit, score)] sorted by score, highest first (ties: lower layer, then lower unit)."""
    rows = [(layer, unit, float(score)) for layer, entry in table.items()
            for unit, score in enumerate(entry['score'].tolist())]
    return sorted(rows, key=lambda row: (-row[2], row[0], row[1]))


def as_layers(pairs, device=None):
    """[(layer, unit), ...] -> {layer: LongTensor of units}."""
    out = {}
    for layer, unit in pairs:
        out.setdefault(layer, []).append(unit)
    return {layer: torch.tensor(sorted(units), dtype=torch.long, device=device) for layer, units in out.items()}


def random_neurons(table, count, seed, exclude=()):
    """`count` units drawn uniformly from all searched units, excluding `exclude` pairs."""
    pool = [(layer, unit) for layer, entry in table.items() for unit in range(len(entry['score']))
            if (layer, unit) not in set(exclude)]
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randperm(len(pool), generator=generator)[:count].tolist()
    return [pool[i] for i in picks]


def matched_neurons(table, chosen):
    """For each chosen unit, the other unit of the same layer whose overall mean activation is closest."""
    taken, out = set(chosen), []
    for layer, unit in chosen:
        mean = table[layer]['mean']
        distance = (mean - mean[unit]).abs()
        for other in distance.argsort().tolist():
            if (layer, other) not in taken:
                taken.add((layer, other))
                out.append((layer, other))
                break
    return out
