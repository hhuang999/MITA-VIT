"""Lens-derived readings of attention (v5; decisions B33-B40).

The fast-weight lens reads an attention map as the hidden-neuron activations of a network the slow weights
write for each image: keys are first-layer weights, values second-layer weights, every token one neuron.
Reading the map as "this image's, position-accurate, contribution-ranked" importance rests on three premises
(M1-M3); each correction below repairs one of them. Notation: layer l, head h, query i (all 197 tokens),
key j (the 196 patches), a = softmax attention.

| reading | per layer l | repairs |
| --- | --- | --- |
| R0 | s_j = sum_h sum_i a_hij (attention received; `attention_layer{l}` of S2) | - |
| C3 | c_j = sum_h sum_i a_hij * ||W_O^h v_hj|| (output-projected value length; bias excluded) | M3 activation is not contribution |
| C2 | r_p = sum_j s_j P_l(p | x_j) (position probe on the block-l attention input, after norm1) | M2 token j is not patch j |
| C1 | s_j - mean_j (the model's mean map of this reading on held-out images) | M1 image-independent part |
| R_all | C3 -> C2 -> C1: relocate c, then subtract the held-out mean of relocated c | all three |
| R_all_alt | C3 -> C1 -> C2: subtract the held-out mean of c, then relocate | order check (appendix A7) |

Controls (each matched to one correction) and references:
- blur (for C2): s relocated by a Gaussian kernel whose mean squared displacement equals C2's on held-out
  tokens at that layer (rows normalised over the grid, so the total is kept, as in C2);
- no_high (for C3): s with the layer's high-norm patches set to 0 (block input norm > 3x the image median);
- centre_sub (for C1): s minus a centre Gaussian with the same spatial standard deviation as the mean map
  C1 subtracts (computed for the evaluated layer set as a whole);
- mean_map: the model's mean map itself (image-independent).

A reading over several layers is the sum of its per-layer readings. All maps are [batch, 196] floats; every
evaluation metric used with them is rank-based per image, so a constant offset or a positive factor does
not change a result.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from .explain import centre
from .interventions import sink_tokens
from .probe.execute import split_qkv

READINGS = ('R0', 'C1', 'C2', 'C3', 'R_all', 'R_all_alt', 'blur', 'no_high', 'centre_sub', 'mean_map',
            'D2', 'D2_shuffle', 'C3D2')
MEAN_VARIANTS = ('R0', 'C3', 'C2', 'C2C3')  # held-out mean maps (C2 = relocated R0, C2C3 = relocated C3)
NEEDS_PROBE = {'C2', 'R_all', 'R_all_alt', 'D2', 'D2_shuffle', 'C3D2'}
NEEDS_MEANS = {'C1': 'R0', 'R_all': 'C2C3', 'R_all_alt': 'C3', 'centre_sub': 'R0', 'mean_map': 'R0'}


# ---------------------------------------------------------------------------------------------------------
# Position probe (C2) and the matched blur (its control)

@dataclass
class PositionProbe:
    """Linear 196-way position classifier on standardised tokens (the E3 probe, kept for reuse)."""
    mean: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor  # [positions, dim]
    bias: torch.Tensor    # [positions]

    def logits(self, tokens):
        x = (tokens.float() - self.mean) / self.std
        return x @ self.weight.T + self.bias

    def probs(self, tokens):
        """[..., patches, dim] -> P(position | token) [..., patches, positions]."""
        return self.logits(tokens).softmax(-1)

    def to(self, device):
        return PositionProbe(*(t.to(device) for t in (self.mean, self.std, self.weight, self.bias)))

    def arrays(self, prefix):
        return {f'{prefix}_{name}': getattr(self, name).cpu().numpy()
                for name in ('mean', 'std', 'weight', 'bias')}

    @classmethod
    def from_arrays(cls, arrays, prefix):
        return cls(*(torch.from_numpy(np.asarray(arrays[f'{prefix}_{name}']))
                     for name in ('mean', 'std', 'weight', 'bias')))


def fit_position_probe(train, epochs=30, batch=4096, lr=1e-2, seed=0):
    """train: [images, patches, dim] -> PositionProbe; the training of identity.train_position_probe (E3)."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    device = train.device
    patches, dim = train.shape[1], train.shape[2]
    x = train.reshape(-1, dim).float()
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    x = (x - mean) / std
    y = torch.arange(patches, device=device).repeat(train.shape[0])
    linear = torch.nn.Linear(dim, patches).to(device)
    torch.manual_seed(seed)
    linear.reset_parameters()
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    for _ in range(epochs):
        for chunk in torch.randperm(len(x), generator=generator).to(device).split(batch):
            optimizer.zero_grad()
            F.cross_entropy(linear(x[chunk]), y[chunk]).backward()
            optimizer.step()
    return PositionProbe(mean.detach(), std.detach(), linear.weight.detach().clone(), linear.bias.detach().clone())


def grid_positions(grid=14, device=None):
    rows, cols = torch.div(torch.arange(grid * grid, device=device), grid, rounding_mode='floor'), \
        torch.arange(grid * grid, device=device) % grid
    return torch.stack([rows, cols], -1).float()


def squared_distances(grid=14, device=None):
    """[patches, patches] squared grid distance between patch j and position p."""
    xy = grid_positions(grid, device)
    return ((xy[:, None] - xy[None]) ** 2).sum(-1)


def displacement(probs, grid=14):
    """Mean squared displacement sum_p P(p | x_j) ||p - j||^2, averaged over all leading dims and tokens j."""
    return float((probs * squared_distances(grid, probs.device)).sum(-1).mean())


def gaussian_relocation(sigma, grid=14, device=None):
    """[patches, positions] row-stochastic Gaussian kernel; sigma 0 is the identity."""
    if sigma <= 0:
        return torch.eye(grid * grid, device=device)
    weights = torch.exp(-squared_distances(grid, device) / (2 * sigma ** 2))
    return weights / weights.sum(-1, keepdim=True)


def match_sigma(target, grid=14, tolerance=1e-6):
    """Sigma whose truncated, renormalised Gaussian has mean squared displacement `target` (bisection)."""
    if target <= 0:
        return 0.
    low, high = 1e-3, 4. * grid
    if displacement(gaussian_relocation(high, grid), grid) < target:
        return high
    while high - low > tolerance:
        middle = (low + high) / 2
        if displacement(gaussian_relocation(middle, grid), grid) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def relocate(scores, probs):
    """C2: [batch, patches] scores and [batch, patches, positions] (or a shared [patches, positions]) -> [batch, positions]."""
    if probs.ndim == 2:
        return scores @ probs
    return torch.einsum('bj,bjp->bp', scores, probs)


def centre_matched(mean_map, grid=14):
    """C1's control: a centre Gaussian scaled to the spatial standard deviation of `mean_map` [positions]."""
    gauss = centre(1, grid, device=mean_map.device)[0]
    return gauss * (mean_map.std() / gauss.std())


def m1_sums(scores, mean_map):
    """Per image: (residual, total) sum of squares around each map's own spatial mean.

    M1 share of a layer = 1 - sum(residual) / sum(total) over images: the part of the map's spatial
    variation that the image-independent mean map explains.
    """
    centred = scores - scores.mean(-1, keepdim=True)
    residual = scores - mean_map
    residual = residual - residual.mean(-1, keepdim=True)
    return (residual ** 2).sum(-1), (centred ** 2).sum(-1)


def radial_share(mean_map, grid=14):
    """Share of a mean map's spatial variance explained by its ring (distance-to-centre) means."""
    from .interventions import centre_rings
    rings = centre_rings(grid)[1:].to(mean_map.device)
    fitted = torch.zeros_like(mean_map)
    for ring in rings.unique():
        members = rings == ring
        fitted[members] = mean_map[members].mean()
    total = ((mean_map - mean_map.mean()) ** 2).sum()
    return float(1 - ((mean_map - fitted) ** 2).sum() / total) if total > 0 else float('nan')


# ---------------------------------------------------------------------------------------------------------
# One forward pass: everything the readings need, per layer

def output_gram(attention):
    """Per head W_O^h^T W_O^h [heads, head_dim, head_dim] of the attention output projection."""
    weight = attention.proj.weight.reshape(-1, attention.num_heads, attention.head_dim)  # [dim, heads, head_dim]
    return torch.einsum('dhe,dhf->hef', weight, weight)


def received_and_length(p, v, gram):
    """p [B, H, N, N] attention, v [B, H, N, d] values -> (received [B, H, N], ||W_O^h v_hj|| [B, H, N])."""
    length = torch.einsum('bhne,hef,bhnf->bhn', v, gram, v).clamp_min(0).sqrt()
    return p.sum(2), length


@contextmanager
def capture(model, layers, prefix=None, per_head=False):
    """Run layers with an explicit softmax; yield {layer: dict(r0, c3, tokens, norms)} filled by the forward.

    r0 [B, patches]: attention received from all queries and heads; c3 [B, patches]: the same weighted by
    ||W_O^h v_hj||; tokens [B, patches, dim]: the block's normalised input (what produces keys and values);
    norms [B, patches]: the block input norm before norm1 (high-norm definition, B24).
    per_head=True also keeps, per head, 'received' [B, H, patches], 'length' ||W_O^h v_hj|| and 'value_norm'
    ||v_hj|| [B, H, patches] (the X2 mechanism checks, B46).
    """
    prefix = model.num_prefix_tokens if prefix is None else prefix
    store, saved, handles = {layer: {} for layer in layers}, {}, []

    def make_forward(layer, attention):
        gram = output_gram(attention)

        def forward(x, attn_mask=None, is_causal=False):
            if attn_mask is not None or is_causal:
                raise ValueError('Expected unmasked, noncausal self-attention')
            if not isinstance(attention.norm, torch.nn.Identity):
                raise ValueError('C3 assumes no normalisation between attention and the output projection')
            batch, tokens, _ = x.shape
            q, k, v = split_qkv(attention, x)  # [B, H, N, d]
            p = ((q * attention.scale) @ k.transpose(-2, -1)).softmax(-1)
            received, length = received_and_length(p, v, gram)
            store[layer]['r0'] = received.sum(1)[:, prefix:]
            store[layer]['c3'] = (received * length).sum(1)[:, prefix:]
            store[layer]['tokens'] = x[:, prefix:]
            if per_head:
                store[layer]['received'] = received[:, :, prefix:]
                store[layer]['length'] = length[:, :, prefix:]
                store[layer]['value_norm'] = v.norm(dim=-1)[:, :, prefix:]
            out = (p @ v).transpose(1, 2).reshape(batch, tokens, -1)
            return attention.proj_drop(attention.proj(attention.norm(out)))
        return forward

    def norm_hook(layer):
        def hook(module, inputs):
            store[layer]['norms'] = inputs[0][:, prefix:].float().norm(dim=-1)
        return hook

    try:
        for layer in layers:
            block = model.blocks[layer - 1]
            saved[layer] = block.attn.__dict__.get('forward')
            block.attn.forward = make_forward(layer, block.attn)
            handles.append(block.register_forward_pre_hook(norm_hook(layer)))
        yield store
    finally:
        for handle in handles:
            handle.remove()
        for layer, forward in saved.items():
            attention = model.blocks[layer - 1].attn
            if forward is None:
                del attention.forward
            else:
                attention.forward = forward


# ---------------------------------------------------------------------------------------------------------
# The lens: probes, blur widths and mean maps estimated on held-out images

@dataclass
class Lens:
    """probes {layer: PositionProbe}; sigmas {layer: float}; means {variant: {layer: [positions]}}."""
    probes: dict = field(default_factory=dict)
    sigmas: dict = field(default_factory=dict)
    means: dict = field(default_factory=dict)
    norm_ratio: float = 3.

    def to(self, device):
        return Lens({layer: probe.to(device) for layer, probe in self.probes.items()}, dict(self.sigmas),
                    {variant: {layer: value.to(device) for layer, value in maps.items()}
                     for variant, maps in self.means.items()}, self.norm_ratio)

    def mean(self, variant, layers):
        return sum(self.means[variant][layer] for layer in layers)


def save_probes(path, probes, sigmas, displacements):
    arrays = {}
    for layer, probe in probes.items():
        arrays.update(probe.arrays(f'layer{layer}'))
    layers = sorted(probes)
    np.savez_compressed(path, layers=np.asarray(layers), sigmas=np.asarray([sigmas[layer] for layer in layers]),
                        displacements=np.asarray([displacements[layer] for layer in layers]), **arrays)


def load_probes(path):
    arrays = np.load(path)
    layers = arrays['layers'].tolist()
    probes = {layer: PositionProbe.from_arrays(arrays, f'layer{layer}') for layer in layers}
    sigmas = dict(zip(layers, arrays['sigmas'].tolist()))
    return probes, sigmas


def save_means(path, means):
    np.savez_compressed(path, **{f'{variant}_layer{layer}': value.cpu().numpy()
                                 for variant, maps in means.items() for layer, value in maps.items()})


def load_means(path):
    arrays = np.load(path)
    means = {}
    for key in arrays.files:
        variant, layer = key.rsplit('_layer', 1)
        means.setdefault(variant, {})[int(layer)] = torch.from_numpy(arrays[key])
    return means


def layer_variants(store, layer, lens, wanted):
    """Token-level pieces of one layer for the wanted mean variants or readings."""
    piece = store[layer]
    out = {'R0': piece['r0'], 'C3': piece['c3']}
    if wanted & ({'C2', 'C2C3'} | NEEDS_PROBE):
        probs = lens.probes[layer].probs(piece['tokens'])
        out['P'] = probs
        out['C2'] = relocate(piece['r0'], probs)
        out['C2C3'] = relocate(piece['c3'], probs)
    return out


def accumulate_means(model, batches, lens, layers, device):
    """Mean map {variant: {layer: [positions]}} of every MEAN_VARIANTS reading over the images of `batches`."""
    sums = {variant: {layer: 0 for layer in layers} for variant in MEAN_VARIANTS}
    count = 0
    for images in batches:
        with torch.no_grad(), capture(model, layers) as store:
            model(images.to(device))
            for layer in layers:
                pieces = layer_variants(store, layer, lens, set(MEAN_VARIANTS))
                for variant in MEAN_VARIANTS:
                    sums[variant][layer] = sums[variant][layer] + pieces[variant].double().sum(0)
        count += len(images)
    return {variant: {layer: (value / count).float() for layer, value in maps.items()} for variant, maps in sums.items()}


def shuffled_rows(values, seed, start, layer):
    """Permute each image's row of per-patch values [B, patches] with a seeded generator (D2's stand-in, B50)."""
    generator = torch.Generator(device=values.device).manual_seed(((seed * 1_000_003 + start) * 101 + layer))
    order = torch.rand(values.shape, generator=generator, device=values.device).argsort(-1)
    return values.gather(-1, order)


def compute(store, entries, lens, grid=14, start=0):
    """{name: [B, positions]} for reading entries dict(name, reading, layers) from one captured forward.

    D2 (B50, exploratory): R0 times P_l(j | x_j), the probe's probability that token j is still at patch j (no
    relocation); D2_shuffle: the same with those probabilities permuted over the patches of the image and layer
    (entry['seed']; `start` = the batch's first row, for the seeded stream); C3D2: C3 times P_l(j | x_j).
    """
    out = {}
    for entry in entries:
        reading, layers = entry['reading'], entry['layers']
        if reading not in READINGS:
            raise ValueError(f'Unknown reading {reading}; choose from {READINGS}')
        total = 0
        for layer in layers:
            piece = layer_variants(store, layer, lens, {reading})
            r0, c3 = piece['R0'], piece['C3']
            if reading in ('R0', 'centre_sub', 'mean_map'):
                value = r0
            elif reading == 'C3':
                value = c3
            elif reading == 'C2':
                value = piece['C2']
            elif reading == 'C1':
                value = r0 - lens.means['R0'][layer]
            elif reading == 'R_all':
                value = piece['C2C3'] - lens.means['C2C3'][layer]
            elif reading == 'R_all_alt':
                value = relocate(c3 - lens.means['C3'][layer], piece['P'])
            elif reading == 'blur':
                value = relocate(r0, gaussian_relocation(lens.sigmas[layer], grid, r0.device))
            elif reading == 'no_high':
                value = r0 * ~sink_tokens(store[layer]['norms'], lens.norm_ratio)
            elif reading in ('D2', 'D2_shuffle', 'C3D2'):
                stay = piece['P'].diagonal(dim1=-2, dim2=-1)  # P_l(j | x_j), [B, patches]
                if reading == 'D2_shuffle':
                    stay = shuffled_rows(stay, entry['seed'], start, layer)
                value = (c3 if reading == 'C3D2' else r0) * stay
            total = total + value
        if reading == 'centre_sub':
            total = total - centre_matched(lens.mean('R0', layers), grid)
        elif reading == 'mean_map':
            total = lens.mean('R0', layers).expand_as(total).clone()
        out[entry['name']] = total
    return out


def needed_layers(entries):
    return sorted({layer for entry in entries for layer in entry['layers']})
