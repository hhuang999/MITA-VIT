"""Derivations the readings must satisfy (R0, C1-C3, combinations, stand-ins), the position probes, and the
statistics of the exploratory analyses; CPU, random weights."""


import numpy as np


import torch


from fastweight_vit import probes, readings, references


from fastweight_vit.data.heldout import best_match, select, thumbnail


from fastweight_vit.experiments import supplementary


from fastweight_vit.experiments.c2_analysis import N_BINS, closeness, distance_bins, error_stats, shift_stats


from fastweight_vit.experiments.supplementary import global_scores, globalisation_rows, rank_groups, ring_groups


from fastweight_vit.tokens import high_norm_tokens


from fakes import tiny


DEEP = [9, 10, 11, 12]


def identity_probe(dim, patches=196):
    """A probe whose softmax is (numerically) one-hot on token j's own position, whatever the token."""
    return probes.PositionProbe(torch.zeros(dim), torch.ones(dim), torch.zeros(patches, dim), torch.zeros(patches))


class OwnPosition(probes.PositionProbe):
    """P(p | x_j) = 1 if p == j: the identity relocation."""

    def probs(self, tokens):
        eye = torch.eye(tokens.shape[-2], device=tokens.device)
        return eye.expand(*tokens.shape[:-1], tokens.shape[-2])


def random_lens(model, layers=range(1, 13), seed=0):
    generator = torch.Generator().manual_seed(seed)
    dim = model.embed_dim
    fitted = {layer: probes.PositionProbe(torch.zeros(dim), torch.ones(dim),
                                          torch.randn(196, dim, generator=generator) * .1, torch.zeros(196))
              for layer in layers}
    means = {variant: {layer: torch.rand(196, generator=generator) for layer in layers}
             for variant in readings.MEAN_VARIANTS}
    return readings.Lens(fitted, {layer: 1.5 for layer in layers}, means, 1.2)


class _FixedProbe:
    def __init__(self, probs):
        self.matrix = probs

    def probs(self, tokens):
        return self.matrix.expand(*tokens.shape[:-1], self.matrix.shape[-1])


def test_confident_relocation_extremes():
    scores = torch.rand(2, 196)
    probs = torch.rand(2, 196, 196).softmax(-1)
    eye = torch.eye(196)
    for confident, expected in ((torch.ones(2, 196, 1), readings.relocate(scores, probs)),
                                (torch.zeros(2, 196, 1), scores)):
        torch.testing.assert_close(readings.relocate(scores, confident * probs + (1 - confident) * eye), expected)


def test_d2_discounts_without_moving():
    model = tiny()
    lens = random_lens(model)
    entries = [dict(name=n, reading=n, layers=[11, 12]) for n in ('R0', 'C3', 'D2', 'C3D2')] + \
        [dict(name=f'D2_shuffle_s{k}', reading='D2_shuffle', seed=k, layers=[11, 12]) for k in (0, 1)]
    with torch.no_grad(), readings.capture(model, [11, 12]) as store:
        model(torch.randn(2, 3, 224, 224))
        maps = readings.compute(store, entries, lens, start=64)
        again = readings.compute(store, entries, lens, start=64)
        stay = {l: lens.probes[l].probs(store[l]['tokens']).diagonal(dim1=-2, dim2=-1) for l in (11, 12)}
        r0 = {l: store[l]['r0'] for l in (11, 12)}
        c3 = {l: store[l]['c3'] for l in (11, 12)}
    torch.testing.assert_close(maps['D2'], sum(r0[l] * stay[l] for l in (11, 12)))
    torch.testing.assert_close(maps['C3D2'], sum(c3[l] * stay[l] for l in (11, 12)))
    torch.testing.assert_close(maps['D2_shuffle_s0'], again['D2_shuffle_s0'])        # seeded
    assert not torch.allclose(maps['D2_shuffle_s0'], maps['D2_shuffle_s1'])
    permuted = readings.shuffled_rows(stay[12], 0, 64, 12)
    torch.testing.assert_close(permuted.sort(-1).values, stay[12].sort(-1).values)
    own = OwnPosition(*(getattr(lens.probes[12], f) for f in ('mean', 'std', 'weight', 'bias')))
    lens.probes = {11: own, 12: own}
    with torch.no_grad(), readings.capture(model, [11, 12]) as store:
        model(torch.randn(2, 3, 224, 224))
        maps = readings.compute(store, entries[:3], lens)
    torch.testing.assert_close(maps['D2'], maps['R0'])                              # P(j | x_j) = 1: no change


def test_shift_statistics():
    assert int(distance_bins().max()) == N_BINS - 1 and int(distance_bins().min()) == 0
    r0 = torch.rand(3, 196, dtype=torch.float64)
    foreground = torch.zeros(3, 196, dtype=torch.bool)
    foreground[:, :98] = True
    valid = torch.ones(3, 196, dtype=torch.bool)
    toward_centre = r0 + 0.1 * (closeness().double() - closeness().double().mean())
    stats = shift_stats(r0, toward_centre, closeness().double(), foreground, valid)
    torch.testing.assert_close(stats['pearson_centre'], torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(stats['bins_r0'].sum(-1), torch.ones(3, dtype=torch.float64))
    background = r0 * (~foreground)
    stats = shift_stats(r0, background, closeness().double(), foreground, valid)
    assert (stats['object_moved'] == 0).all() and (stats['object_r0'] > 0).all()


def test_other_correction_order():
    model = tiny()
    images = torch.randn(2, 3, 224, 224)
    lens = random_lens(model)
    names = ['R0', 'C3', 'C2', 'C1', 'R_all', 'R_all_alt', 'stand_in_blur', 'stand_in_no_high', 'stand_in_centre',
             'mean_map']
    entries = [dict(name=name, reading=name, layers=[11, 12]) for name in names]
    with torch.no_grad(), readings.capture(model, [11, 12]) as store:
        model(images)
        maps = readings.compute(store, entries, lens)
        probs = {layer: lens.probes[layer].probs(store[layer]['tokens']) for layer in (11, 12)}
        r0 = {layer: store[layer]['r0'] for layer in (11, 12)}
        c3 = {layer: store[layer]['c3'] for layer in (11, 12)}
        high = {layer: high_norm_tokens(store[layer]['norms'], 1.2) for layer in (11, 12)}
    m = lens.means
    expected = sum(torch.einsum('bj,bjp->bp', c3[l] - m['C3'][l], probs[l]) for l in (11, 12))
    torch.testing.assert_close(maps['R_all_alt'], expected, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(maps['R_all'], maps['R_all_alt'])


def test_additional_supplementary_maps():
    model = tiny()
    lens = random_lens(model)
    eye = torch.eye(196)
    with torch.no_grad(), readings.capture(model, supplementary.LAYERWISE, per_head=True) as store:
        model(torch.randn(2, 3, 224, 224))
        mlp = {layer: _FixedProbe(eye) for layer in supplementary.DEEP}
        maps, dose = supplementary.supp_maps(store, lens, mlp, {layer: 0. for layer in supplementary.DEEP}, start=0)
        r0 = sum(store[layer]['r0'] for layer in supplementary.DEEP)
        c3 = sum(store[layer]['c3'] for layer in supplementary.DEEP)
        received = {layer: store[layer]['received'] for layer in supplementary.DEEP}
        length = {layer: store[layer]['length'] for layer in supplementary.DEEP}
    torch.testing.assert_close(maps['C3C2MLP'], c3, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(maps['C1C3'], c3 - lens.mean('C3', supplementary.DEEP), rtol=1e-5, atol=1e-5)
    headmean = sum((received[l] * length[l].mean(-1, keepdim=True)).sum(1) for l in supplementary.DEEP)
    torch.testing.assert_close(maps['C3headmean'], headmean, rtol=1e-5, atol=1e-5)
    for layer in supplementary.LAYERWISE:
        assert f'C3_l{layer}' in maps and f'stand_in_no_high_l{layer}' in maps
