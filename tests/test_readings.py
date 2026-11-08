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


# --- derivations ------------------------------------------------------------------------------------------

def test_c2_with_the_identity_probe_returns_r0_and_keeps_the_total():
    scores = torch.rand(3, 196)
    base = identity_probe(8)
    eye = OwnPosition(base.mean, base.std, base.weight, base.bias).probs(torch.randn(3, 196, 8))
    torch.testing.assert_close(readings.relocate(scores, eye), scores)
    probs = torch.rand(3, 196, 196).softmax(-1)
    torch.testing.assert_close(readings.relocate(scores, probs).sum(-1), scores.sum(-1))
    shared = readings.gaussian_relocation(2.0)
    torch.testing.assert_close(readings.relocate(scores, shared).sum(-1), scores.sum(-1))
    torch.testing.assert_close(readings.gaussian_relocation(0), torch.eye(196))


def test_c3_with_equal_value_lengths_is_r0_times_a_constant():
    p = torch.rand(2, 3, 5, 5).softmax(-1)
    v = torch.nn.functional.normalize(torch.randn(2, 3, 5, 4), dim=-1) * 2.5
    gram = torch.eye(4).expand(3, 4, 4)  # W_O^h orthonormal: ||W_O^h v|| = ||v|| = 2.5
    received, length = readings.received_and_length(p, v, gram)
    torch.testing.assert_close(length, torch.full_like(length, 2.5))
    torch.testing.assert_close((received * length).sum(1), 2.5 * received.sum(1))


def test_c3_length_is_the_projected_value_norm():
    model = tiny()
    attention = model.blocks[0].attn
    v = torch.randn(2, attention.num_heads, 7, attention.head_dim)
    _, length = readings.received_and_length(torch.rand(2, attention.num_heads, 7, 7), v,
                                             readings.output_gram(attention))
    weight = attention.proj.weight.reshape(-1, attention.num_heads, attention.head_dim)
    direct = torch.einsum('dhe,bhne->bhnd', weight, v).norm(dim=-1)
    torch.testing.assert_close(length, direct, rtol=1e-4, atol=1e-5)


def test_capture_r0_is_attention_mass_and_leaves_logits_unchanged():
    model = tiny()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        plain = model(images)
        with readings.capture(model, DEEP) as store:
            captured = model(images)
    torch.testing.assert_close(captured, plain, rtol=1e-5, atol=1e-5)
    for layer in DEEP:
        torch.testing.assert_close(store[layer]['r0'], references.attention_mass(model, images, [layer]),
                                   rtol=1e-5, atol=1e-5)
        assert store[layer]['tokens'].shape == (2, 196, model.embed_dim)


def test_c1_averages_to_zero_on_the_held_out_images():
    model = tiny()
    images = torch.randn(4, 3, 224, 224)
    lens = random_lens(model)
    lens.means = readings.accumulate_means(model, [images[:2], images[2:]], lens, list(range(1, 13)), 'cpu')
    entries = [dict(name='C1', reading='C1', layers=DEEP), dict(name='R_all', reading='R_all', layers=[12]),
               dict(name='C1_l3', reading='C1', layers=[3])]
    with torch.no_grad(), readings.capture(model, readings.needed_layers(entries)) as store:
        model(images)
        maps = readings.compute(store, entries, lens)
    for name in maps:
        assert maps[name].mean(0).abs().max() < 1e-3 * (1 + maps[name].abs().max())


def test_r_all_follows_the_fixed_order_and_the_other_order():
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
    expect = dict(
        R0=r0[11] + r0[12], C3=c3[11] + c3[12],
        C2=sum(torch.einsum('bj,bjp->bp', r0[l], probs[l]) for l in (11, 12)),
        C1=sum(r0[l] - m['R0'][l] for l in (11, 12)),
        R_all=sum(torch.einsum('bj,bjp->bp', c3[l], probs[l]) - m['C2C3'][l] for l in (11, 12)),
        R_all_alt=sum(torch.einsum('bj,bjp->bp', c3[l] - m['C3'][l], probs[l]) for l in (11, 12)),
        stand_in_blur=sum(r0[l] @ readings.gaussian_relocation(1.5) for l in (11, 12)),
        stand_in_no_high=sum(r0[l] * ~high[l] for l in (11, 12)))
    for name, value in expect.items():
        torch.testing.assert_close(maps[name], value, rtol=1e-4, atol=1e-4)
    mean = m['R0'][11] + m['R0'][12]
    control = expect['R0'] - maps['stand_in_centre']
    torch.testing.assert_close(control.std(-1), mean.std().expand(2), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(maps['mean_map'], mean.expand(2, -1))
    assert not torch.allclose(maps['R_all'], maps['R_all_alt'])


def test_blur_width_matches_a_displacement():
    for target in (0.3, 1.0, 4.0):
        sigma = readings.match_sigma(target)
        assert abs(readings.displacement(readings.gaussian_relocation(sigma)) - target) < 1e-3
    assert readings.match_sigma(0) == 0


def test_m1_share_limits():
    scores = torch.rand(5, 196)
    residual, total = readings.m1_sums(scores, scores)
    assert residual.max() < 1e-10 and (total > 0).all()
    residual, total = readings.m1_sums(scores, torch.full((196,), 7.))
    torch.testing.assert_close(residual, total)
    radial = torch.from_numpy(np.hypot(*np.divmod(np.arange(196), 14) - np.array([[6.5], [6.5]]))).float()
    assert abs(readings.radial_share(radial) - 1) < 1e-6


def test_probe_fit_save_and_load(tmp_path):
    torch.manual_seed(0)
    patches, dim = 196, 16
    codes = torch.randn(patches, dim)
    train = codes.expand(6, -1, -1) + .01 * torch.randn(6, patches, dim)
    probe = probes.fit_position_probe(train, epochs=60)
    assert (probe.probs(codes[None]).argmax(-1) == torch.arange(patches)).float().mean() > .95
    readings.save_probes(tmp_path / 'p.npz', {1: probe}, {1: 0.7}, {1: 0.4})
    loaded, sigmas = readings.load_probes(tmp_path / 'p.npz')
    torch.testing.assert_close(loaded[1].weight, probe.weight)
    assert sigmas == {1: 0.7}


def test_heldout_selection_skips_evaluation_images():
    rng = np.random.default_rng(0)
    from PIL import Image
    pictures = [Image.fromarray(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)) for _ in range(6)]
    names = [f'n{i}' for i in range(6)]
    labels = [0, 0, 1, 1, 2, 2]
    clickme = {0: np.stack([thumbnail(pictures[1].resize((256, 256)), 'squash')])}
    kept, info = select([0, 1, 2, 3, 4, 5], names, labels, {'n2'}, clickme, lambda row: pictures[row], 3)
    assert kept == [0, 3, 4]
    assert info['skipped_imagenet_s'] == 1 and [d['row'] for d in info['clickme_duplicates']] == [1]
    value, geometry = best_match({'squash': thumbnail(pictures[1], 'squash')}, clickme[0])
    assert value > .99 and geometry == 'squash'


def test_probe_error_statistics_limits():
    eye = torch.eye(196).expand(2, -1, -1)
    stats = error_stats(eye)
    for key, value in dict(exact=1, top5=1, distance=0, near3x3=1, far3=0).items():
        torch.testing.assert_close(stats[key], torch.full((2,), float(value)))
    assert torch.isnan(stats['far_share']).all()             # no error mass at all: undefined
    uniform = torch.full((1, 196, 196), 1 / 196)
    stats = error_stats(uniform)
    assert stats['far3'] > .8 and abs(float(stats['near3x3']) - (9 * 144 + 6 * 48 + 4 * 4) / 196 / 196) < 1e-6
    blur = readings.gaussian_relocation(1.0)[None]
    assert float(error_stats(blur)['exact']) == 1.0
    sharp = (torch.eye(196) * 30).softmax(-1)[None]          # P(j) close to 1: the share must stay in [0, 1]
    assert 0 <= float(error_stats(sharp)['far_share']) <= 1


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


class _FixedProbe:
    def __init__(self, probs):
        self.matrix = probs

    def probs(self, tokens):
        return self.matrix.expand(*tokens.shape[:-1], self.matrix.shape[-1])


def test_per_head_capture_sums_to_r0_and_c3():
    model = tiny()
    with torch.no_grad(), readings.capture(model, [11, 12], per_head=True) as store:
        model(torch.randn(2, 3, 224, 224))
    for layer in (11, 12):
        piece = store[layer]
        torch.testing.assert_close(piece['received'].sum(1), piece['r0'])
        torch.testing.assert_close((piece['received'] * piece['length']).sum(1), piece['c3'], rtol=1e-5, atol=1e-5)
        assert piece['value_norm'].shape == piece['length'].shape


def test_supp_maps_identities():
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
    torch.testing.assert_close(maps['C2MLP'], r0, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(maps['stand_in_blur_MLP'], r0, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(maps['C3C2MLP'], c3, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(maps['C1C3'], c3 - lens.mean('C3', supplementary.DEEP), rtol=1e-5, atol=1e-5)
    headmean = sum((received[l] * length[l].mean(-1, keepdim=True)).sum(1) for l in supplementary.DEEP)
    torch.testing.assert_close(maps['C3headmean'], headmean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(maps['C3shuffle_s0'].sum(-1) * 0 + maps['C3shuffle_s0'].shape[-1],
                               torch.full((2,), 196.))
    shuffled = supplementary.shuffled(length[12], 0, 0, 12)
    torch.testing.assert_close(shuffled.sort(-1).values, length[12].sort(-1).values)
    assert not torch.equal(shuffled, length[12])
    assert ((dose >= 0) & (dose <= 1)).all()
    for layer in supplementary.LAYERWISE:
        assert f'C3_l{layer}' in maps and f'stand_in_no_high_l{layer}' in maps


def test_confident_relocation_extremes():
    scores = torch.rand(2, 196)
    probs = torch.rand(2, 196, 196).softmax(-1)
    eye = torch.eye(196)
    for confident, expected in ((torch.ones(2, 196, 1), readings.relocate(scores, probs)),
                                (torch.zeros(2, 196, 1), scores)):
        torch.testing.assert_close(readings.relocate(scores, confident * probs + (1 - confident) * eye), expected)


def test_mlp_probe_learns_and_stops_early():
    torch.manual_seed(0)
    codes = torch.randn(196, 16)
    train = codes.expand(8, -1, -1) + .01 * torch.randn(8, 196, 16)
    probe, history = probes.fit_mlp_probe(train[:6], train[6:], hidden=64, epochs=10)
    assert len(history) == 10 and max(history) > .9
    assert (probe.probs(codes[None]).argmax(-1) == torch.arange(196)).float().mean() > .9


def test_globalisation_statistics():
    rings = ring_groups()
    assert np.bincount(rings).tolist() == [49, 49, 49, 49]
    assert np.bincount(rank_groups(np.arange(10.))).tolist() == [2, 2, 2, 2, 2]
    tokens = torch.randn(196, 8)
    keep = torch.ones(196, dtype=torch.bool)
    keep[:3] = False
    g = global_scores(tokens, keep)
    assert torch.isnan(g[:3]).all() and torch.isfinite(g[3:]).all()
    row = globalisation_rows(torch.eye(196), tokens, ~keep, rings)
    assert row['n_kept'] == 193 and np.isnan(row['rho_exact'])          # all exact: constant
    assert abs(row['move_q1']) < 1e-9


