"""v5 lens readings (decisions B36-B39): the derivations the corrections must satisfy, and the diagnose and
correct experiments end to end, on CPU with random weights and fake data."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from mita_probe import explain, readings
from mita_probe.data.heldout import best_match, select, thumbnail
from mita_probe.experiments import common, correct, diagnose, sanity
from mita_probe.interventions import sink_tokens
from mita_probe.io import load_config
from mita_probe.models.deit import load_deit

from test_new_experiments import FakeMasks

ROOT = Path(__file__).resolve().parents[1]
DEEP = [9, 10, 11, 12]


def tiny():
    torch.manual_seed(0)
    return load_deit(pretrained=False, size='tiny')


def identity_probe(dim, patches=196):
    """A probe whose softmax is (numerically) one-hot on token j's own position, whatever the token."""
    return readings.PositionProbe(torch.zeros(dim), torch.ones(dim), torch.zeros(patches, dim), torch.zeros(patches))


class OwnPosition(readings.PositionProbe):
    """P(p | x_j) = 1 if p == j: the identity relocation."""

    def probs(self, tokens):
        eye = torch.eye(tokens.shape[-2], device=tokens.device)
        return eye.expand(*tokens.shape[:-1], tokens.shape[-2])


def random_lens(model, layers=range(1, 13), seed=0):
    generator = torch.Generator().manual_seed(seed)
    dim = model.embed_dim
    probes = {layer: readings.PositionProbe(torch.zeros(dim), torch.ones(dim),
                                            torch.randn(196, dim, generator=generator) * .1, torch.zeros(196))
              for layer in layers}
    means = {variant: {layer: torch.rand(196, generator=generator) for layer in layers}
             for variant in readings.MEAN_VARIANTS}
    return readings.Lens(probes, {layer: 1.5 for layer in layers}, means, 1.2)


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
        torch.testing.assert_close(store[layer]['r0'], explain.attention_mass(model, images, [layer]),
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
    names = ['R0', 'C3', 'C2', 'C1', 'R_all', 'R_all_alt', 'blur', 'no_high', 'centre_sub', 'mean_map']
    entries = [dict(name=name, reading=name, layers=[11, 12]) for name in names]
    with torch.no_grad(), readings.capture(model, [11, 12]) as store:
        model(images)
        maps = readings.compute(store, entries, lens)
        probs = {layer: lens.probes[layer].probs(store[layer]['tokens']) for layer in (11, 12)}
        r0 = {layer: store[layer]['r0'] for layer in (11, 12)}
        c3 = {layer: store[layer]['c3'] for layer in (11, 12)}
        high = {layer: sink_tokens(store[layer]['norms'], 1.2) for layer in (11, 12)}
    m = lens.means
    expect = dict(
        R0=r0[11] + r0[12], C3=c3[11] + c3[12],
        C2=sum(torch.einsum('bj,bjp->bp', r0[l], probs[l]) for l in (11, 12)),
        C1=sum(r0[l] - m['R0'][l] for l in (11, 12)),
        R_all=sum(torch.einsum('bj,bjp->bp', c3[l], probs[l]) - m['C2C3'][l] for l in (11, 12)),
        R_all_alt=sum(torch.einsum('bj,bjp->bp', c3[l] - m['C3'][l], probs[l]) for l in (11, 12)),
        blur=sum(r0[l] @ readings.gaussian_relocation(1.5) for l in (11, 12)),
        no_high=sum(r0[l] * ~high[l] for l in (11, 12)))
    for name, value in expect.items():
        torch.testing.assert_close(maps[name], value, rtol=1e-4, atol=1e-4)
    mean = m['R0'][11] + m['R0'][12]
    control = expect['R0'] - maps['centre_sub']
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
    probe = readings.fit_position_probe(train, epochs=60)
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


# --- experiments end to end ------------------------------------------------------------------------------

class FakeClickMe(Dataset):
    def __init__(self, *args, n=6, **kwargs):
        self.n = n

    def __len__(self):
        return self.n

    def record(self, index):
        from PIL import Image
        image = Image.fromarray(np.full((256, 256, 3), index * 20, dtype=np.uint8))
        return image, np.zeros((256, 256), np.uint8), index % 3

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return torch.randn(3, 224, 224, generator=generator), torch.rand(224, 224, generator=generator), index % 3, index


class FakeFiles:
    names = [f'ILSVRC2012_val_{i:08d}' for i in range(40)]

    def labels(self):
        return [i % 5 for i in range(40)]

    def image_at(self, row):
        from PIL import Image
        rng = np.random.default_rng(row)
        return Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))


@pytest.fixture
def patched(monkeypatch, tmp_path):
    common.DEVICE = 'cpu'
    model = tiny()
    for module in (diagnose, correct, sanity):
        monkeypatch.setattr(module, 'load_model', lambda name, device: model)
    monkeypatch.setattr(diagnose.datasets, 'imagenet_files', lambda: FakeFiles())
    monkeypatch.setattr(diagnose, 'ImageNetS', lambda *a, **k: FakeImageNetS())
    monkeypatch.setattr(diagnose, 'ClickMe', FakeClickMe)
    monkeypatch.setattr(correct, 'ClickMe', FakeClickMe)
    monkeypatch.setattr(diagnose, 'preprocessing_of', lambda model, name: dict(
        input_size=(3, 224, 224), interpolation='bicubic', crop_pct=0.875, crop_mode='center',
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))

    def build(name, model_, model_name, config, deit_reference=None):
        return FakeMasks(model_, n=6), list(range(6))
    monkeypatch.setattr(diagnose.datasets, 'build', build)
    monkeypatch.setattr(sanity.datasets, 'build', build)
    return tmp_path


class FakeImageNetS:
    def __len__(self):
        return 3

    def name(self, index):
        return FakeFiles.names[index]


def test_diagnose_and_correct_run_end_to_end(patched):
    tmp = patched
    config = load_config(ROOT / 'configs/v5.yaml')
    config.update(batch_size=4, num_workers=0)
    config['diagnose'].update(probe_images=6, mean_images=4, epochs=2, probe_run=str(tmp / 'probe'))
    config['correct'].update(lens_runs=dict(probe=str(tmp / 'probe'), m1=str(tmp / 'm1')), faithfulness_images=3,
                             budgets=[1, 98], gallery_images=2)
    for name in ('probe', 'm1', 'imagenet_s', 'clickme', 'order', 'sanity'):
        (tmp / name).mkdir()
    run = lambda **fields: SimpleNamespace(**({'model': 'deit_t', 'smoke': 0, 'resume': False} | fields))
    diagnose.run(run(stage='probe'), config, tmp / 'probe', 'test')
    held = json.loads((tmp / 'probe' / 'heldout.json').read_text())
    assert len(held['rows']) == 10 and not set(held['rows']) & {0, 1, 2}
    identity = pd.read_csv(tmp / 'probe' / 'identity.csv')
    assert list(identity.layer) == list(range(1, 13)) and (identity.sigma >= 0).all()
    assert len(pd.read_csv(tmp / 'probe' / 'identity_images.csv')) == 4 * 12
    diagnose.run(run(stage='m1'), config, tmp / 'm1', 'test')
    m1 = pd.read_csv(tmp / 'm1' / 'm1.csv')
    assert len(m1) == 6 * 12 * 4 and (m1.residual >= 0).all()
    assert (tmp / 'm1' / 'means_crop.npz').exists() and (tmp / 'm1' / 'means_clickme.npz').exists()

    correct.run(run(), config, tmp / 'imagenet_s', 'test')
    plaus = pd.read_csv(tmp / 'imagenet_s' / 'plausibility.csv')
    names = [entry['name'] for entry in config['correct']['methods']]
    assert set(plaus.method) == set(names) and len(plaus) == 6 * len(names)
    faith = pd.read_csv(tmp / 'imagenet_s' / 'faithfulness.csv')
    assert faith['index'].nunique() == 3 and set(faith.condition) == {'drop_top', 'drop_bottom'}
    assert not {'R0_early', 'C2_early'} & set(faith.method)
    gallery = np.load(next((tmp / 'imagenet_s').glob('gallery_*.npz')))
    assert 'map_R_all' in gallery.files

    config['correct']['dataset'] = 'clickme'
    correct.run(run(), config, tmp / 'clickme', 'test')
    aligned = pd.read_csv(tmp / 'clickme' / 'alignment.csv')
    assert set(aligned.method) == set(names)
    assert any('image' in np.load(path).files for path in (tmp / 'clickme').glob('gallery_*.npz'))

    order = load_config(ROOT / 'configs/correct_order_imagenet_s.yaml')['correct']
    config['correct'].update(dataset='imagenet_s', methods=order['methods'], gallery_images=0)
    correct.run(run(), config, tmp / 'order', 'test')
    assert set(pd.read_csv(tmp / 'order' / 'plausibility.csv').method) == {'R0', 'R_all', 'R_all_alt'}

    sanity_config = load_config(ROOT / 'configs/correct_sanity.yaml')
    sanity_config.update(batch_size=4, num_workers=0)
    sanity_config['correct'] = config['correct']
    sanity_config['sanity']['images'] = 4
    sanity.run(run(), sanity_config, tmp / 'sanity', 'test')
    table = pd.read_csv(tmp / 'sanity' / 'sanity.csv')
    assert {'R0', 'C1', 'C2', 'C3', 'R_all', 'centre'} == set(table.method)
    assert (table[table.method == 'centre'].spearman > .999).all()


def test_variant_runs_take_r0_only(patched):
    tmp = patched
    config = load_config(ROOT / 'configs/correct_register_imagenet_s.yaml')
    config.update(batch_size=4, num_workers=0)
    gate = dict(neurons=[[2, 5], [3, 7]])
    (tmp / 'gate.json').write_text(json.dumps(gate))
    config['correct']['gate'] = str(tmp / 'gate.json')
    out = tmp / 'register'
    out.mkdir()
    correct.run(SimpleNamespace(model='deit_t', smoke=0, resume=False), config, out, 'test')
    plaus = pd.read_csv(out / 'plausibility.csv')
    assert set(plaus.method) == {'R0', 'R0_l12'} and len(plaus) == 12
    config['correct']['faithfulness_images'] = 3
    again = tmp / 'register_srg'
    again.mkdir()
    correct.run(SimpleNamespace(model='deit_t', smoke=0, resume=False), config, again, 'test')
    faith = pd.read_csv(again / 'faithfulness.csv')
    assert faith['index'].nunique() == 3 and set(faith.method) == {'R0', 'R0_l12'}


# --- E-C2 (B45, exploratory) ------------------------------------------------------------------------------

def test_probe_error_statistics_limits():
    from mita_probe.experiments.explore_c2 import error_stats
    eye = torch.eye(196).expand(2, -1, -1)
    stats = error_stats(eye)
    for key, value in dict(exact=1, top5=1, distance=0, near3x3=1, far3=0).items():
        torch.testing.assert_close(stats[key], torch.full((2,), float(value)))
    assert torch.isnan(stats['far_share']).all()             # no error mass at all: undefined (B47)
    uniform = torch.full((1, 196, 196), 1 / 196)
    stats = error_stats(uniform)
    assert stats['far3'] > .8 and abs(float(stats['near3x3']) - (9 * 144 + 6 * 48 + 4 * 4) / 196 / 196) < 1e-6
    blur = readings.gaussian_relocation(1.0)[None]
    assert float(error_stats(blur)['exact']) == 1.0
    sharp = (torch.eye(196) * 30).softmax(-1)[None]          # P(j) close to 1: the share must stay in [0, 1]
    assert 0 <= float(error_stats(sharp)['far_share']) <= 1


def test_shift_statistics():
    from mita_probe.experiments.explore_c2 import N_BINS, closeness, distance_bins, shift_stats
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


def test_explore_c2_runs_end_to_end(patched):
    from mita_probe.experiments import explore_c2
    tmp = patched
    monkey = pytest.MonkeyPatch()
    monkey.setattr(explore_c2, 'load_model', lambda name, device: tiny())
    config = load_config(ROOT / 'configs/explore_c2.yaml')
    config.update(batch_size=4, num_workers=0)
    config['diagnose'].update(probe_images=6, mean_images=4, epochs=2, probe_run=str(tmp / 'probe'))
    config['correct'].update(lens_runs=dict(probe=str(tmp / 'probe'), m1=str(tmp / 'm1')))
    for name in ('probe', 'm1', 'explore'):
        (tmp / name).mkdir()
    run = lambda **fields: SimpleNamespace(**({'model': 'deit_t', 'smoke': 0, 'resume': False} | fields))
    diagnose.run(run(stage='probe'), config, tmp / 'probe', 'test')
    diagnose.run(run(stage='m1'), config, tmp / 'm1', 'test')
    explore_c2.run(run(), config, tmp / 'explore', 'test')
    monkey.undo()
    errors = pd.read_csv(tmp / 'explore' / 'probe_errors.csv')
    assert len(errors) == 4 * 7 * 2 and set(errors.source) == {'probe', 'blur'}
    assert (errors[errors.source == 'blur'].exact == 1).all()
    prediction = np.load(tmp / 'explore' / 'mean_prediction.npz')
    assert prediction['layer12'].shape == (4, 196)
    assert np.allclose(prediction['layer12'].sum(1), 1, atol=1e-5)
    shift = pd.read_csv(tmp / 'explore' / 'shift.csv')
    assert len(shift) == 6 * 2
    bins = shift[[f'bin{b}_moved' for b in range(10)]].sum(1)
    assert np.allclose(bins, 1, atol=1e-6)


# --- Phase-1 X1-X5 (B46, exploratory) ---------------------------------------------------------------------

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
    from mita_probe.experiments import supp
    model = tiny()
    lens = random_lens(model)
    eye = torch.eye(196)
    with torch.no_grad(), readings.capture(model, supp.LAYERWISE, per_head=True) as store:
        model(torch.randn(2, 3, 224, 224))
        mlp = {layer: _FixedProbe(eye) for layer in supp.DEEP}
        maps, dose = supp.supp_maps(store, lens, mlp, {layer: 0. for layer in supp.DEEP}, start=0)
        r0 = sum(store[layer]['r0'] for layer in supp.DEEP)
        c3 = sum(store[layer]['c3'] for layer in supp.DEEP)
        received = {layer: store[layer]['received'] for layer in supp.DEEP}
        length = {layer: store[layer]['length'] for layer in supp.DEEP}
    torch.testing.assert_close(maps['C2MLP'], r0, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(maps['blurMLP'], r0, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(maps['C3C2MLP'], c3, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(maps['C1C3'], c3 - lens.mean('C3', supp.DEEP), rtol=1e-5, atol=1e-5)
    headmean = sum((received[l] * length[l].mean(-1, keepdim=True)).sum(1) for l in supp.DEEP)
    torch.testing.assert_close(maps['C3headmean'], headmean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(maps['C3shuffle_s0'].sum(-1) * 0 + maps['C3shuffle_s0'].shape[-1],
                               torch.full((2,), 196.))
    shuffled = supp.shuffled(length[12], 0, 0, 12)
    torch.testing.assert_close(shuffled.sort(-1).values, length[12].sort(-1).values)
    assert not torch.equal(shuffled, length[12])
    assert ((dose >= 0) & (dose <= 1)).all()
    for layer in supp.LAYERWISE:
        assert f'C3_l{layer}' in maps and f'no_high_l{layer}' in maps


def test_confident_relocation_extremes():
    scores = torch.rand(2, 196)
    probs = torch.rand(2, 196, 196).softmax(-1)
    eye = torch.eye(196)
    for confident, expected in ((torch.ones(2, 196, 1), readings.relocate(scores, probs)),
                                (torch.zeros(2, 196, 1), scores)):
        torch.testing.assert_close(readings.relocate(scores, confident * probs + (1 - confident) * eye), expected)


def test_mlp_probe_learns_and_stops_early():
    from mita_probe.experiments.supp import fit_mlp_probe
    torch.manual_seed(0)
    codes = torch.randn(196, 16)
    train = codes.expand(8, -1, -1) + .01 * torch.randn(8, 196, 16)
    probe, history = fit_mlp_probe(train[:6], train[6:], hidden=64, epochs=10)
    assert len(history) == 10 and max(history) > .9
    assert (probe.probs(codes[None]).argmax(-1) == torch.arange(196)).float().mean() > .9


def test_globalisation_statistics():
    from mita_probe.experiments.supp import global_scores, globalisation_rows, rank_groups, ring_groups
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


def test_supp_runs_end_to_end(patched):
    from mita_probe.experiments import supp
    tmp = patched
    monkey = pytest.MonkeyPatch()
    monkey.setattr(supp, 'load_model', lambda name, device: tiny())
    monkey.setattr(supp, 'ClickMe', FakeClickMe)
    config = load_config(ROOT / 'configs/supp.yaml')
    config.update(batch_size=4, num_workers=0)
    config['diagnose'].update(probe_images=6, mean_images=4, epochs=2, probe_run=str(tmp / 'probe'))
    config['correct'].update(lens_runs=dict(probe=str(tmp / 'probe'), m1=str(tmp / 'm1')), faithfulness_images=3,
                             budgets=[1, 98])
    config['supp'].update(heldout_run=str(tmp / 'held'), mlp_hidden=32)
    for name in ('probe', 'm1', 'held', 'eval_s', 'eval_c'):
        (tmp / name).mkdir()
    run = lambda **fields: SimpleNamespace(**({'model': 'deit_t', 'smoke': 0, 'resume': False} | fields))
    diagnose.run(run(stage='probe'), config, tmp / 'probe', 'test')
    diagnose.run(run(stage='m1'), config, tmp / 'm1', 'test')
    supp.run(run(stage='heldout'), config, tmp / 'held', 'test')
    mlp = pd.read_csv(tmp / 'held' / 'mlp_probe.csv')
    assert set(mlp.probe) == {'linear', 'mlp'} and set(mlp.layer) == set(range(1, 13))
    glob = pd.read_csv(tmp / 'held' / 'global.csv')
    assert set(glob.layer) == set(range(6, 13)) and len(glob) == 4 * 7
    supp.run(run(stage='eval'), config, tmp / 'eval_s', 'test')
    plaus = pd.read_csv(tmp / 'eval_s' / 'plausibility.csv')
    assert {'C1C3', 'C2C3', 'C1C2', 'C3v', 'C3headmean', 'C3shuffle_s4', 'C2conf', 'C2MLP', 'C3_l6',
            'no_high_l12', 'C3_centre'} <= set(plaus.method)
    faith = pd.read_csv(tmp / 'eval_s' / 'faithfulness.csv')
    assert set(faith.method) == set(config['supp']['srg_maps']) and faith['index'].nunique() == 3
    assert len(pd.read_csv(tmp / 'eval_s' / 'dose.csv')) == 6
    assert len(pd.read_csv(tmp / 'eval_s' / 'head_weights.csv')) == 4 * model_heads(tiny())
    config['supp']['dataset'] = 'clickme'
    supp.run(run(stage='eval'), config, tmp / 'eval_c', 'test')
    monkey.undo()
    assert 'C2MLP' in set(pd.read_csv(tmp / 'eval_c' / 'alignment.csv').method)


def model_heads(model):
    return model.blocks[0].attn.num_heads


# --- D2 (B50, exploratory) --------------------------------------------------------------------------------

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
