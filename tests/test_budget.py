"""Attention-budget study (decisions B27-B30): budget recording, MiTA pass-back, read edits, controls,
test-time registers and the T0-T7 experiments, on CPU with random weights and fake data."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from mita_probe import budget as B
from mita_probe.data.splits import split_indices
from mita_probe.experiments import attention_off, budget, common, model_check, register, sink_use
from mita_probe.interventions import centre_rings, sink_tokens
from mita_probe.io import load_config
from mita_probe.models.deit import load_deit
from mita_probe.models.mita import build_mita, checkpoint_settings
from mita_probe.probe import Plan, execute
from mita_probe.registers import (NeuronStats, edit_neurons, first_high_layer, matched_neurons, random_neurons,
                                  ranked_neurons)
from mita_probe.sinks import equal_mass, ring_random, top_ordinary

from test_new_experiments import FakeMasks

ROOT = Path(__file__).resolve().parents[1]


def tiny():
    torch.manual_seed(0)
    return load_deit(pretrained=False, size='tiny')


def tiny_mita():
    torch.manual_seed(0)
    skeleton = load_deit(pretrained=False, size='tiny')
    return build_mita(skeleton.state_dict(), argparse.Namespace(model='deit_tiny_patch16_224'))


def test_mita_settings_come_from_the_checkpoint_arguments():
    assert checkpoint_settings(argparse.Namespace(model='deit_small_patch16_224')) == ('deit_small_patch16_224', 5, 25)
    assert checkpoint_settings(argparse.Namespace(model='deit_tiny_patch16_224', pool_size=4, kv_topk=3)) == \
        ('deit_tiny_patch16_224', 4, 9)
    with pytest.raises(ValueError):
        checkpoint_settings(argparse.Namespace(model='vit_base_patch16_224'))


@pytest.mark.parametrize('make', [tiny, tiny_mita])
def test_recorded_budget_reproduces_the_model_and_sums_to_one(make):
    model = make()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        plain = model(images)
        with B.instrument(model, values=True, rank=True, selected=True) as store:
            recorded = model(images)
    torch.testing.assert_close(recorded, plain, rtol=1e-5, atol=1e-5)  # MiTA: A V equals the native output
    tokens = 197
    for layer, record in store.items():
        torch.testing.assert_close(record['received'].sum(-1), torch.full((2, model.blocks[0].attn.num_heads),
                                                                           float(tokens)), rtol=1e-5, atol=1e-3)
        torch.testing.assert_close(record['cls'].sum(-1), torch.ones(2, model.blocks[0].attn.num_heads))
        assert record['rank'].shape == (2, model.blocks[0].attn.num_heads) and record['selected'].shape == (2, tokens)


def test_mita_selection_is_its_own_top_k_union():
    model = tiny_mita()
    with torch.no_grad(), B.instrument(model, layers=[12], selected=True) as store:
        model(torch.randn(1, 3, 224, 224))
    # 25 regions x 25 tokens per head can cover at most 197 tokens, at least 25.
    assert 25 <= int(store[12]['selected'].sum()) <= 197


def test_categories_and_shares():
    fg = torch.zeros(2, 196, dtype=torch.bool)
    fg[:, :98] = True
    valid = torch.ones(2, 196, dtype=torch.bool)
    valid[:, 0] = False
    high = torch.zeros(2, 196, dtype=torch.bool)
    high[:, 150] = True
    codes = B.categories(fg, valid, high, prefix=2)
    assert codes[0, 0] == B.CLS and codes[0, 1] == B.REGISTER and codes[0, 2] == B.UNKNOWN
    assert codes[0, 2 + 150] == B.HIGH and codes[0, 2 + 5] == B.OBJECT and codes[0, 2 + 120] == B.BACKGROUND
    received = torch.rand(2, 3, 198)
    part = B.shares(received, codes)
    torch.testing.assert_close(part.sum(1), torch.ones(2))


def test_block_moves_the_budget_and_zero_keeps_it():
    model = tiny()
    images = torch.randn(2, 3, 224, 224)
    target = torch.zeros(2, 197, dtype=torch.bool)
    target[:, 5] = True
    with torch.no_grad():
        with B.instrument(model) as plain:
            base = model(images)
        with B.instrument(model, edits=B.Edits(block={12: torch.zeros_like(target)})):
            same = model(images)
        with B.instrument(model, edits=B.Edits(block={12: target})) as blocked:
            moved = model(images)
        with B.instrument(model, edits=B.Edits(zero={12: target})) as zeroed:
            emptied = model(images)
    torch.testing.assert_close(same, base)
    assert float(blocked[12]['received'][:, :, 5].abs().max()) == 0 and not torch.allclose(moved, base)
    torch.testing.assert_close(zeroed[12]['received'], plain[12]['received'])
    assert not torch.allclose(emptied, base)
    with pytest.raises(ValueError):
        with B.instrument(tiny_mita(), edits=B.Edits(zero={12: target})):
            pass


def test_mean_edit_replaces_values():
    model = tiny()
    images = torch.randn(1, 3, 224, 224)
    target = torch.zeros(1, 197, dtype=torch.bool)
    target[:, 7] = True
    heads, dim = model.blocks[0].attn.num_heads, model.blocks[0].attn.head_dim
    with torch.no_grad(), B.instrument(model, values=True, edits=B.Edits(mean={3: (target, torch.ones(heads, dim))})) \
            as store:
        model(images)
    torch.testing.assert_close(store[3]['value_norm'][0, :, 7], torch.full((heads,), dim ** .5))


def test_ring_enrichment():
    fg = torch.zeros(1, 196, dtype=torch.bool)
    fg[0, :50] = True
    valid = torch.ones(1, 196, dtype=torch.bool)
    assert abs(float(B.ring_enrichment(valid, fg, valid))) < 1e-12  # choosing everything = ring base rate
    assert float(B.ring_enrichment(fg, fg, valid)) > 0
    assert torch.isnan(B.ring_enrichment(torch.zeros_like(valid), fg, valid)).all()


def test_split_is_disjoint_and_fixed():
    pilot, formal = split_indices(1000, 'pilot', 100), split_indices(1000, 'formal', 100)
    assert len(pilot) == 100 and len(formal) == 900 and not set(pilot) & set(formal)
    assert sorted(pilot + formal) == list(range(1000)) and pilot == split_indices(1000, 'pilot', 100)
    assert split_indices(5, 'all') == list(range(5))


def test_same_count_controls():
    received = torch.rand(2, 196)
    high = torch.zeros(2, 196, dtype=torch.bool)
    high[0, [0, 50, 100]] = True
    high[1, 10] = True
    c1 = top_ordinary(received, high)
    assert c1.sum(1).tolist() == [3, 1] and not (c1 & high).any()
    assert received[0][c1[0]].min() >= received[0][~high[0] & ~c1[0]].max()
    c2, short = ring_random(high, torch.Generator().manual_seed(0))
    rings = centre_rings()[1:]
    for b in range(2):
        assert sorted(rings[c2[b]].tolist()) == sorted(rings[high[b]].tolist()) and short[b] == 0
    assert not (c2 & high).any()
    mass, feasible = equal_mass(received, high)
    assert feasible.all() and ((received * mass).sum(1) >= (received * high).sum(1)).all()
    fewer = mass.clone()
    for b in range(2):  # dropping the smallest chosen patch falls short
        chosen = mass[b].nonzero().flatten()
        fewer[b, chosen[received[b, chosen].argmin()]] = False
    assert ((received * fewer).sum(1) < (received * high).sum(1)).all()
    impossible = torch.zeros(1, 196)
    impossible[0, 0] = 10.
    _, ok = equal_mass(impossible + 1e-3, torch.nn.functional.one_hot(torch.tensor([0]), 196).bool())
    assert not ok.all()


def test_test_time_register_shift_and_zero():
    model = tiny()
    images = torch.randn(2, 3, 224, 224)
    units = {3: torch.tensor([1, 5])}
    with torch.no_grad():
        plain = model(images)
        with edit_neurons(model, register=True) as _:
            with_token = model(images)
        with edit_neurons(model, units, register=True, capture={3}) as after:
            shifted = model(images)
        with edit_neurons(model, units, mode='zero', capture={3}) as zeroed:
            model(images)
    assert with_token.shape == plain.shape and not torch.allclose(with_token, plain)
    act = after[3]
    assert act.shape[1] == 198
    assert float(act[:, 2:, [1, 5]].abs().max()) == 0
    assert float(zeroed[3][:, 1:, [1, 5]].abs().max()) == 0 and zeroed[3].shape[1] == 197
    assert not torch.allclose(shifted, with_token)
    with pytest.raises(ValueError):
        with edit_neurons(model, units):
            pass


def test_register_value_is_the_patch_maximum_at_the_edited_layer():
    model = tiny()
    images = torch.randn(1, 3, 224, 224)
    units = {1: torch.tensor([3])}
    with torch.no_grad():
        with edit_neurons(model, register=True, capture={1}) as before:
            model(images)
        with edit_neurons(model, units, register=True, capture={1}) as after:
            model(images)
    # Layer 1 is the first edit, so its patch activations are those of the register-only run.
    torch.testing.assert_close(after[1][:, 1, 3], before[1][:, 2:, 3].max(1).values)


def test_neuron_statistics_and_selection():
    stats = NeuronStats()
    act = torch.zeros(1, 1 + 196, 4)
    future = torch.zeros(1, 196, dtype=torch.bool)
    future[0, :4] = True
    act[0, 1:5, 2] = 10.
    act[0, 1:, 0] = torch.randn(196)
    stats.add({1: act}, future)
    table = stats.table()
    ranked = ranked_neurons(table)
    assert ranked[0][:2] == (1, 2)
    assert random_neurons(table, 2, 0, [(1, 2)]) == random_neurons(table, 2, 0, [(1, 2)])
    assert (1, 2) not in random_neurons(table, 3, 0, [(1, 2)])
    matched = matched_neurons(table, [(1, 2)])
    assert len(matched) == 1 and matched[0] != (1, 2)
    assert first_high_layer(torch.tensor([[False, True], [False, True]])) == 2
    assert first_high_layer(torch.zeros(3, 2, dtype=torch.bool)) is None


def test_attention_off_is_the_residual_and_ffn_path():
    model = tiny()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        with execute(model, {layer: Plan(drop=True) for layer in range(1, 13)}):
            dropped = model(images)
        x = model.norm_pre(model._pos_embed(model.patch_embed(images)))
        for block in model.blocks:
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        manual = model.forward_head(model.norm(x))
    torch.testing.assert_close(dropped, manual, rtol=1e-5, atol=1e-5)


class FakeImageNet(Dataset):
    def __init__(self, *args, n=4, **kwargs):
        self.n = n
        self.preprocessing = kwargs.get('preprocessing')

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return torch.randn(3, 224, 224, generator=generator), index % 3


@pytest.fixture
def patched(monkeypatch, tmp_path):
    common.DEVICE = 'cpu'
    model = tiny()
    for module in (budget, sink_use, register, model_check, attention_off):
        monkeypatch.setattr(module, 'load_model', lambda name, device: model)

    def build(name, model_, model_name, config, deit_reference=None):
        if name == 'imagenet':
            return FakeImageNet(), list(range(4))
        dataset = FakeMasks(model_, n=8)
        return dataset, (list(range(3)) if config.get('split') == 'pilot' else list(range(3, 8)))
    monkeypatch.setattr(budget.datasets, 'build', build)
    monkeypatch.setattr(model_check, 'ImageNetValidation', FakeImageNet)
    monkeypatch.setattr(register, 'ImageNetValidation', FakeImageNet)
    return tmp_path


def configured(tmp_path):
    config = load_config(ROOT / 'configs/attention_budget.yaml')
    config.update(batch_size=2, num_workers=0)
    config['budget']['norm_ratio'] = 1.2  # random weights: make some patches "high-norm"
    config['register'].update(counts=[1, 2], seeds=[0, 1], presence_share=0.0)
    config['sink_use'].update(seeds=[0, 1])
    return config


def args(**fields):
    return SimpleNamespace(**({'model': 'deit_t', 'smoke': 0, 'resume': False, 'stage': None} | fields))


def test_experiments_run_end_to_end(patched):
    tmp = patched
    config = configured(tmp)
    pilot, formal = tmp / 'pilot', tmp / 'formal'
    for out in (pilot, formal):
        out.mkdir()
    budget.run(args(stage='pilot'), config, pilot, 'test')
    budget.run(args(stage='formal'), config, formal, 'test')
    table = pd.read_csv(pilot / 'budget.csv')
    assert len(table) == 3 * 12 * 2
    assert ((table[[f'share_{c}' for c in B.CATEGORIES]].sum(axis=1) - 1).abs() < 1e-4).all()
    enrichment = pd.read_csv(formal / 'enrichment.csv')
    assert len(enrichment) == 5 * 13 * 2 and set(enrichment.scope) == {'deep'} | {str(l) for l in range(1, 13)}
    assert len(pd.read_csv(formal / 'selected.csv')) == 5 * 12 and (formal / 'value_means.pt').exists() is False
    assert len(list((formal / 'raw').glob('batch_*.npz'))) == 3
    means = torch.load(pilot / 'value_means.pt', weights_only=True)
    assert set(means) == set(range(1, 13))

    config['sink_use']['value_means'] = str(pilot / 'value_means.pt')
    use = tmp / 'use'
    use.mkdir()
    sink_use.run(args(), config, use, 'test')
    rows = pd.read_csv(use / 'use.csv')
    assert set(rows.condition) == {'high', 'c1', 'c2_s0', 'c2_s1', 'mass', 'zero', 'mean'}
    pivot = rows.pivot(index='index', columns='condition', values='removed')
    assert (pivot['high'] == pivot['c1']).all() and (pivot['high'] == pivot['c2_s0']).all()

    search = tmp / 'search'
    search.mkdir()
    register.run(args(stage='search'), config, search, 'test')
    gate = json.loads((search / 'gate.json').read_text())
    assert gate['first_high_layer'] == 1 and not gate['found']  # presence 0: layer 1, nothing before it

    config['register']['presence_share'] = 0.5
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(register, 'first_high_layer', lambda presence, share: 4)
        searched = tmp / 'searched'
        searched.mkdir()
        register.run(args(stage='search'), config, searched, 'test')
    gate_rows = pd.read_csv(searched / 'gate.csv')
    assert set(gate_rows.condition) == {'found', 'random'} and len(gate_rows) == 2 * 3
    assert len(pd.read_csv(searched / 'neurons.csv')) == 3 * 768

    forced = dict(gate, found=True, first_high_layer=4, neurons=[[2, 1], [3, 7]], matched=[[2, 2], [3, 8]])
    (search / 'forced.json').write_text(json.dumps(forced))
    config['register']['gate'] = str(search / 'forced.json')
    final = tmp / 'final'
    final.mkdir()
    register.run(args(stage='formal'), config, final, 'test')
    per_layer = pd.read_csv(final / 'register.csv')
    assert set(per_layer.condition) == {'R0', 'R1', 'R2', 'R3_s0', 'R3_s1', 'R4', 'R5'}
    shares = per_layer[[f'share_{c}' for c in B.CATEGORIES]].sum(axis=1)
    assert ((shares - 1).abs() < 1e-4).all()
    assert (per_layer[per_layer.condition == 'R0'].share_register == 0).all()
    assert (per_layer[per_layer.condition == 'R1'].share_register > 0).all()
    deep = pd.read_csv(final / 'deep.csv')
    assert (deep[deep.condition == 'R0'].agree == 1).all()
    accuracy = pd.read_csv(final / 'accuracy.csv')
    assert len(accuracy) == 4 * 7


def test_model_check_and_attention_off_run(patched):
    config = configured(patched)
    check, off = patched / 'check', patched / 'off'
    for out in (check, off):
        out.mkdir()
    model_check.run(args(), config, check, 'test')
    assert len(pd.read_csv(check / 'predictions.csv')) == 4
    attention_off.run(args(), config, off, 'test')
    rows = pd.read_csv(off / 'attention_off.csv')
    assert len(rows) == 4 * 13 and set(rows.layers.astype(str)) == {'all'} | {str(l) for l in range(1, 13)}
