"""The attention instrument (R0 of the model variants), test-time registers and the register-neuron search."""

import argparse

import pytest
import torch

from fastweight_vit.execution import Plan, execute
from fastweight_vit.instrument import instrument
from fastweight_vit.models.mita import checkpoint_settings
from fastweight_vit.registers import (NeuronStats, edit_neurons, first_high_layer, matched_neurons, random_neurons,
                                      ranked_neurons)

from fakes import small_mita, tiny


def test_mita_settings_come_from_the_checkpoint_arguments():
    assert checkpoint_settings(argparse.Namespace(model='deit_small_patch16_224')) == ('deit_small_patch16_224', 5, 25)
    assert checkpoint_settings(argparse.Namespace(model='deit_small_patch16_224', pool_size=4, kv_topk=3)) == \
        ('deit_small_patch16_224', 4, 9)
    with pytest.raises(ValueError):
        checkpoint_settings(argparse.Namespace(model='vit_base_patch16_224'))


@pytest.mark.parametrize('make', [tiny, small_mita])
def test_recorded_attention_reproduces_the_model_and_sums_to_the_query_count(make):
    model = make()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        plain = model(images)
        with instrument(model, values=True) as store:
            recorded = model(images)
    torch.testing.assert_close(recorded, plain, rtol=1e-5, atol=1e-5)  # MiTA: A V equals the native output
    heads = model.blocks[0].attn.num_heads
    for layer, record in store.items():
        torch.testing.assert_close(record['received'].sum(-1), torch.full((2, heads), 197.), rtol=1e-5, atol=1e-3)
        assert record['value_norm'].shape == (2, heads, 197)


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


