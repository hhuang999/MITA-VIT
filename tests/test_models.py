import pytest
import torch

from mita_probe.models import execute_attention, load_model, read_attention
from mita_probe.probe import execution_mask


@pytest.fixture
def model():
    return load_model(pretrained=False)


@pytest.mark.parametrize("fused", [False, True])
def test_read_only_and_all_keys_match_original_and_restore_hooks(model, fused):
    for block in model.blocks:
        block.attn.fused_attn = fused
    images = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        original = model(images)
        with read_attention(model, k_top=197) as patterns:
            observed = model(images)
        assert set(patterns) == set(range(1, 13))
        assert all(not block.attn.qkv._forward_hooks for block in model.blocks)
        torch.testing.assert_close(observed, original, atol=0, rtol=0)
        with execute_attention(model, patterns):
            executed = model(images)
        torch.testing.assert_close(executed, original, atol=1e-5, rtol=0)
        if not fused:
            torch.testing.assert_close(executed, original, atol=0, rtol=0)
        assert all(not block.attn._forward_pre_hooks for block in model.blocks)
        torch.testing.assert_close(model(images), original, atol=0, rtol=0)


def test_layer_scope_frozen_sets_and_manual_attention_reference(model):
    attention = model.blocks[2].attn
    captured = []
    handle = attention.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    with torch.inference_mode():
        with read_attention(model, k_top=8, layers=(3,)) as patterns:
            model(torch.randn(1, 3, 224, 224))
        handle.remove()
        assert set(patterns) == {3}
        pattern = patterns[3]
        saved = pattern.readsets.clone()
        x = captured[0]
        qkv = attention.qkv(x).reshape(1, 197, 3, 6, 64).permute(2, 0, 3, 1, 4)
        q, keys, values = qkv.unbind(0)
        mask = execution_mask(pattern)
        weights = ((q * attention.scale) @ keys.transpose(-1, -2)).masked_fill(~mask, -torch.inf).softmax(-1)
        expected = (weights @ values).transpose(1, 2).reshape(1, 197, 384)
        expected = attention.proj(expected)
        with execute_attention(model, patterns):
            assert len(attention._forward_pre_hooks) == 1
            assert all(not b.attn._forward_pre_hooks for i, b in enumerate(model.blocks) if i != 2)
            actual = attention(x)
            assert torch.isfinite(model(torch.randn(1, 3, 224, 224))).all()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(pattern.readsets, saved, atol=0, rtol=0)


def test_cls_excluded_from_pooling_but_routed_and_selectable(model):
    attention = model.blocks[0].attn
    fake_qkv = torch.zeros(1, 197, 3, 6, 64)
    fake_qkv[:, 0, 0] = 100  # CLS query must not move patch landmarks.
    fake_qkv[:, 1:, 0] = torch.rand(1, 196, 6, 64)
    fake_qkv[:, 0, 1] = 10  # CLS is the highest-scoring key.
    from mita_probe.grouping import grid_landmarks
    from mita_probe.probe import read_patterns

    q, keys, _ = fake_qkv.permute(2, 0, 3, 1, 4).unbind(0)
    expected = read_patterns(q, keys, grid_landmarks(q[:, :, 1:], (14, 14)), 1)
    replace = attention.qkv.register_forward_hook(lambda module, args, output: fake_qkv.reshape(1, 197, -1))
    with torch.inference_mode(), read_attention(model, k_top=1, layers=(1,)) as patterns:
        model(torch.zeros(1, 3, 224, 224))
    replace.remove()
    assert patterns[1].readsets.eq(0).all()
    torch.testing.assert_close(patterns[1].routes, expected.routes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("grouping,readset", [("grid", "ours"), ("kmeans", "ours"), ("grid", "random")])
def test_cuda_sparse_controls_and_all_keys(model, grouping, readset):
    model.to("cuda")
    images = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.inference_mode():
        original = model(images)
        for k_top in [1, 16, 197]:
            with read_attention(
                model, k_top=k_top, grouping=grouping, readset=readset,
                generator=torch.Generator(device="cuda").manual_seed(0),
            ) as patterns:
                observed = model(images)
            torch.testing.assert_close(observed, original, atol=0, rtol=0)
            with execute_attention(model, patterns):
                executed = model(images)
            assert torch.isfinite(executed).all()
            if k_top == 197:
                torch.testing.assert_close(executed, original, atol=1e-5, rtol=0)


def test_hooks_are_removed_when_forward_raises(model):
    with pytest.raises(RuntimeError, match="test failure"):
        with read_attention(model, k_top=4):
            raise RuntimeError("test failure")
    assert all(not block.attn.qkv._forward_hooks for block in model.blocks)
    with torch.inference_mode(), read_attention(model, k_top=4) as patterns:
        model(torch.zeros(1, 3, 224, 224))
    with pytest.raises(RuntimeError, match="test failure"):
        with execute_attention(model, patterns):
            raise RuntimeError("test failure")
    assert all(not block.attn._forward_pre_hooks for block in model.blocks)


def test_group_count_diagnostics_preserve_summaries_and_count_all_queries(model):
    images = torch.randn(1, 3, 224, 224)
    counts = {}
    with torch.inference_mode():
        with read_attention(model, k_top=4, grouping="kmeans", layers=(1,)) as baseline:
            original = model(images)
        with read_attention(model, k_top=4, grouping="kmeans", layers=(1,),
                            grouping_counts=counts) as diagnostic:
            observed = model(images)
    torch.testing.assert_close(original, observed, atol=0, rtol=0)
    torch.testing.assert_close(baseline[1].readsets, diagnostic[1].readsets)
    torch.testing.assert_close(baseline[1].routes, diagnostic[1].routes)
    assert counts[1]["euclidean_patch_clusters"].sum(-1).eq(196).all()
    assert counts[1]["dot_product_routes_including_cls"].sum(-1).eq(197).all()


@pytest.mark.parametrize("grouping,readset", [("kmeans", "ours"), ("grid", "random")])
def test_control_paths_are_read_only_then_execute(model, grouping, readset):
    images = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        original = model(images)
        with read_attention(
            model, k_top=4, grouping=grouping, readset=readset,
            generator=torch.Generator().manual_seed(0),
        ) as patterns:
            observed = model(images)
        torch.testing.assert_close(observed, original, atol=0, rtol=0)
        with execute_attention(model, patterns):
            assert torch.isfinite(model(images)).all()
