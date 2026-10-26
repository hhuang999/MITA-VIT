import numpy as np
import torch
import timm

from mita_probe.models import read_attention, capture_cls_attention
from mita_probe.probe import ReadingPattern
from mita_probe.rq3_extension import (
    capture_input_norms, sink_tokens, replace_category, replace_random,
    numpy_heatmap_metrics,
)


def test_sink_uses_midpoint_median_strict_threshold():
    norms = torch.tensor([[1., 2., 4., 9.], [1., 1., 1., 4.]])
    assert sink_tokens(norms).tolist() == [[False]*4, [False, False, False, True]]


def test_category_refills_without_leakage_duplicates_and_preserves_routes():
    scores = torch.tensor([[[[9., 8., 7., 6., 5., 4.]]]])
    pattern = ReadingPattern(torch.tensor([[[[0, 1, 2]]]]), torch.zeros(1, 1, 6, dtype=torch.long))
    forbidden = torch.tensor([[False, True, True, False, False, True]])
    replaced, valid, dose = replace_category(pattern, scores, forbidden)
    assert replaced.readsets.tolist() == [[[[0, 3, 4]]]]
    assert valid.item() and dose.item() == 2
    assert replaced.routes is pattern.routes
    forbidden[:, 4] = True
    replaced, valid, _ = replace_category(pattern, scores, forbidden)
    assert not valid.item()
    assert torch.equal(replaced.readsets, pattern.readsets)


def test_random_dose_is_exact_never_refills_deleted_or_original_keys():
    scores = torch.arange(12.).reshape(1, 1, 1, 12).expand(2, 2, 3, 12)
    reads = scores.topk(4, -1).indices
    pattern = ReadingPattern(reads, torch.zeros(2, 2, 12, dtype=torch.long))
    number = torch.tensor([0, 2, 4]).reshape(1, 1, 3).expand(2, 2, 3)
    result = replace_random(pattern, scores, number, torch.Generator().manual_seed(0))
    overlap = (result.readsets[..., :, None] == reads[..., None, :]).any(-1).sum(-1)
    assert torch.equal(4 - overlap, number)
    assert (result.readsets.sort(-1).values.diff(dim=-1) > 0).all()


def test_cpu_heatmap_matches_hand_example_and_invalid_mask():
    mass, hit, peak = numpy_heatmap_metrics(
        np.array([[1, 9, 2, 0], [0, 0, 0, 0]]),
        np.array([[True, False, True, False]]*2),
        np.array([[True, False, True, True]]*2))
    assert mass[0] == 1 and hit[0] == 1 and peak[0] == 2
    assert np.isnan(mass[1]) and np.isnan(hit[1])


def test_register_prefix_exclusion_norm_timing_and_observe_identity():
    model = timm.create_model('vit_small_patch14_reg4_dinov2', pretrained=False,
                              img_size=224, depth=2).eval()
    images = torch.randn(1, 3, 224, 224)
    before_norm = []
    handle = model.blocks[0].norm1.register_forward_pre_hook(
        lambda module, args: before_norm.append(args[0].clone()))
    with torch.inference_mode():
        original = model(images)
        scores = {}
        with read_attention(model, k_top=16, region_scores=scores) as patterns, \
                capture_input_norms(model) as norms, capture_cls_attention(model) as cls:
            observed = model(images)
    handle.remove()
    torch.testing.assert_close(observed, original, atol=0, rtol=0)
    assert model.num_prefix_tokens == 5
    assert norms[1].shape == (1, 256) and cls['heatmap'].shape == (1, 256)
    torch.testing.assert_close(norms[1], before_norm[-1][:, 5:].norm(dim=-1))
    assert scores[1].shape == (1, 6, 16, 261)
    assert patterns[1].routes.shape == (1, 6, 261)
    assert all(not block._forward_pre_hooks for block in model.blocks)
