"""Model wrappers: DINO linear evaluation head, the model registry and checkpoint verification."""

import hashlib

import pytest
import timm
import torch

from fastweight_vit.models import NAMES, data_reference, load_model, verified
from fastweight_vit.models.dino import DINO_NAME, N_LAST_BLOCKS, DinoLinear, load_dino_model
from fastweight_vit.execution import Plan, capture_qk, execute


def make_model():
    torch.manual_seed(0)
    return load_dino_model(None, pretrained=False)


def test_features_are_normed_cls_of_last_four_blocks():
    model = make_model()
    x = torch.randn(2, 3, 224, 224)
    seen = {}
    model.linear.register_forward_pre_hook(lambda module, inputs: seen.update(features=inputs[0]))
    with torch.no_grad():
        model(x)
        reference = model.backbone.forward_intermediates(x, indices=N_LAST_BLOCKS, norm=True,
                                                         return_prefix_tokens=True, intermediates_only=True,
                                                         output_fmt='NLC')
    features = seen['features']
    assert features.shape == (2, N_LAST_BLOCKS * 384)
    expected = torch.cat([prefix[:, 0] for _, prefix in reference], -1)
    assert torch.allclose(features, expected, atol=1e-5)
    with torch.no_grad():
        assert torch.allclose(features[:, -384:], model.backbone(x), atol=1e-5)


def test_head_loads_official_key_layout(tmp_path):
    linear = torch.nn.Linear(N_LAST_BLOCKS * 384, 1000)
    path = tmp_path / 'head.pth'
    torch.save({'state_dict': {f'module.linear.{key}': value for key, value in linear.state_dict().items()}},
               path)
    model = load_dino_model(path, pretrained=False)
    assert torch.equal(model.linear.weight, linear.weight) and torch.equal(model.linear.bias, linear.bias)


def test_readings_see_backbone_blocks():
    model = make_model()
    assert isinstance(model, DinoLinear) and timm.list_models(DINO_NAME.split('.')[0])
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad(), capture_qk(model) as qk:
        original = model(x)
    assert len(qk) == 12 and qk[1][0].shape[2] == 197
    q = qk[1][0]
    full = torch.ones(*q.shape[:3], q.shape[2], dtype=torch.bool)
    with torch.no_grad(), execute(model, {layer: Plan(token_mask=full) for layer in range(1, 13)}):
        dense = model(x)
    assert (dense - original).abs().max() < 1e-4


def test_registry_names_and_data_reference(tmp_path):
    assert NAMES == ('deit', 'deit_b', 'dino', 'mita')
    model = make_model()
    assert data_reference(model) is model.backbone
    with pytest.raises(ValueError):
        load_model('resnet')
    path = tmp_path / 'weights.bin'
    path.write_bytes(b'abc')
    assert verified(path, hashlib.sha256(b'abc').hexdigest()) == path
    with pytest.raises(ValueError):
        verified(path, '0' * 64)
