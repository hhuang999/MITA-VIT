"""Check the installed model architecture without downloading weights."""

import timm
import torch


def test_deit_small_architecture_and_cpu_forward():
    torch.set_num_threads(2)
    model = timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=False).eval()
    assert len(model.blocks) == 12
    assert model.patch_embed.num_patches == 196
    assert model.num_prefix_tokens == 1
    assert model.blocks[0].attn.num_heads == 6
    assert model.embed_dim == 384
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 1000)
    assert torch.isfinite(logits).all()
