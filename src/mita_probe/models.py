"""Observe DeiT-S Q/K, then apply fixed original-model summaries through timm masks."""

from contextlib import contextmanager

from .runtime import configure_runtime

configure_runtime()

import timm
import torch

from .baselines import random_readsets
from .assets import MODEL_NAME, MODEL_REPO, MODEL_REVISION
from .grouping import grid_landmarks, kmeans_landmarks
from .probe import ReadingPattern, execution_mask, read_patterns


def load_model(device: str = "cpu", pretrained: bool = True):
    return timm.create_model(
        MODEL_NAME, pretrained=pretrained,
        pretrained_cfg_overlay={"hf_hub_id": f"{MODEL_REPO}@{MODEL_REVISION}"},
    ).eval().to(device)


def _selected_layers(model, layers):
    if model.training:
        raise ValueError("Attention probing requires model.eval()")
    selected = tuple(range(1, len(model.blocks) + 1)) if layers is None else tuple(layers)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select a nonempty set of distinct layers")
    if any(layer < 1 or layer > len(model.blocks) for layer in selected):
        raise ValueError("Layer numbers are one-based")
    return selected


@contextmanager
def read_attention(
    model,
    *,
    k_top: int,
    m_side: int = 4,
    grouping: str = "grid",
    readset: str = "ours",
    generator: torch.Generator | None = None,
    layers: tuple[int, ...] | None = None,
    grouping_counts: dict | None = None,
):
    """Capture the latest original forward; leave this context before executing it.

    Layer keys are one-based. Random baselines require a caller-owned generator,
    which should persist across batches. Returned integer tensors retain no graph.
    """
    selected = _selected_layers(model, layers)
    if grouping not in {"grid", "kmeans"} or readset not in {"ours", "random"}:
        raise ValueError("Use grid/kmeans grouping and ours/random reading sets")
    if readset == "random" and generator is None:
        raise ValueError("Random reading sets require a seeded torch.Generator")
    if model.num_prefix_tokens != 1:
        raise ValueError("This probe expects one CLS token")
    if not 1 <= k_top <= model.patch_embed.num_patches + 1:
        raise ValueError("k_top must lie between 1 and the number of tokens")
    if not 1 <= m_side <= min(model.patch_embed.grid_size):
        raise ValueError("m_side must fit within the patch grid")
    patterns = {}
    handles = []

    def make_hook(layer, attention):
        @torch.no_grad()
        def capture(module, inputs, output):
            batch, tokens, _ = output.shape
            qkv = output.reshape(
                batch, tokens, 3, attention.num_heads, attention.head_dim
            ).permute(2, 0, 3, 1, 4)
            queries, keys, _ = qkv.unbind(0)
            queries, keys = attention.q_norm(queries), attention.k_norm(keys)
            patches = queries[:, :, 1:]
            landmarks = grid_landmarks(patches, model.patch_embed.grid_size, (m_side, m_side))
            if grouping == "kmeans":
                landmarks = kmeans_landmarks(patches, landmarks)
            pattern = read_patterns(queries, keys, landmarks, k_top, attention.scale)
            if grouping_counts is not None:
                distances = (patches.square().sum(-1, keepdim=True)
                             + landmarks.square().sum(-1).unsqueeze(-2)
                             - 2 * patches @ landmarks.transpose(-2, -1))
                assignments = distances.argmin(-1)
                grouping_counts[layer] = {
                    "euclidean_patch_clusters": torch.nn.functional.one_hot(
                        assignments, m_side ** 2).sum(-2),
                    "dot_product_routes_including_cls": torch.nn.functional.one_hot(
                        pattern.routes, m_side ** 2).sum(-2),
                }
            if readset == "random":
                pattern = random_readsets(pattern, generator)
            patterns[layer] = pattern
        return capture

    try:
        for layer in selected:
            attention = model.blocks[layer - 1].attn
            handles.append(attention.qkv.register_forward_hook(make_hook(layer, attention)))
        yield patterns
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def execute_attention(model, patterns: dict[int, ReadingPattern]):
    """Freeze the captured reading sets; timm still computes each query's own logits."""
    selected = _selected_layers(model, patterns)
    handles = []

    def make_hook(pattern):
        mask = execution_mask(pattern)

        def restrict(module, args, kwargs):
            if kwargs.get("attn_mask") is not None or kwargs.get("is_causal", False):
                raise ValueError("The DeiT probe expects unmasked, noncausal self-attention")
            if args[0].shape[:2] != (mask.shape[0], mask.shape[2]):
                raise ValueError("Execute on the same image batch used to read the patterns")
            return args, {**kwargs, "attn_mask": mask}
        return restrict

    try:
        for layer in selected:
            handles.append(model.blocks[layer - 1].attn.register_forward_pre_hook(
                make_hook(patterns[layer]), with_kwargs=True
            ))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_cls_attention(model):
    """Observe final-layer CLS-to-patch attention, averaged over heads."""
    attention = model.blocks[-1].attn
    captured = {}

    @torch.no_grad()
    def hook(module, inputs, output):
        batch, tokens, _ = output.shape
        q, keys, _ = output.reshape(batch, tokens, 3, attention.num_heads,
                                    attention.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, keys = attention.q_norm(q), attention.k_norm(keys)
        weights = ((q[:, :, :1] * attention.scale) @ keys.transpose(-2, -1)).softmax(-1)
        captured["heatmap"] = weights[:, :, 0, 1:].mean(1)

    handle = attention.qkv.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()
