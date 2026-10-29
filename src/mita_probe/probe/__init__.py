"""The top-k neuron probe: read shared reading sets from a ViT, then execute or observe it.

pattern  — which tokens (neurons) each query may read: landmarks, top-k sets, routes, masks
execute  — run selected layers through a plan; capture Q/K, CLS attention, block-input norms
specs    — turn a setting (L1-L4, random controls, budget) into per-layer plans
"""

from .execute import (Plan, capture_cls_attention, capture_input_norms, capture_qk, execute, install,
                      mixed_attention, selected_layers)
from .pattern import (M_SIDES, Pattern, build_pattern, coverage, grid_landmarks, kmeans_landmarks, landmarks,
                      per_query_pattern, random_pattern, spatial_routes, token_mask, union_fraction)
from .specs import base_pattern, batch_generator, expand_rq2, make_plan, resolve

__all__ = ['M_SIDES', 'Pattern', 'Plan', 'base_pattern', 'batch_generator', 'build_pattern', 'capture_cls_attention',
           'capture_input_norms', 'capture_qk', 'coverage', 'execute', 'expand_rq2', 'grid_landmarks', 'install',
           'kmeans_landmarks', 'landmarks', 'make_plan', 'mixed_attention', 'per_query_pattern', 'random_pattern',
           'resolve', 'selected_layers', 'spatial_routes', 'token_mask', 'union_fraction']
