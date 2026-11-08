"""Top-k reading sets read from a ViT's own attention, and running the model restricted to them.

pattern  — which tokens (units) each query may read: region landmarks, shared top-k sets, routes, masks
execute  — run selected layers through a plan (restricted sets or no attention); capture Q/K, block-input norms
specs    — turn a setting (reading sets, random sets, no attention; budget k) into per-layer plans
"""

from .execute import Plan, capture_input_norms, capture_qk, execute, install, mixed_attention, selected_layers
from .pattern import Pattern, build_pattern, grid_landmarks, landmarks, random_pattern, spatial_routes, token_mask
from .specs import base_pattern, batch_generator, expand_settings, make_plan, resolve

__all__ = ['Pattern', 'Plan', 'base_pattern', 'batch_generator', 'build_pattern', 'capture_input_norms', 'capture_qk',
           'execute', 'expand_settings', 'grid_landmarks', 'install', 'landmarks', 'make_plan', 'mixed_attention',
           'random_pattern', 'resolve', 'selected_layers', 'spatial_routes', 'token_mask']
