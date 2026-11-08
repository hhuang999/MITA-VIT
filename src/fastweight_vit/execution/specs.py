"""Turn setting specs (mode L3 / rand / drop, budget k, routing) into per-layer plans.

L3: every query reads only the shared top-k set of its region (m regions, k units each; the reading sets).
rand: random sets of the same size in place of the reading sets. drop: the attention branch outputs zero.
"""

import torch

from .execute import Plan
from .pattern import build_pattern, random_pattern, token_mask

MODES = {'L3', 'rand', 'drop'}
PATTERN_KEYS = ('m_side', 'k', 'routing', 'cls', 'scoring', 'landmark_mode')


def resolve(spec, main):
    """Fill a setting with the main budget; summary-free modes ignore summary fields."""
    full = {'landmark_mode': 'clean', **main, **spec}
    if full['mode'] not in MODES:
        raise ValueError(f"Unknown mode {full['mode']}")
    return full


def expand_settings(config):
    """All-layer settings of config['sweep'] with the k sweep and seeds; drop exact duplicates, keep first name."""
    out, seen = [], set()
    for entry in config['sweep']['settings']:
        values = {'k': config['sweep']['k_sweep'], 'm': config['sweep']['m_sweep']}.get(entry.get('sweep'), [None])
        seeds = config['random_seeds'] if entry.get('seeds') else [-1]
        for value in values:
            for seed in seeds:
                spec = {key: val for key, val in entry.items() if key not in {'sweep', 'seeds'}}
                if entry.get('sweep') == 'k':
                    spec['k'] = value
                elif entry.get('sweep') == 'm':
                    spec['m_side'] = value
                spec = resolve(spec, config['main']) | {'seed': seed}
                key = tuple(sorted((k, v) for k, v in spec.items() if k != 'name'))
                if key not in seen:
                    seen.add(key)
                    out.append(spec)
    return out


def batch_generator(device, seed, layer, k, batch_start):
    """Independent random stream per (seed, layer, k, batch)."""
    generator = torch.Generator(device=device)
    return generator.manual_seed(((seed * 1_000_003 + layer) * 1_009 + k) * 1_000_033 + batch_start)


def base_pattern(spec, q, k, scale, grid):
    return build_pattern(q, k, m_side=spec['m_side'], k_top=spec['k'], scale=scale, grid=grid,
                         routing=spec['routing'], cls=spec['cls'], scoring=spec['scoring'],
                         landmark_mode=spec['landmark_mode'])


def make_plan(spec, q, k, scale, grid, *, layer=0, batch_start=0):
    mode = spec['mode']
    if mode == 'drop':
        return Plan(drop=True)
    pattern, _ = base_pattern(spec, q, k, scale, grid)
    if mode == 'rand':
        pattern = random_pattern(pattern, batch_generator(q.device, spec['seed'], layer, spec['k'], batch_start))
    return Plan(token_mask=token_mask(pattern))
