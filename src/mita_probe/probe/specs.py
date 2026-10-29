"""Turn setting specs (mode L1-L4 / rand / randC / drop, budget, routing) into per-layer plans.

L1: each query reads its own top-k. L2: region summaries only. L3: shared top-k of its region.
L4: L3 plus summaries (the MiTA mixture). rand/randC: random sets in place of L3/L4. drop: no attention.
"""

import torch

from .execute import Plan
from .pattern import build_pattern, per_query_pattern, random_pattern, token_mask

MODES = {'L1', 'L2', 'L3', 'L4', 'rand', 'randC', 'drop'}
PATTERN_KEYS = ('m_side', 'k', 'routing', 'cls', 'scoring', 'landmark_mode')


def resolve(spec, main):
    """Fill a setting with the main budget; summary-free modes ignore summary fields."""
    full = {'landmark_mode': 'clean', **main, **spec}
    if full['mode'] not in MODES:
        raise ValueError(f"Unknown mode {full['mode']}")
    return full


def expand_rq2(config):
    """All-layer settings with sweeps and seeds; drop exact duplicates, keep first name."""
    out, seen = [], set()
    for entry in config['rq2']['settings']:
        values = {'k': config['rq2']['k_sweep'], 'm': config['rq2']['m_sweep']}.get(entry.get('sweep'), [None])
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
    """Independent stream per (seed, layer, k, batch); identical sets for rand and randC."""
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
    summary = dict(summary_side=spec['m_side'], summary_scale=spec['summary_scale'],
                   landmark_mode=spec['landmark_mode'])
    if mode == 'L2':
        return Plan(**summary)
    if mode == 'L1':
        return Plan(token_mask=token_mask(per_query_pattern(q, k, spec['k'], scale)))
    pattern, _ = base_pattern(spec, q, k, scale, grid)
    if mode in {'rand', 'randC'}:
        pattern = random_pattern(pattern, batch_generator(q.device, spec['seed'], layer, spec['k'], batch_start))
    mask = token_mask(pattern)
    return Plan(token_mask=mask, **summary) if mode in {'L4', 'randC'} else Plan(token_mask=mask)
