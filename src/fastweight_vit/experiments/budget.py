"""Attention budget ledger (T1 pilot, T3 formal; decisions B27-B29) on ImageNet-S.

For each image and layer, where the budget of all queries (and of the CLS query alone) goes: CLS,
high-norm patches, object, background, boundary (`budget.csv`). Object focus is the same-distance-ring
enrichment of the budget each patch receives in layers 9-12 (attention_mass), with the high-norm patches of
those layers excluded and, as a secondary result, with every patch (`enrichment.csv`, also per layer).

Also written:
- heads.csv: per batch, layer and head, sums for the sink diagnostics: share on high-norm patches, value
  norm of high-norm patches over the median ordinary patch, effective rank of the head output (overall and
  for heads that give more than half their budget to high-norm patches);
- selected.csv (MiTA figure 4): per layer, the union of tokens read by any query (MiTA's top-k sets; each
  query's top-25 for dense models), its size, object share and ring enrichment;
- raw/batch_<start>.npz: deep scores, high-norm masks, object masks and predictions, for paired analyses;
- pilot stage only: value_means.pt, per layer the mean value vector of high-norm patches per head (the
  "mean" sink edit of sink_use).
"""

import logging
import time

import numpy as np
import torch

from .. import plausibility
from ..budget import CATEGORIES, categories, instrument, ring_enrichment, shares
from ..interventions import sink_tokens
from ..io import write_json
from ..models import load_model
from ..execution import capture_input_norms
from . import common, datasets
from .common import Sink

STAGES = ('pilot', 'formal')
SHARE_FIELDS = [f'share_{name}' for name in CATEGORIES]
BUDGET = ['index', 'layer', 'query'] + SHARE_FIELDS
ENRICHMENT = ['index', 'scope', 'exclude', 'n_excluded', 'mass', 'enrichment']
HEADS = ['start', 'layer', 'head', 'images', 'high_share', 'dominant', 'value_ratio', 'value_ratio_n', 'rank',
         'rank_dominant', 'rank_dominant_n']
SELECTED = ['index', 'layer', 'fraction', 'object_fraction', 'enrichment']


@torch.no_grad()
def ledger(model, images, foreground, valid, settings, *, values=True, selected=False, prefix=None):
    """One forward: logits, per-layer records and high-norm masks [B, layers, patches]."""
    prefix = model.num_prefix_tokens if prefix is None else prefix
    with instrument(model, values=values, rank=values, selected=selected) as store, \
            capture_input_norms(model, prefix) as norms:
        logits = model(images)
    high = torch.stack([sink_tokens(norms[layer], settings['norm_ratio']) for layer in sorted(norms)], 1)
    return logits, store, high


def deep_scores(store, layers, prefix):
    return sum(store[layer]['received'].float().mean(1) for layer in layers)[:, prefix:]


def enrichment_rows(ids, scores, foreground, valid, exclude, scope, generator_seed):
    other = foreground.roll(1, 0) if len(ids) > 1 else torch.zeros_like(foreground)
    rows = []
    for name, mask in (('high', exclude), ('none', torch.zeros_like(exclude))):
        values = plausibility.measures(scores, foreground, valid & ~mask, other,
                                       generator=torch.Generator().manual_seed(generator_seed))
        n_excluded = (mask & valid).sum(1).tolist()
        rows += [dict(index=index, scope=scope, exclude=name, n_excluded=n_excluded[n], mass=float(values['mass'][n]),
                      enrichment=float(values['enrichment'][n])) for n, index in enumerate(ids)]
    return rows


def head_rows(start, store, high, prefix, n_images):
    rows = []
    for layer in sorted(store):
        record = store[layer]
        received = record['received'].float()  # [B, H, N]
        share = (received[:, :, prefix:] * high[:, None, layer - 1]).sum(-1) / received.sum(-1)
        dominant = share > .5
        norms = record['value_norm'][:, :, prefix:].float()
        has = high[:, layer - 1].any(1)
        # nanquantile: midpoint median with a deterministic CUDA kernel (nanmedian returns indices and has none)
        ordinary = norms.masked_fill(high[:, None, layer - 1], float('nan')).nanquantile(.5, dim=-1)
        high_norm = (norms * high[:, None, layer - 1]).sum(-1) / high[:, None, layer - 1].sum(-1).clamp_min(1)
        ratio = torch.where(has[:, None], high_norm / ordinary, torch.full_like(high_norm, float('nan')))
        rank = record['rank']
        for head in range(received.shape[1]):
            r = ratio[:, head]
            ok = ~torch.isnan(r)
            rows.append(dict(start=start, layer=layer, head=head, images=n_images,
                             high_share=float(share[:, head].sum()), dominant=int(dominant[:, head].sum()),
                             value_ratio=float(r[ok].sum()), value_ratio_n=int(ok.sum()),
                             rank=float(rank[:, head].sum()), rank_dominant=float(rank[dominant[:, head], head].sum()),
                             rank_dominant_n=int(dominant[:, head].sum())))
    return rows


def run(args, config, output, commit):
    """args: model, stage (pilot | formal), smoke, resume. Settings in config['budget']."""
    settings = config['budget']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    dataset, indices = datasets.build(settings['dataset'], model, args.model, config | {'split': args.stage})
    indices = indices[:args.smoke] if args.smoke else indices
    if args.stage == 'pilot' and args.resume:
        raise ValueError('The pilot run accumulates value means in memory; rerun it from the start')
    progress_path, done = common.progress(output, args.resume)
    deep = settings['deep_layers']
    prefix = model.num_prefix_tokens
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, stage=args.stage, dataset=settings['dataset'], images=len(indices),
        preprocessing=config.get('preprocessing'), norm_ratio=settings['norm_ratio'], deep_layers=deep,
        mita_plan=str(getattr(model, 'mita_plan', None)), smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    sinks = dict(budget=Sink(output / 'budget.csv', BUDGET), enrichment=Sink(output / 'enrichment.csv', ENRICHMENT),
                 heads=Sink(output / 'heads.csv', HEADS), selected=Sink(output / 'selected.csv', SELECTED))
    (output / 'raw').mkdir(exist_ok=True)
    sums = {}
    for start, ids, batch in batches(f'budget_{args.model}', dataset, indices):
        images, foreground, valid = (tensor.to(device) for tensor in batch[:3])
        logits, store, high = ledger(model, images, foreground, valid, settings, selected=True)
        if args.stage == 'pilot':
            accumulate_values(model, images, high, prefix, sums)
        rows = []
        for layer in sorted(store):
            codes = categories(foreground, valid, high[:, layer - 1], prefix)
            for query in ('all', 'cls'):
                part = shares(store[layer]['received' if query == 'all' else 'cls'], codes).tolist()
                rows += [dict(index=index, layer=layer, query=query, **dict(zip(SHARE_FIELDS, part[n])))
                         for n, index in enumerate(ids)]
        sinks['budget'].write(rows)
        exclude = high[:, [layer - 1 for layer in deep]].any(1)
        rows = enrichment_rows(ids, deep_scores(store, deep, prefix), foreground, valid, exclude, 'deep', start)
        for layer in sorted(store):
            rows += enrichment_rows(ids, deep_scores(store, [layer], prefix), foreground, valid, high[:, layer - 1],
                                    str(layer), start)
        sinks['enrichment'].write(rows)
        sinks['heads'].write(head_rows(start, store, high, prefix, len(ids)))
        rows = []
        for layer in sorted(store):
            chosen = store[layer]['selected'][:, prefix:]
            fraction = chosen.float().mean(1).tolist()
            fg = foreground & valid
            object_fraction = ((chosen & fg).sum(1) / (chosen & valid).sum(1).clamp_min(1)).tolist()
            enrich = ring_enrichment(chosen, foreground, valid).tolist()
            rows += [dict(index=index, layer=layer, fraction=fraction[n], object_fraction=object_fraction[n],
                          enrichment=enrich[n]) for n, index in enumerate(ids)]
        sinks['selected'].write(rows)
        np.savez_compressed(output / 'raw' / f'batch_{start:06d}.npz', index=np.array(ids),
                            deep_scores=deep_scores(store, deep, prefix).cpu().numpy().astype(np.float32),
                            high=np.packbits(high.cpu().numpy(), axis=-1), foreground=foreground.cpu().numpy(),
                            valid=valid.cpu().numpy(), prediction=logits.argmax(-1).cpu().numpy())
        common.update_checks(output, max_share_error=float(max(
            (shares(store[layer]['received'], categories(foreground, valid, high[:, layer - 1], prefix)).sum(1) - 1)
            .abs().max() for layer in store)))
    if args.stage == 'pilot':
        torch.save({layer: dict(mean=entry['sum'] / entry['count'].clamp_min(1)[:, None], count=entry['count'])
                    for layer, entry in sums.items()}, output / 'value_means.pt')
    common.finish(output, started)
    logging.info('Complete: attention budget (%s) for %s', args.stage, args.model)


@torch.no_grad()
def accumulate_values(model, images, high, prefix, sums):
    """Add each layer's value vectors of high-norm patches, per head, to sums[layer]."""
    handles, captured = [], {}

    def make_hook(layer, attention):
        def hook(module, inputs, output):
            batch, tokens, _ = output.shape
            v = output.reshape(batch, tokens, 3, attention.num_heads, attention.head_dim)[:, :, 2]
            captured[layer] = v[:, prefix:].transpose(1, 2).double()  # [B, H, P, D]
        return hook

    for layer, block in enumerate(model.blocks, 1):
        handles.append(block.attn.qkv.register_forward_hook(make_hook(layer, block.attn)))
    try:
        model(images)
    finally:
        for handle in handles:
            handle.remove()
    for layer, v in captured.items():
        mask = high[:, layer - 1][:, None, :, None].double()
        entry = sums.setdefault(layer, dict(sum=0., count=torch.zeros(v.shape[1], dtype=torch.float64,
                                                                       device=v.device)))
        entry['sum'] = entry['sum'] + (v * mask).sum((0, 2))
        entry['count'] = entry['count'] + float(mask.sum())
