"""RQ3 parts (E8): which object parts do the maps favour, and which parts does the model rely on?

PartImageNet_Seg (ImageNet training images; see mita_probe.data.partimagenet), each model's own ImageNet
preprocessing. The methods are the explanations list (without the object mask) plus, when `layerwise` is
set, the probe at each single layer. Two outputs:

parts.csv, one row per image, method and part present in the crop:
- area: the part's share of the object pixels;
- mass: the part's share of the method's (non-negative) score mass on the object, each patch's score
  split by its pixel fractions;
- enrichment = mass / area (> 1: the map favours the part beyond its size);
- auroc: probability that a patch of this part outscores a patch of another part of the same object
  (hard assignment: the majority part of a patch whose pixels are at least half object);
- patches: the number of patches assigned to the part.

reliance.csv, the model's own dependence on each part, by count-matched patch removal in every layer (the
faithfulness removal). For each count r in `removed` and each seed, r random patches of one part are
removed, for every part with at least r assigned patches; controls remove r random object patches or r
random pure-background patches. Equal counts make the parts comparable per patch. Recorded: agreement
with the original top-1, its probability before and after, and KL.
"""

import logging
import time

import torch

from ..data.partimagenet import BACKGROUND
from ..faithfulness import measures, run_kept
from ..io import write_json
from ..models import load_model
from ..probe import resolve
from . import common, datasets
from .common import Sink
from .explanations import layerwise_methods, method_scores

PART_FIELDS = ['model', 'index', 'method', 'part', 'area', 'mass', 'enrichment', 'auroc', 'patches']
RELIANCE_FIELDS = ['model', 'index', 'condition', 'part', 'seed', 'removed', 'agree', 'p_original', 'p_target',
                   'kl']


def assignment(fractions):
    """[B, 196, 41] pixel fractions -> [B, 196] majority part of mostly-object patches, else -1."""
    parts = fractions[..., :BACKGROUND]
    return torch.where(parts.sum(-1) >= 0.5, parts.argmax(-1), torch.full_like(parts[..., 0], -1).long())


def part_measures(scores, fractions, assign):
    """Dict of [B, 40] tensors (NaN where undefined) for one method."""
    parts = fractions[..., :BACKGROUND]
    obj = parts.sum(-1)
    s = scores.clamp_min(0)
    area = parts.sum(1) / obj.sum(1, keepdim=True)
    mass = torch.einsum('bt,btp->bp', s, parts) / (s * obj).sum(1, keepdim=True)
    onehot = torch.nn.functional.one_hot(assign.clamp_min(0), BACKGROUND).float() * (assign >= 0)[..., None]
    others = (assign >= 0)[..., None].float() - onehot
    beats = (scores[:, :, None] > scores[:, None, :]).float() + 0.5 * (scores[:, :, None] == scores[:, None, :])
    wins = torch.einsum('bip,bij,bjp->bp', onehot, beats, others)
    pairs = onehot.sum(1) * others.sum(1)
    nan = torch.tensor(float('nan'), device=scores.device)
    return dict(area=area, mass=mass, enrichment=torch.where(area > 0, mass / area, nan),
                auroc=torch.where(pairs > 0, wins / pairs.clamp_min(1), nan), patches=onehot.sum(1))


def removals(assign, fractions, settings, start, device):
    """[(row in batch, condition, part, seed, removed, keep [196] bool)] for the reliance test."""
    counts = torch.nn.functional.one_hot(assign.clamp_min(0), BACKGROUND).mul((assign >= 0)[..., None]).sum(1)
    background = fractions[..., :BACKGROUND].sum(-1) == 0
    out = []
    for seed in settings['seeds']:
        generator = torch.Generator(device=device).manual_seed(seed * 1_000_003 + start)
        noise = torch.rand(assign.shape, generator=generator, device=device)
        for n in range(assign.shape[0]):
            for r in settings['removed']:
                eligible = (counts[n] >= r).nonzero()[:, 0].tolist()
                if not eligible:
                    continue
                candidates = [('part', p, assign[n] == p) for p in eligible] + [('object', -1, assign[n] >= 0)]
                if int(background[n].sum()) >= r:
                    candidates.append(('background', -1, background[n]))
                for condition, part, members in candidates:
                    chosen = noise[n].masked_fill(~members, -1).topk(r).indices
                    keep = torch.ones(assign.shape[1], dtype=torch.bool, device=device)
                    keep[chosen] = False
                    out.append((n, condition, part, seed, r, keep))
    return out


def run(args, config, output, commit):
    """args: model, smoke, resume. Settings in config['parts']; methods from config['explanations']."""
    settings, explain_settings = config['parts'], config['explanations']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    dataset, indices = datasets.build('partimagenet', model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    methods = [entry for entry in explain_settings['methods'] if entry['method'] != 'mask']
    if settings['layerwise']:
        methods += layerwise_methods(len(model.blocks))
    spec = resolve(explain_settings['probe_spec'], config['main'])
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, dataset='partimagenet_seg', images=len(indices),
        methods=[entry['name'] for entry in methods], probe_spec=spec, smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    part_sink, reliance_sink = Sink(output / 'parts.csv', PART_FIELDS), Sink(output / 'reliance.csv', RELIANCE_FIELDS)
    prefix = model.num_prefix_tokens
    for start, ids, (images, _, _, fractions, _) in batches(f'parts_{args.model}', dataset, indices):
        images, fractions = images.to(device), fractions.to(device)
        assign = assignment(fractions)
        with torch.no_grad():
            original = model(images)
        noise = torch.Generator(device=device).manual_seed(settings['random_seed'] * 1_000_003 + start)
        rows = []
        for entry in methods:
            scores = method_scores(entry, model, images, None, spec, noise).float()
            values = {key: value.tolist() for key, value in part_measures(scores, fractions, assign).items()}
            rows += [dict(model=args.model, index=index, method=entry['name'], part=p,
                          **{key: values[key][n][p] for key in ('area', 'mass', 'enrichment', 'auroc')},
                          patches=int(values['patches'][n][p]))
                     for n, index in enumerate(ids) for p in range(BACKGROUND) if values['area'][n][p] > 0]
        part_sink.write(rows)
        plan = removals(assign, fractions, settings, start, device)
        rows = []
        for chunk in range(0, len(plan), config['batch_size']):
            part_plan = plan[chunk:chunk + config['batch_size']]
            members = torch.tensor([item[0] for item in part_plan], device=device)
            keep = torch.stack([item[5] for item in part_plan])
            keep = torch.cat([torch.ones(len(keep), prefix, dtype=torch.bool, device=device), keep], 1)
            with torch.no_grad():
                values = measures(original[members], run_kept(model, images[members], keep))
            p_original = original[members].softmax(-1).max(-1).values.tolist()
            values = {key: value.tolist() for key, value in values.items()}
            rows += [dict(model=args.model, index=ids[n], condition=condition, part=part, seed=seed, removed=r,
                          agree=values['agree'][m], p_original=p_original[m], p_target=values['p_target'][m],
                          kl=values['kl'][m])
                     for m, (n, condition, part, seed, r, _) in enumerate(part_plan)]
        reliance_sink.write(rows)
    common.finish(output, started)
    logging.info('Complete: parts for %s', args.model)
