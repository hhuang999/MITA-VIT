"""Part reliance (appendix B): does the model depend more on an animal's head patches than on its body patches?

PartImageNet_Seg (ImageNet training images; see fastweight_vit.data.partimagenet), each model's own ImageNet
preprocessing. For each image, the model's dependence on each part is measured by count-matched patch removal in
every layer (the faithfulness removal, faithfulness.run_kept). For each count r in `removed` and each seed, r
random patches of one part are removed, for every part with at least r assigned patches; controls remove r random
object patches or r random pure-background patches. Equal counts make the parts comparable per patch. A patch
belongs to a part when at least half of its pixels are object and that part has most of them.

reliance.csv: per image, condition, part, seed and count, agreement with the original top-1, its probability
before and after, and KL(original || reduced). The head - body comparison is made in scripts/make_tables.py.
"""

import logging
import time

import torch

from ..data.build import build
from ..data.partimagenet import BACKGROUND
from ..faithfulness import measures, run_kept
from ..io import write_json
from ..models import load_model
from . import common
from .common import Sink

FIELDS = ['model', 'index', 'condition', 'part', 'seed', 'removed', 'agree', 'p_original', 'p_target', 'kl']


def assignment(fractions):
    """[B, 196, 41] pixel fractions -> [B, 196] majority part of mostly-object patches, else -1."""
    parts = fractions[..., :BACKGROUND]
    return torch.where(parts.sum(-1) >= 0.5, parts.argmax(-1), torch.full_like(parts[..., 0], -1).long())


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
    """args: model, smoke, resume. Settings in config['parts']."""
    settings = config['parts']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    dataset, indices = build('partimagenet', model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, dataset='partimagenet_seg', images=len(indices), smoke_images=args.smoke,
        resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    sink = Sink(output / 'reliance.csv', FIELDS)
    prefix = model.num_prefix_tokens
    for start, ids, (images, _, _, fractions, _) in batches(f'parts_{args.model}', dataset, indices):
        images, fractions = images.to(device), fractions.to(device)
        assign = assignment(fractions)
        with torch.no_grad():
            original = model(images)
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
        sink.write(rows)
    common.finish(output, started)
    logging.info('Complete: part reliance for %s', args.model)
