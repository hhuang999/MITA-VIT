"""Sanity check for every map (Adebayo et al., NeurIPS 2018): cascading model-parameter randomisation.

From the classifier down to the patch embedding, one more stage is re-initialised at each level (the
changes accumulate): head (classifier and final norm), block 12, ..., block 1, embed (patch and position
embeddings, CLS token). A map that explains the model must change as its weights are destroyed; a map
that does not (the centre reference) reflects a prior, not the model. Per image, level and method:
Spearman between the map of the randomised model and the map of the trained model. Gradient methods keep
the trained model's top-1 class as their target at every level. Class-agnostic maps (probe, CLS
attention, rollout) cannot depend on the head, so they are unchanged at the 'head' level by construction.
v5 readings (`sanity.methods` entries with a `reading`, decision B39) keep the trained model's lens: the
position probes and mean maps are not re-estimated at each level, so a reading dominated by its fixed mean
map would fail the check. Re-initialisation follows timm's ViT init: linear and conv weights truncated normal (std 0.02), biases 0,
norms (1, 0), embeddings truncated normal (std 0.02).
"""

import copy
import logging
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ..io import write_json
from ..metrics import spearman
from ..models import load_model
from ..models.dino import DinoLinear
from ..execution import resolve
from . import common, datasets
from .common import Sink
from .explanations import method_scores


def stages(model):
    """[(level name, modules, extra parameters)] from the output down to the input."""
    backbone = model.backbone if isinstance(model, DinoLinear) else model
    head = [model.linear] if isinstance(model, DinoLinear) else [model.head]
    out = [('head', head + [backbone.norm], [])]
    out += [(f'block{layer}', [backbone.blocks[layer - 1]], []) for layer in range(len(backbone.blocks), 0, -1)]
    out.append(('embed', [backbone.patch_embed], [backbone.pos_embed, backbone.cls_token]))
    return out


@torch.no_grad()
def reinitialise(modules, parameters, generator):
    for module in modules:
        for sub in module.modules():
            if isinstance(sub, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(sub.weight, std=0.02, generator=generator)
                if sub.bias is not None:
                    sub.bias.zero_()
            elif isinstance(sub, nn.LayerNorm):
                sub.weight.fill_(1)
                sub.bias.zero_()
    for parameter in parameters:
        nn.init.trunc_normal_(parameter, std=0.02, generator=generator)


def run(args, config, output, commit):
    """args: model, smoke. Settings in config['sanity']; methods from config['explanations']."""
    settings, explain_settings = config['sanity'], config['explanations']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    trained = load_model(args.model, device)
    dataset, indices = datasets.build(settings['dataset'], trained, args.model, config)
    rng = np.random.default_rng(settings['subset_seed'])
    indices = sorted(rng.choice(indices, min(settings['images'], len(indices)), replace=False).tolist())
    indices = indices[:args.smoke] if args.smoke else indices
    methods = settings.get('methods') or [entry for entry in explain_settings['methods'] if entry['method'] != 'mask']
    spec = resolve(explain_settings['probe_spec'], config['main'])
    lens = None
    if any('reading' in entry for entry in methods):  # v5 readings: the trained model's lens stays fixed
        from .correct import load_lens
        lens = load_lens(config['correct'], args.model, 'crop', device)
    write_json(output / 'metadata.json', common.provenance(
        commit, model=args.model, dataset=settings['dataset'], images=len(indices),
        methods=[entry['name'] for entry in methods], levels=[name for name, _, _ in stages(trained)],
        smoke_images=args.smoke))
    loader = DataLoader(Subset(dataset, indices), batch_size=config['batch_size'],
                        num_workers=config['num_workers'], shuffle=False)

    def maps(model, targets=None):
        """{method: [N, 196]} on the CPU, plus the top-1 classes; the same noise stream at every level."""
        out, top1 = {entry['name']: [] for entry in methods}, []
        for start, (images, *_) in zip(range(0, len(indices), config['batch_size']), loader):
            images = images.to(device)
            with torch.no_grad():
                top1.append(model(images).argmax(-1))
            target = None if targets is None else targets[start:start + len(images)]
            noise = torch.Generator(device=device).manual_seed(settings['random_seed'] * 1_000_003 + start)
            if lens is not None:
                from .correct import all_maps
                lens_maps = all_maps(model, images, [entry for entry in methods if 'reading' in entry], lens)
            for entry in methods:
                value = lens_maps[entry['name']] if 'reading' in entry else \
                    method_scores(entry, model, images, None, spec, noise, target)
                out[entry['name']].append(value.float().cpu())
        return {name: torch.cat(value) for name, value in out.items()}, torch.cat(top1)

    started = time.perf_counter()
    reference, targets = maps(trained)
    sink = Sink(output / 'sanity.csv', ['model', 'index', 'level', 'method', 'spearman', 'top1_kept'])
    model = copy.deepcopy(trained)
    generator = torch.Generator(device=device).manual_seed(settings['init_seed'])
    for level, modules, parameters in stages(model):
        reinitialise(modules, parameters, generator)
        current, top1 = maps(model, targets)
        kept = top1.eq(targets).int().tolist()
        rows = []
        for name in current:
            rho = spearman(current[name].double(), reference[name].double()).tolist()
            rows += [dict(model=args.model, index=index, level=level, method=name, spearman=rho[n], top1_kept=kept[n])
                     for n, index in enumerate(indices)]
        sink.write(rows)
        logging.info('level %s done; %.1fs', level, time.perf_counter() - started)
    common.finish(output, started)
