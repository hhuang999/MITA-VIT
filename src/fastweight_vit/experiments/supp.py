"""Phase-1 supplementary analyses X1-X5 (decision B46; exploratory, post hoc). Changes no registered verdict.

Stage `heldout` (one run per model; the v5 held-out images and linear probes):
- X5(a): a two-layer MLP position probe per layer (D -> 1,024 -> GELU -> 196; standardised input, Adam, lr 0.01,
  batch 4,096, at most 30 epochs, seed 0), trained on the first 1,800 of the 2,000 probe-training images, early
  stopping on the last 200 (by image). mlp_probe.csv: per test image and layer, exact / top-5 accuracy and the
  expected distance error of the linear and the MLP probe. mlp_probes.pt (not committed) and mlp_sigmas.json
  (blur width matched to the MLP probe's displacement) feed the eval stage.
- X4: globalisation on the 2,000 test held-out images, layers 6-12, high-norm tokens excluded. g_j = cosine of
  token j with the mean of the image's other (non-high-norm) patch tokens. Within each image the tokens are put in
  5 groups by g; per image, Spearman of group number with the probe's exact accuracy, prediction entropy and C2's
  expected move distance, overall and within 4 distance-to-centre rings (49 positions each, averaged over rings).
  global.csv.

Stage `eval` (one run per model and dataset; `supp.dataset` imagenet_s or clickme): one forward with per-head
attention, ||W_O v|| and ||v||, then every map below (layers 9-12 summed unless named _l<layer>):
- X1: C1C3, C2C3, C1C2 (fixed order C3 -> C2 -> C1, v5 mean maps), C3_centre (C3 minus a centre Gaussian scaled
  to the C3 mean map's spatial spread; reported only under branch X1-A);
- X2: C3v (||v||), C3shuffle_s<seed> (||W_O v|| permuted over the 196 patch tokens within image, layer, head;
  seeds 0-4), C3headmean (per-head mean weight);
- X3: R0_l, C3_l, no_high_l for layers 6-12, and dose.csv (share of R0 on high-norm tokens, layers 9-12);
- X5: C2MLP, C2conf (linear probe, only tokens with top probability >= 0.5 are moved); and, reported only under
  branch version C, blurMLP (blur matched to the MLP probe) and C3C2MLP.
Outputs as `correct`: plausibility.csv + faithfulness.csv (SRG maps in `srg_maps`) for ImageNet-S,
alignment.csv for ClickMe; head_weights.csv (mean ||W_O v|| per layer and head, X2 branch "head weighting").
"""

import json
import logging
import time

import numpy as np
import torch
from timm.data import resolve_model_data_config
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .. import plausibility, readings
from ..data.clickme import ClickMe
from ..faithfulness import curve, rank
from ..interventions import sink_tokens
from ..io import write_csv, write_json
from ..models import IMAGENET_PREPROCESSING, data_reference, load_model
from ..runtime import data_root, project_path
from . import common, datasets
from .common import Sink
from .correct import load_lens
from .diagnose import HeldoutImages, preprocessing_of
from .explanations import PLAUSIBILITY, _rows
from .explore_c2 import distances, error_stats
from .human_alignment import MEASURES, alignment

STAGES = ('heldout', 'eval')
DEEP = [9, 10, 11, 12]
LAYERWISE = list(range(6, 13))
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------------------------------------
# X5(a): the MLP probe

class MLPProbe(nn.Module):
    def __init__(self, dim, hidden=1024, positions=196):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, positions))

    def logits(self, tokens):
        return self.net((tokens.float() - self.mean) / self.std)

    @torch.no_grad()
    def probs(self, tokens):
        return self.logits(tokens).softmax(-1)


def fit_mlp_probe(train, validation, hidden=1024, epochs=30, batch=4096, lr=1e-2, seed=0):
    """train / validation: [images, patches, dim]. Returns (probe at the best validation epoch, history)."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    device = train.device
    patches, dim = train.shape[1], train.shape[2]
    x = train.reshape(-1, dim).float()
    torch.manual_seed(seed)
    probe = MLPProbe(dim, hidden, patches).to(device)
    probe.mean.copy_(x.mean(0))
    probe.std.copy_(x.std(0).clamp_min(1e-6))
    y = torch.arange(patches, device=device).repeat(train.shape[0])
    target = torch.arange(patches, device=device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    best, best_state, history = -1., None, []
    for _ in range(epochs):
        probe.train()
        for chunk in torch.randperm(len(x), generator=generator).to(device).split(batch):
            optimizer.zero_grad()
            F.cross_entropy(probe.logits(x[chunk]), y[chunk]).backward()
            optimizer.step()
        probe.eval()
        accuracy = float((probe.probs(validation).argmax(-1) == target).float().mean())
        history.append(accuracy)
        if accuracy > best:
            best, best_state = accuracy, {key: value.clone() for key, value in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    return probe.eval(), history


# ---------------------------------------------------------------------------------------------------------
# X4: globalisation

def ring_groups(grid=14, rings=4):
    """[positions] ring id 0..rings-1: positions sorted by distance to the centre, split into equal parts."""
    distance = readings.grid_positions(grid).sub((grid - 1) / 2).pow(2).sum(-1).numpy()
    out = np.empty(grid * grid, dtype=int)
    for ring, members in enumerate(np.array_split(np.argsort(distance, kind='stable'), rings)):
        out[members] = ring
    return out


def rank_groups(values, groups=5):
    """Group number 0..groups-1 of each value by rank (equal-size split)."""
    order = np.argsort(values, kind='stable')
    out = np.empty(len(values), dtype=int)
    for group, members in enumerate(np.array_split(order, groups)):
        out[members] = group
    return out


def rank_correlation(a, b):
    """Spearman with average ranks; NaN when either side is constant."""
    import pandas as pd
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denominator) if denominator > 0 else float('nan')


def global_scores(tokens, keep):
    """tokens [patches, dim], keep [patches] bool -> g [patches] (NaN where not kept): cosine with the mean of
    the other kept tokens of the image."""
    x = tokens.float()
    kept = x[keep]
    total = kept.sum(0)
    others = (total[None] - x) / (keep.sum() - keep.float())[:, None].clamp_min(1)
    g = F.cosine_similarity(x, others, dim=-1)
    return torch.where(keep, g, torch.full_like(g, float('nan')))


def globalisation_rows(probs, tokens, high, rings):
    """One image and layer: dict of per-image statistics (X4)."""
    keep = ~high
    g = global_scores(tokens, keep).cpu().numpy()
    own = torch.arange(probs.shape[0], device=probs.device)
    exact = (probs.argmax(-1) == own).float().cpu().numpy()
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).cpu().numpy()
    move = (probs * distances(device=probs.device)).sum(-1).cpu().numpy()
    kept = keep.cpu().numpy()
    quantities = dict(exact=exact, entropy=entropy, move=move)
    group = np.full(len(g), -1)
    group[kept] = rank_groups(g[kept])
    out = dict(n_kept=int(kept.sum()), mean_g=float(np.nanmean(g)))
    for name, value in quantities.items():
        out[f'rho_{name}'] = rank_correlation(group[kept], value[kept])
        within = []
        for ring in range(rings.max() + 1):
            members = kept & (rings == ring)
            if members.sum() >= 5:
                within.append(rank_correlation(rank_groups(g[members]), value[members]))
        out[f'rho_{name}_ring'] = float(np.nanmean(within)) if np.isfinite(within).any() else float('nan')
        for q in range(5):
            out[f'{name}_q{q + 1}'] = float(value[kept & (group == q)].mean())
    return out


# ---------------------------------------------------------------------------------------------------------
# Stage heldout

def collect(model, loader, layers, device):
    """{layer: (tokens [N, patches, dim] float16 CPU, norms [N, patches] float32 CPU)}."""
    prefix = model.num_prefix_tokens
    store, handles, out = {}, [], {layer: ([], []) for layer in layers}
    for layer in layers:
        block = model.blocks[layer - 1]
        handles.append(block.register_forward_pre_hook(
            lambda module, inputs, layer=layer: store.__setitem__((layer, 'norm'),
                                                                  inputs[0][:, prefix:].float().norm(dim=-1))))
        handles.append(block.norm1.register_forward_hook(
            lambda module, inputs, output, layer=layer: store.__setitem__((layer, 'tokens'), output[:, prefix:])))
    try:
        for images in loader:
            with torch.no_grad():
                model(images.to(device))
            for layer in layers:
                out[layer][0].append(store[(layer, 'tokens')].half().cpu())
                out[layer][1].append(store[(layer, 'norm')].cpu())
    finally:
        for handle in handles:
            handle.remove()
    return {layer: (torch.cat(t), torch.cat(n)) for layer, (t, n) in out.items()}


def stage_heldout(args, config, output, model, device):
    settings = config['supp']
    folder = project_path(config['diagnose']['probe_run'].format(model=args.model))
    held = json.loads((folder / 'heldout.json').read_text())
    n_probe = held['probe_images']
    train_rows, test_rows = held['rows'][:n_probe], held['rows'][n_probe:]
    if args.smoke:
        train_rows, test_rows = train_rows[:max(args.smoke - 8, 10)], test_rows[:8]
    n_validation = max(len(train_rows) // 10, 1)
    lens = load_lens(config['correct'], args.model, 'crop', device)
    pre = preprocessing_of(model, args.model)
    files = datasets.imagenet_files()
    rings = ring_groups()
    depth = len(model.blocks)
    mlp_rows, global_rows, probes, sigmas, histories = [], [], {}, {}, {}
    for group in np.array_split(np.arange(1, depth + 1), settings['layer_groups']):
        layers = group.tolist()
        loaders = [DataLoader(HeldoutImages(files, rows, pre), batch_size=config['batch_size'],
                              num_workers=config['num_workers'], shuffle=False) for rows in (train_rows, test_rows)]
        train, test = (collect(model, loader, layers, device) for loader in loaders)
        for layer in layers:
            tokens = train[layer][0]
            probe, history = fit_mlp_probe(tokens[:-n_validation].to(device), tokens[-n_validation:].to(device),
                                           hidden=settings['mlp_hidden'], epochs=config['diagnose']['epochs'],
                                           seed=config['diagnose']['seed'])
            probes[layer] = {key: value.cpu() for key, value in probe.state_dict().items()}  # probe stays on device
            histories[layer] = history
            test_tokens, test_norms = test[layer][0].to(device), test[layer][1].to(device)
            mlp_probs, linear_probs = probe.probs(test_tokens), lens.probes[layer].probs(test_tokens)
            sigmas[layer] = readings.match_sigma(readings.displacement(mlp_probs))
            for name, probs in (('linear', linear_probs), ('mlp', mlp_probs)):
                stats = {key: value.tolist() for key, value in error_stats(probs).items()}
                mlp_rows += [dict(model=args.model, row=row, layer=layer, probe=name, exact=stats['exact'][n],
                                  top5=stats['top5'][n], distance=stats['distance'][n])
                             for n, row in enumerate(test_rows)]
            logging.info('layer %d: linear %.3f, MLP %.3f (best validation epoch %d)', layer,
                         float(error_stats(linear_probs)['exact'].mean()), float(error_stats(mlp_probs)['exact'].mean()),
                         int(np.argmax(history)) + 1)
            if layer in LAYERWISE:
                high = sink_tokens(test_norms, lens.norm_ratio)
                for n, row in enumerate(test_rows):
                    global_rows.append(dict(model=args.model, row=row, layer=layer,
                                            **globalisation_rows(linear_probs[n], test_tokens[n], high[n], rings)))
        del train, test
    write_csv(output / 'mlp_probe.csv', mlp_rows)
    write_csv(output / 'global.csv', global_rows)
    torch.save(dict(probes=probes, hidden=settings['mlp_hidden']), output / 'mlp_probes.pt')
    write_json(output / 'mlp_sigmas.json', dict(sigmas={str(k): v for k, v in sigmas.items()},
                                                histories={str(k): v for k, v in histories.items()}))


def load_mlp(config, model_name, model, device):
    folder = project_path(config['supp']['heldout_run'].format(model=model_name))
    state = torch.load(folder / 'mlp_probes.pt', weights_only=True)
    dim = model.blocks[0].norm1.normalized_shape[0]
    probes = {}
    for layer, weights in state['probes'].items():
        probe = MLPProbe(dim, state['hidden']).to(device)
        probe.load_state_dict(weights)
        probes[int(layer)] = probe.eval()
    sigmas = {int(k): v for k, v in json.loads((folder / 'mlp_sigmas.json').read_text())['sigmas'].items()}
    return probes, sigmas


# ---------------------------------------------------------------------------------------------------------
# Stage eval: all maps from one forward

def shuffled(length, seed, start, layer):
    """Permute each (image, head) row of patch-token weights with a seeded generator."""
    generator = torch.Generator(device=length.device).manual_seed(((seed * 1_000_003 + start) * 101 + layer))
    order = torch.rand(length.shape, generator=generator, device=length.device).argsort(-1)
    return length.gather(-1, order)


def supp_maps(store, lens, mlp, mlp_sigmas, start, grid=14):
    """{name: [B, positions]} for every X1-X5 map, plus 'dose' [B] (share of R0 on high-norm tokens, layers 9-12)."""
    out, deep = {}, {}
    eye = torch.eye(grid * grid, device=store[DEEP[0]]['r0'].device)
    high_mass = 0
    for layer in LAYERWISE:
        piece = store[layer]
        r0, c3, received = piece['r0'], piece['c3'], piece['received']
        high = sink_tokens(piece['norms'], lens.norm_ratio)
        no_high = r0 * ~high
        out[f'R0_l{layer}'], out[f'C3_l{layer}'], out[f'no_high_l{layer}'] = r0, c3, no_high
        if layer not in DEEP:
            continue
        probs = lens.probes[layer].probs(piece['tokens'])
        mlp_probs = mlp[layer].probs(piece['tokens'])
        confident = (probs.max(-1).values >= 0.5).float()[..., None]
        maps = dict(
            R0=r0, C3=c3, no_high=no_high,
            C3v=(received * piece['value_norm']).sum(1),
            C3headmean=(received * piece['length'].mean(-1, keepdim=True)).sum(1),
            C1C3=c3 - lens.means['C3'][layer],
            C2C3=readings.relocate(c3, probs),
            C1C2=readings.relocate(r0, probs) - lens.means['C2'][layer],
            C2conf=readings.relocate(r0, confident * probs + (1 - confident) * eye),
            C2MLP=readings.relocate(r0, mlp_probs),
            blurMLP=readings.relocate(r0, readings.gaussian_relocation(mlp_sigmas[layer], grid, r0.device)),
            C3C2MLP=readings.relocate(c3, mlp_probs),
            **{f'C3shuffle_s{seed}': (received * shuffled(piece['length'], seed, start, layer)).sum(1)
               for seed in SHUFFLE_SEEDS})
        for name, value in maps.items():
            deep[name] = deep.get(name, 0) + value
        high_mass = high_mass + (r0 * high).sum(-1)
    out.update(deep)
    out['C3_centre'] = deep['C3'] - readings.centre_matched(lens.mean('C3', DEEP), grid)
    return out, high_mass / deep['R0'].sum(-1)


def stage_eval(args, config, output, model, device):
    settings = config['supp']
    clickme = settings['dataset'] == 'clickme'
    lens = load_lens(config['correct'], args.model, 'clickme' if clickme else 'crop', device)
    mlp, mlp_sigmas = load_mlp(config, args.model, model, device)
    if clickme:
        pre = IMAGENET_PREPROCESSING[args.model] or resolve_model_data_config(data_reference(model))
        dataset = ClickMe(data_root() / 'clickme', pre['mean'], pre['std'], expected_shards=config['correct']['shards'])
        indices = list(range(len(dataset)))
    else:
        dataset, indices = datasets.build('imagenet_s', model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    correct = config['correct']
    rng = np.random.default_rng(correct['subset_seed'])
    n_faith = 0 if clickme else min(correct['faithfulness_images'], len(indices))
    faithful = set(np.asarray(indices)[rng.choice(len(indices), n_faith, replace=False)].tolist()) if n_faith else set()
    progress_path, done = common.progress(output, args.resume)
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    if clickme:
        sink = Sink(output / 'alignment.csv', ['model', 'index', 'label', 'predicted', 'method'] + MEASURES)
    else:
        sink = Sink(output / 'plausibility.csv', ['model', 'index', 'method'] + PLAUSIBILITY)
        faith = Sink(output / 'faithfulness.csv', ['model', 'index', 'method', 'condition', 'budget', 'agree',
                                                   'p_target', 'kl'])
    dose_sink = Sink(output / 'dose.csv', ['model', 'index', 'dose'])
    head_sum, head_count = 0, 0
    for start, ids, batch in batches(f'supp_{args.model}', dataset, indices):
        images = batch[0].to(device)
        with torch.no_grad(), readings.capture(model, LAYERWISE, per_head=True) as store:
            original = model(images)
            maps, dose = supp_maps(store, lens, mlp, mlp_sigmas, start)
            head_sum = head_sum + torch.stack([store[layer]['length'].mean(-1).sum(0) for layer in DEEP]).cpu()
            head_count += len(ids)
        maps = {name: value.float() for name, value in maps.items()}
        dose_sink.write([dict(model=args.model, index=index, dose=float(dose[n])) for n, index in enumerate(ids)])
        rows = []
        if clickme:
            human = batch[1].to(device)
            predicted = original.argmax(-1).tolist()
            for name, scores in maps.items():
                values = {key: value.tolist() for key, value in alignment(scores, human).items()}
                rows += [dict(model=args.model, index=index, label=int(batch[2][n]), predicted=predicted[n],
                              method=name, **{key: values[key][n] for key in MEASURES}) for n, index in enumerate(ids)]
        else:
            foreground, valid = batch[1].to(device), batch[2].to(device)
            other = foreground.roll(1, 0) if len(ids) > 1 else torch.zeros_like(foreground)
            for name, scores in maps.items():
                values = plausibility.measures(scores, foreground, valid, other,
                                               generator=torch.Generator().manual_seed(start))
                rows += [dict(model=args.model, index=index, method=name,
                              **{key: float(values[key][n]) for key in PLAUSIBILITY}) for n, index in enumerate(ids)]
            chosen = [n for n, index in enumerate(ids) if index in faithful]
            if chosen:
                sub = torch.tensor(chosen, device=device)
                tie = torch.Generator(device=device).manual_seed(correct['tie_seed'] * 1_000_003 + start)
                out = []
                with torch.no_grad():
                    for name in settings['srg_maps']:
                        order = rank(maps[name][sub], tie)
                        for condition in correct['conditions']:
                            for budget, values in curve(model, images[sub], original[sub], order, correct['budgets'],
                                                        condition).items():
                                out += _rows(args.model, [ids[n] for n in chosen], name, condition, budget, values)
                faith.write(out)
        sink.write(rows)
    weights = (head_sum / max(head_count, 1)).numpy()
    write_csv(output / 'head_weights.csv', [dict(model=args.model, layer=layer, head=head, mean_length=float(value))
                                             for layer, row in zip(DEEP, weights) for head, value in enumerate(row)])


def run(args, config, output, commit):
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    write_json(output / 'metadata.json', common.provenance(commit, stage=args.stage, model=args.model,
                                                           decision='B46 (exploratory)', settings=config['supp'],
                                                           smoke_images=args.smoke))
    started = time.perf_counter()
    (stage_heldout if args.stage == 'heldout' else stage_eval)(args, config, output, model, device)
    common.finish(output, started)
    logging.info('Complete: supp %s for %s', args.stage, args.model)
