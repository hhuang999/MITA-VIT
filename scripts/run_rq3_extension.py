"""Collect preregistered RQ3 observations or frozen-summary causal predictions."""

import argparse
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root
ROOT = configure_runtime()

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from timm.data import resolve_model_data_config
from torch.utils.data import DataLoader, Subset

from mita_probe.assets import MODEL_REVISION
from mita_probe.io import load_config, source_commit, write_json
from mita_probe.models import load_model, read_attention, execute_attention, capture_cls_attention
from mita_probe.pet import PetSegmentation
from mita_probe.probe import ReadingPattern
from mita_probe.rq3_extension import (
    capture_input_norms, sink_tokens, token_counts, replace_category, replace_random,
)


class DiagnosticPet(PetSegmentation):
    def __getitem__(self, index):
        tensor, _, mask, fg, valid = self.aligned(index)
        xy = mask.eq(1).nonzero().double()
        distance = float((xy.mean(0) - 111.5).norm() / 224) if len(xy) else float('nan')
        return tensor, fg, valid, index, distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['observe', 'causal'], default='observe')
    parser.add_argument('--model', choices=['deit', 'dino', 'register'], default='deit')
    parser.add_argument('--smoke', type=int, default=0)
    args = parser.parse_args()
    config = load_config(ROOT / 'configs/rq3_extension.yaml')
    config.update(mode=args.mode, model_alias=args.model, smoke_images=args.smoke)
    commit = source_commit(ROOT)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'raw').mkdir()
    (output / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.FileHandler(output / 'log.txt', encoding='utf-8'), logging.StreamHandler()])
    try:
        run(args, config, output, commit)
    except Exception:
        logging.exception('Extension run failed')
        raise


def run(args, config, output, commit):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    reference = load_model('cpu')
    preprocessing = resolve_model_data_config(reference)
    if args.model == 'deit':
        model, revision = reference.to('cuda'), MODEL_REVISION
    else:
        name = ('vit_small_patch14_reg4_dinov2.lvd142m' if args.model == 'register'
                else 'vit_small_patch14_dinov2.lvd142m')
        revision = config[f'{args.model}_revision']
        model = timm.create_model(name, pretrained=True, img_size=224,
            pretrained_cfg_overlay={'hf_hub_id': f'timm/{name}@{revision}'}).eval().to('cuda')
        actual = resolve_model_data_config(model)
        assert tuple(actual['mean']) == tuple(preprocessing['mean'])
        assert tuple(actual['std']) == tuple(preprocessing['std'])
    dataset = DiagnosticPet(data_root(), model, patch_size=model.patch_embed.patch_size[0],
                            preprocessing=preprocessing)
    n = args.smoke or len(dataset)
    assert args.smoke or n == 3669
    loader = DataLoader(Subset(dataset, range(n)), batch_size=32, num_workers=0)
    prefix = model.num_prefix_tokens
    patches = model.patch_embed.num_patches
    examples = np.random.default_rng(0).choice(3669, 5, replace=False).tolist()
    write_json(output / 'metadata.json', dict(git_commit=commit, checkpoint_revision=revision,
        timestamp=datetime.now().astimezone().isoformat(), torch=torch.__version__, timm=timm.__version__,
        device=torch.cuda.get_device_name(), n_images=n, num_prefix_tokens=prefix,
        grid_size=model.patch_embed.grid_size, preprocessing=preprocessing,
        example_indices=examples, preregistration_commit=config['preregistration_commit']))
    arrays = {key: [] for key in ['norms', 'counts', 'cls', 'foreground', 'valid', 'distance', 'original']}
    rows = []
    generators = {(k, scope, seed): torch.Generator(device='cuda').manual_seed(seed)
                  for k in [16, 32] for scope in ['last4', 'all12'] for seed in range(5)}
    started = time.perf_counter()
    max_error, identity_error = 0., 0.
    with torch.inference_mode():
        for images, fg, valid, indices, distances in loader:
            images, fg, valid = images.cuda(), fg.cuda(), valid.cuda()
            original = model(images)
            scores = {}
            with read_attention(model, k_top=16, region_scores=scores) as patterns16, \
                    capture_input_norms(model) as norms, capture_cls_attention(model) as cls:
                observed = model(images)
            error = (original-observed).abs().max().item()
            max_error = max(error, max_error)
            assert error == 0 and torch.isfinite(observed).all()
            patterns = {16: patterns16, 32: {layer: ReadingPattern(
                scores[layer].topk(32, -1).indices, pattern.routes)
                for layer, pattern in patterns16.items()}}
            if args.mode == 'observe':
                counts = torch.stack([torch.stack([token_counts(patterns[k][layer].readsets,
                    patches+prefix) for layer in range(1, 13)], 1) for k in [16, 32]], 1)
                arrays['norms'].append(torch.stack(list(norms.values()), 1).cpu().numpy())
                arrays['counts'].append(counts.cpu().numpy().astype(np.uint16))
                arrays['cls'].append(cls['heatmap'].cpu().numpy())
                arrays['foreground'].append(fg.cpu().numpy())
                arrays['valid'].append(valid.cpu().numpy())
                arrays['distance'].append(distances.numpy())
                arrays['original'].append(original.argmax(-1).cpu().numpy())
                for index in indices.tolist():
                    if index in examples:
                        dataset.aligned(index)[1].save(output / f'example_{index}.png')
            else:
                assert args.model == 'deit'
                if int(indices[0]) == 0:
                    identity = {layer: ReadingPattern(torch.arange(patches+prefix, device='cuda')
                        .expand(len(images), 6, 16, -1), pattern.routes)
                        for layer, pattern in patterns16.items()}
                    with execute_attention(model, identity):
                        identity_error = (model(images)-original).abs().max().item()
                    assert identity_error == 0
                zero = torch.zeros(len(images), 1, device='cuda', dtype=torch.bool)
                for k in [16, 32]:
                    for scope in ['last4', 'all12']:
                        layers = list(range(9, 13) if scope == 'last4' else range(1, 13))
                        base = {layer: patterns[k][layer] for layer in layers}
                        for kind, seed in [('baseline', -1), ('sink', -1), ('foreground', -1),
                                           ('background', -1)] + [('random', s) for s in range(5)]:
                            changed, eligible, doses = {}, torch.ones(len(images), dtype=torch.bool, device='cuda'), []
                            for layer, pattern in base.items():
                                sink = sink_tokens(norms[layer])
                                category = (sink if kind in ['sink', 'random'] else
                                            (fg & valid if kind == 'foreground' else ~fg & valid & ~sink))
                                forbidden = torch.cat([zero, category], -1)
                                replaced, feasible, dose = replace_category(pattern, scores[layer], forbidden)
                                if kind == 'baseline':
                                    changed[layer], dose = pattern, torch.zeros_like(dose)
                                elif kind == 'random':
                                    changed[layer] = replace_random(pattern, scores[layer], dose,
                                                                    generators[k, scope, seed])
                                else:
                                    changed[layer] = replaced
                                    eligible &= feasible
                                doses.append(dose.float().mean((1, 2)))
                            with execute_attention(model, changed):
                                executed = model(images)
                            assert torch.isfinite(executed).all()
                            prediction = executed.argmax(-1)
                            dose = torch.stack(doses, -1).mean(-1)
                            for j, index in enumerate(indices.tolist()):
                                ok = bool(eligible[j])
                                rows.append(dict(index=index, k=k, scope=scope, method=kind, seed=seed,
                                    original=int(original[j].argmax()), executed=int(prediction[j]) if ok else -1,
                                    agree=float(prediction[j] == original[j].argmax()) if ok else np.nan,
                                    eligible=int(ok), mean_removed_per_set=float(dose[j]),
                                    reason='' if ok else 'fewer_than_k_non_target_tokens'))
            if (int(indices[-1])+1) % 320 == 0 or int(indices[-1]) == n-1:
                logging.info('%s %s %d/%d, %.1fs', args.model, args.mode, int(indices[-1])+1,
                             n, time.perf_counter()-started)
                if rows:
                    pd.DataFrame(rows).to_csv(output / 'predictions.csv', index=False)
    if args.mode == 'observe':
        np.savez_compressed(output / 'raw/observations.npz', **{
            key: np.concatenate(value) for key, value in arrays.items()})
        digest = hashlib.sha256((output / 'raw/observations.npz').read_bytes()).hexdigest()
    else:
        pd.DataFrame(rows).to_csv(output / 'predictions.csv', index=False)
        digest = hashlib.sha256((output / 'predictions.csv').read_bytes()).hexdigest()
    write_json(output / 'checks.json', dict(n_images=n, max_observe_error=max_error,
        identity_error=identity_error, output_sha256=digest, elapsed_seconds=time.perf_counter()-started))
    logging.info('Complete %.1fs; observe error %g', time.perf_counter()-started, max_error)


if __name__ == '__main__':
    main()
