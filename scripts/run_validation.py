"""Run approved V01, V02 or V03 on the fixed data; preserve original experiments."""

import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root, project_path
ROOT = configure_runtime()

import numpy as np
import timm
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mita_probe.assets import MODEL_REVISION
from mita_probe.data import ImageNetValidation, experiment_indices
from mita_probe.io import load_config, source_commit, write_json
from mita_probe.metrics import kl_divergence
from mita_probe.models import load_model, read_attention, execute_attention
from mita_probe.pet import PetSegmentation
from mita_probe.rq3_extension import capture_input_norms, sink_tokens, replace_category
from mita_probe.validation import execute_online, layer_settings, replace_random_patches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['v01', 'v02', 'v03'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', type=int, default=0)
    args = parser.parse_args()
    config = load_config(ROOT / 'configs/overnight_validation.yaml')
    commit = source_commit(ROOT)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    config.update(stage=args.stage, smoke_images=args.smoke, preregistration_commit='d956f42')
    (output / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.FileHandler(output / 'log.txt'), logging.StreamHandler()])
    try:
        run(args, config, output, commit)
    except Exception:
        logging.exception('Validation failed; preserve this directory')
        raise


def run(args, config, output, commit):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available(), 'Formal inference requires authorized remote GPU'
    model = load_model('cuda')
    if args.stage == 'v01':
        dataset = PetSegmentation(data_root(), model)
        assert len(dataset) == config['pet_images']
        indices = list(range(len(dataset)))
    else:
        dataset = ImageNetValidation(data_root() / 'imagenet_val', model)
        indices = experiment_indices(len(dataset), 5000, 0, project_path(config['indices']))
    indices = indices[:args.smoke] if args.smoke else indices
    write_json(output / 'indices.json', indices)
    write_json(output / 'metadata.json', dict(git_commit=commit, stage=args.stage,
        timestamp=datetime.now().astimezone().isoformat(), torch=torch.__version__,
        timm=timm.__version__, device=torch.cuda.get_device_name(), dtype='float32, no autocast',
        checkpoint_revision=MODEL_REVISION, preprocessing=dataset.preprocessing,
        n_images=len(indices), batch_size=32, preregistration_commit='d956f42'))
    loader = DataLoader(Subset(dataset, indices), batch_size=32, num_workers=0, shuffle=False)
    fields = ['index', 'target', 'k', 'setting', 'scope', 'category', 'seed', 'original',
              'executed', 'agree', 'kl', 'eligible', 'mean_removed_per_set', 'reason']
    generators = {(k, scope, category, seed): torch.Generator(device='cuda').manual_seed(seed)
                  for k in config['pet_k'] for scope in config['scopes']
                  for category in config['categories'] for seed in config['random_seeds']}
    checks = dict(n_images=len(indices), max_observe_error=0., max_identity_error=0.,
                  identity_agreement=1., random_sets_checked=0, max_single_layer_error=0.,
                  equivalence_images=0)
    started, offset, identities, equivalence = time.perf_counter(), 0, [], []
    with (output / 'predictions.csv').open('x', newline='') as stream, \
            (output / 'patterns.csv').open('x', newline='') as pattern_stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        pattern_writer = csv.DictWriter(pattern_stream,
            fieldnames=['index', 'k', 'layer', 'readset_overlap', 'route_changed'])
        pattern_writer.writeheader()
        with torch.inference_mode():
            for batch in loader:
                images = batch[0].cuda()
                ids = indices[offset:offset + len(images)]
                labels = [-1]*len(images) if args.stage == 'v01' else batch[1].tolist()
                original = model(images)
                with read_attention(model, k_top=197) as full:
                    observed = model(images)
                with (execute_online(model, 197) if args.stage == 'v03' else execute_attention(model, full)):
                    identity = model(images)
                checks['max_observe_error'] = max(checks['max_observe_error'], (observed-original).abs().max().item())
                checks['max_identity_error'] = max(checks['max_identity_error'], (identity-original).abs().max().item())
                assert torch.equal(observed, original)
                assert torch.equal(identity.argmax(-1), original.argmax(-1))
                assert checks['max_identity_error'] < 1e-5
                for i, index in enumerate(ids):
                    identities.append(dict(index=index, target=labels[i], original=int(original[i].argmax()),
                                           identity=int(identity[i].argmax()), agree=1))

                def record(executed, k, setting, scope='', category='', seed=-1, eligible=None, dose=None):
                    assert torch.isfinite(executed).all()
                    predictions = executed.argmax(-1).tolist()
                    originals = original.argmax(-1).tolist()
                    kl = kl_divergence(original.double(), executed.double()).tolist()
                    eligible = [True]*len(ids) if eligible is None else eligible.tolist()
                    dose = [0.]*len(ids) if dose is None else dose.tolist()
                    writer.writerows(dict(index=index, target=labels[i], k=k, setting=setting,
                        scope=scope, category=category, seed=seed, original=originals[i],
                        executed=predictions[i] if eligible[i] else -1,
                        agree=int(predictions[i] == originals[i]) if eligible[i] else np.nan,
                        kl=kl[i] if eligible[i] else np.nan, eligible=int(eligible[i]),
                        mean_removed_per_set=dose[i],
                        reason='' if eligible[i] else 'fewer_than_k_non_target_tokens')
                        for i, index in enumerate(ids))

                if args.stage == 'v01':
                    fg, valid = batch[1].cuda(), batch[2].cuda()
                    zero = torch.zeros(len(images), 1, dtype=torch.bool, device='cuda')
                    for k in config['pet_k']:
                        scores = {}
                        with read_attention(model, k_top=k, region_scores=scores) as patterns, capture_input_norms(model) as norms:
                            assert torch.equal(model(images), original)
                        for scope in config['scopes']:
                            layers = range(9, 13) if scope == 'last4' else range(1, 13)
                            base = {layer: patterns[layer] for layer in layers}
                            with execute_attention(model, base):
                                record(model(images), k, 'baseline', scope)
                            for category in config['categories']:
                                changed, doses = {}, {}
                                eligible = torch.ones(len(images), dtype=torch.bool, device='cuda')
                                for layer, pattern in base.items():
                                    sink = sink_tokens(norms[layer], config['sink_threshold'])
                                    mask = sink if category == 'sink' else (fg & valid if category == 'foreground' else ~fg & valid & ~sink)
                                    forbidden = torch.cat([zero, mask], -1)
                                    changed[layer], feasible, doses[layer] = replace_category(pattern, scores[layer], forbidden)
                                    eligible &= feasible
                                dose = torch.stack([d.float().mean((1, 2)) for d in doses.values()], -1).mean(-1)
                                with execute_attention(model, changed):
                                    record(model(images), k, category, scope, category, eligible=eligible, dose=dose)
                                for seed in config['random_seeds']:
                                    random_patterns = {}
                                    for layer, pattern in base.items():
                                        new = replace_random_patches(pattern, scores[layer], doses[layer],
                                                generators[k, scope, category, seed])
                                        old_mask = torch.zeros_like(scores[layer], dtype=torch.bool).scatter_(-1, pattern.readsets, True)
                                        new_mask = torch.zeros_like(old_mask).scatter_(-1, new.readsets, True)
                                        assert torch.equal((old_mask & ~new_mask).sum(-1), doses[layer])
                                        assert (new_mask.sum(-1) == k).all()
                                        assert (~old_mask[..., 0] | new_mask[..., 0]).all()
                                        assert new.routes is pattern.routes
                                        checks['random_sets_checked'] += doses[layer].numel()
                                        random_patterns[layer] = new
                                    with execute_attention(model, random_patterns):
                                        # Keep all random predictions; eligibility of the target is paired in analysis.
                                        record(model(images), k, 'random_patch', scope, category, seed, dose=dose)
                else:
                    for k in config['k']:
                        with read_attention(model, k_top=k) as frozen:
                            assert torch.equal(model(images), original)
                        if args.stage == 'v02':
                            for setting, layers in layer_settings():
                                with execute_attention(model, {layer: frozen[layer] for layer in layers}):
                                    record(model(images), k, setting)
                        else:
                            n_eq = max(0, min(len(images), config['equivalence_images']-offset))
                            if n_eq:
                                eq_images = images[:n_eq]
                                with read_attention(model, k_top=k) as eq_frozen:
                                    model(eq_images)
                                for layer in range(1, 13):
                                    with execute_attention(model, {layer: eq_frozen[layer]}):
                                        expected = model(eq_images)
                                    with execute_online(model, k, layers=(layer,)) as eq_online:
                                        actual = model(eq_images)
                                    error = (actual-expected).abs().max().item()
                                    assert error == 0
                                    assert torch.equal(eq_online[layer].readsets, eq_frozen[layer].readsets)
                                    assert torch.equal(eq_online[layer].routes, eq_frozen[layer].routes)
                                    equivalence.append(dict(k=k, layer=layer, n=n_eq, max_error=error))
                                checks['max_single_layer_error'] = max(checks['max_single_layer_error'], error)
                            with execute_online(model, k) as online:
                                record(model(images), k, 'online_all12')
                            for layer in range(1, 13):
                                old, new = frozen[layer], online[layer]
                                old_mask = torch.zeros(len(images), 6, 16, 197, dtype=torch.bool, device='cuda').scatter_(-1, old.readsets, True)
                                overlap = old_mask.gather(-1, new.readsets).float().mean((1, 2, 3)).tolist()
                                changed = (old.routes != new.routes).float().mean((1, 2)).tolist()
                                pattern_writer.writerows(dict(index=index, k=k, layer=layer,
                                    readset_overlap=overlap[i], route_changed=changed[i]) for i, index in enumerate(ids))
                offset += len(images)
                stream.flush()
                pattern_stream.flush()
                if offset % 320 == 0 or offset == len(indices):
                    logging.info('%s %d/%d images; %.1fs', args.stage, offset, len(indices), time.perf_counter()-started)
    import pandas as pd
    pd.DataFrame(identities).to_csv(output / 'identity_predictions.csv', index=False)
    if equivalence:
        pd.DataFrame(equivalence).to_csv(output / 'equivalence.csv', index=False)
        checks['equivalence_images'] = min(config['equivalence_images'], len(indices))
    checks.update(settings_completed={'v01': 76, 'v02': 57, 'v03': 3}[args.stage],
                  elapsed_seconds=time.perf_counter()-started)
    write_json(output / 'checks.json', checks)
    logging.info('Complete: %s', checks)


if __name__ == '__main__':
    main()
