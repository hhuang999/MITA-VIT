"""Centre-matched deletion on Pet (decision A15): target, random and ring-matched removal, all 12 layers."""

import argparse
import hashlib
import importlib.util
import logging
import time
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root, project_path
ROOT = configure_runtime()

import torch
from torch.utils.data import DataLoader, Subset

from mita_probe.anatomy import Plan, capture_qk, execute, token_mask
from mita_probe.dino_model import HEAD, HEAD_SHA256, load_dino_model
from mita_probe.io import load_config, source_commit, write_json
from mita_probe.mita_model import CHECKPOINT_SHA256, load_mita_model
from mita_probe.models import load_model
from mita_probe.pet import PetSegmentation
from mita_probe.probe import ReadingPattern
from mita_probe.rq3_extension import capture_input_norms, replace_category, sink_tokens
from mita_probe.settings import base_pattern, batch_generator, resolve
from mita_probe.validation import centre_rings, replace_matched_patches, replace_random_patches

spec = importlib.util.spec_from_file_location('run_anatomy', ROOT / 'scripts/run_anatomy.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
CATEGORIES = ['foreground', 'background']
FIELDS = ['model'] + runner.OUTCOME + ['category', 'kind', 'seed', 'eligible', 'mean_removed_per_set']


def deletion(model, kind, qk, norms, images, fg, valid, ids, original, start, config, device, output):
    spec = resolve(runner.MITA_SPEC, dict(scoring='query', summary_scale='mita'))
    scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
    summary = dict(summary_side=5, summary_scale='mita', landmark_mode='mita')
    base, scores = {}, {}
    for layer in range(1, 13):
        pattern, scores[layer] = base_pattern(spec, *qk[layer], scale, grid)
        base[layer] = ReadingPattern(pattern.readsets, pattern.routes)
    plan = lambda patterns: {layer: Plan(token_mask=token_mask(patterns[layer]), **summary) for layer in patterns}
    labels = [-1] * len(ids)
    run = lambda patterns: _execute(model, plan(patterns), images)
    executed = run(base)
    if kind == 'mita':
        error = (executed - original).abs().max().item()
        assert error < 1e-5, error
        runner.update_checks(output, max_pet_frozen_native_error=error)
    rows = runner.outcomes(ids, labels, original, executed, model=kind, category='', kind='baseline', seed=-1,
                           eligible=1, mean_removed_per_set=0.)
    zero = torch.zeros(len(ids), 1, dtype=torch.bool, device=device)
    rings = centre_rings()
    for category in CATEGORIES:
        c = config['pet']['deletion_categories'].index(category)  # same random streams as the V2 stage
        changed, doses, masks = {}, {}, {}
        eligible = torch.ones(len(ids), dtype=torch.bool, device=device)
        for layer in range(1, 13):
            sink = sink_tokens(norms[layer], config['pet']['sink_threshold'])
            mask = fg & valid if category == 'foreground' else ~fg & valid & ~sink
            masks[layer] = torch.cat([zero, mask], -1)
            changed[layer], feasible, doses[layer] = replace_category(base[layer], scores[layer], masks[layer])
            eligible &= feasible
        dose = torch.stack([d.float().mean((1, 2)) for d in doses.values()], -1).mean(-1).tolist()

        def record(output_logits, condition, seed):
            for row, ok, removed in zip(runner.outcomes(ids, labels, original, output_logits, model=kind,
                                                        category=category, kind=condition, seed=seed),
                                        eligible.tolist(), dose):
                rows.append(row | dict(eligible=int(ok), mean_removed_per_set=removed))

        record(run(changed), 'target', -1)
        for seed in config['random_seeds']:
            randomized = {layer: replace_random_patches(base[layer], scores[layer], doses[layer],
                          batch_generator(device, seed * 10 + c, layer, 25, start)) for layer in range(1, 13)}
            record(run(randomized), 'random', seed)
            matched = {}
            for layer in range(1, 13):
                matched[layer], removed = replace_matched_patches(
                    base[layer], scores[layer], masks[layer], rings,
                    batch_generator(device, (seed + 100) * 10 + c, layer, 25, start))
                assert torch.equal(removed.sum(-1), doses[layer])
            record(run(matched), 'centre_matched', seed)
    return rows


def _execute(model, plans, images):
    with execute(model, plans):
        return model(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', type=int, default=0)
    args = parser.parse_args()
    config = load_config(ROOT / 'configs/anatomy_v2.yaml')
    commit = source_commit(ROOT)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.FileHandler(output / 'log.txt'), logging.StreamHandler()])
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    device = runner.DEVICE
    deit = load_model(device)
    mita_path = project_path(config['mita']['checkpoint'])
    assert hashlib.sha256(mita_path.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    head = project_path(HEAD)
    assert hashlib.sha256(head.read_bytes()).hexdigest() == HEAD_SHA256
    models = dict(deit=deit, dino=load_dino_model(head, device), mita=load_mita_model(mita_path, device))
    dataset = PetSegmentation(data_root(), deit)
    assert len(dataset) == config['pet']['images']
    indices = list(range(args.smoke or len(dataset)))
    write_json(output / 'metadata.json', dict(git_commit=commit, decision='A15', images=len(indices),
                                              device=torch.cuda.get_device_name(), torch=torch.__version__,
                                              seeds=config['random_seeds'], categories=CATEGORIES))
    sink = runner.Sink(output / 'deletion.csv', FIELDS)
    started = time.perf_counter()
    for kind, model in models.items():
        start = 0
        with torch.inference_mode():
            for images, fg, valid, ids in DataLoader(Subset(dataset, indices), batch_size=config['batch_size'],
                                                     num_workers=config['num_workers'], shuffle=False):
                images, fg, valid = images.to(device), fg.to(device), valid.to(device)
                with capture_qk(model) as qk, capture_input_norms(model) as norms:
                    original = model(images)
                sink.write(deletion(model, kind, qk, norms, images, fg, valid, ids.tolist(), original, start,
                                    config, device, output))
                start += len(ids)
                if start % 640 < len(ids) or start == len(indices):
                    logging.info('%s %d/%d images; %.1fs', kind, start, len(indices), time.perf_counter() - started)
    write_json(output / 'checks.json', dict(complete=True, elapsed_seconds=time.perf_counter() - started))


if __name__ == '__main__':
    main()
