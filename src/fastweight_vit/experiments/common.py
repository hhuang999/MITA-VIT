"""Shared plumbing for every experiment: output folder, provenance, resumable batches, result rows."""

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import timm
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..io import write_json
from ..metrics import kl_divergence
from ..execution import Plan, capture_qk, execute

DEVICE = 'cuda'  # tests set 'cpu' to exercise experiments without a GPU
OUTCOME = ['index', 'target', 'original', 'executed', 'agree', 'kl']


def prepare_output(output: Path, config: dict, *, resume: bool, **labels) -> None:
    """Create the run folder, save the effective config and log to log.txt as well as stderr."""
    output.mkdir(parents=True, exist_ok=resume)
    if not resume:
        (output / 'config.yaml').write_text(yaml.safe_dump(config | labels), encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True, handlers=[
        logging.FileHandler(output / 'log.txt'), logging.StreamHandler()])


def deterministic(threads=2):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def provenance(commit, **fields):
    return dict(git_commit=commit, timestamp=datetime.now().astimezone().isoformat(), torch=torch.__version__,
                timm=timm.__version__, device=torch.cuda.get_device_name() if DEVICE == 'cuda' else 'cpu',
                dtype='float32, no autocast', **fields)


class Sink:
    """Append-only CSV written one whole batch at a time."""

    def __init__(self, path, fields):
        fresh = not path.exists()
        self.stream = path.open('a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        if fresh:
            self.writer.writeheader()

    def write(self, rows):
        self.writer.writerows(rows)
        self.stream.flush()


def outcomes(ids, labels, original, executed, **fields):
    """One row per image: original vs executed top-1, agreement and KL(original || executed)."""
    assert torch.isfinite(executed).all()
    kl = kl_divergence(original.double(), executed.double()).tolist()
    predicted, reference = executed.argmax(-1).tolist(), original.argmax(-1).tolist()
    return [dict(index=index, target=labels[n], original=reference[n], executed=predicted[n],
                 agree=int(predicted[n] == reference[n]), kl=kl[n], **fields)
            for n, index in enumerate(ids)]


def full_plans(n_layers, plan):
    return {layer: plan(layer) for layer in range(1, n_layers + 1)}


def progress(output, resume):
    path = output / 'progress.json'
    return path, (json.loads(path.read_text())['done'] if resume and path.exists() else {})


def make_batches(config, done, progress_path, started):
    def batches(name, dataset, indices):
        """Yield (start, ids, batch) for batches not yet completed; record completion after use."""
        start = 0
        for batch in DataLoader(Subset(dataset, indices), batch_size=config['batch_size'],
                                num_workers=config['num_workers'], shuffle=False):
            ids = indices[start:start + len(batch[0])]
            if start >= done.get(name, 0):
                yield start, ids, batch
                done[name] = start + len(ids)
                write_json(progress_path, dict(done=done))
                if done[name] % 640 < len(ids) or done[name] == len(indices):
                    logging.info('%s %d/%d images; %.1fs', name, done[name], len(indices),
                                 time.perf_counter() - started)
            start += len(ids)
    return batches


def update_checks(output, **values):
    path = output / 'running_checks.json'
    checks = json.loads(path.read_text()) if path.exists() else {}
    for key, value in values.items():
        checks[key] = max(checks.get(key, value), value) if key.startswith('max_') else value
    write_json(path, checks)


def finish(output, started):
    running = output / 'running_checks.json'
    checks = json.loads(running.read_text()) if running.exists() else {}
    write_json(output / 'checks.json', dict(complete=True, elapsed_seconds=time.perf_counter() - started, **checks))


def dense_identity(model, images, original):
    """Captured Q/K leave logits unchanged; dense attention over all keys reproduces them.

    Tolerance 5e-4 on the logits (was 1e-4; B42): DeiT-B on an L4 differed by 1.4e-4 in float32 between the fused
    and the explicit softmax. The predicted classes must still be identical; the largest error is recorded.
    """
    with capture_qk(model) as qk:
        observed = model(images)
    assert torch.equal(observed, original)
    q = qk[1][0]
    full = torch.ones(*q.shape[:3], q.shape[2], dtype=torch.bool, device=q.device)
    with execute(model, full_plans(len(model.blocks), lambda layer: Plan(token_mask=full))):
        dense = model(images)
    error = (dense - original).abs().max().item()
    assert error < 5e-4 and torch.equal(dense.argmax(-1), original.argmax(-1)), error
    return qk, error
