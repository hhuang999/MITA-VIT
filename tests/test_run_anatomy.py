"""Exercise every V2 stage end to end on CPU with random weights and fake data."""

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import timm
import torch
from torch.utils.data import Dataset, TensorDataset

from mita_probe.io import load_config
from mita_probe.mita_model import install, native_plan
from mita_probe.models import load_model
from mita_probe.settings import expand_rq2

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_anatomy', ROOT / 'scripts/run_anatomy.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakePet(Dataset):
    def __len__(self):
        return 2400

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        foreground = torch.rand(196, generator=generator) < .5
        valid = torch.rand(196, generator=generator) < .9
        return torch.randn(3, 224, 224, generator=generator), foreground & valid, valid, index


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    runner.DEVICE = 'cpu'
    config = load_config(ROOT / 'configs/anatomy_v2.yaml')
    config.update(batch_size=2, num_workers=0, random_seeds=[0, 1])
    deit = load_model(pretrained=False)
    mita = timm.create_model('deit_small_patch16_224', pretrained=False).eval()
    for block in mita.blocks:
        install(block.attn, native_plan(), mita.patch_embed.grid_size, 1)
    images = TensorDataset(torch.randn(4, 3, 224, 224), torch.randint(0, 1000, (4,)))
    data = SimpleNamespace(imagenet=lambda kind, n=None, shift=0: (images, [2, 0, 3, 1]),
                           pet=lambda: (FakePet(), [2335, 0, 7, 989]))
    return config, deit, mita, data, tmp_path_factory


def run_stage(setup, name, smoke=4):
    config, deit, mita, data, factory = setup
    output = factory.mktemp(name)
    batches = runner.make_batches(config, {}, output / 'progress.json', time.perf_counter())
    with torch.inference_mode():
        runner.STAGE_FUNCTIONS[name](SimpleNamespace(smoke=smoke), config, output, deit, mita, data, batches)
    return output


def test_rq1_single_layer_anatomy(setup):
    output = run_stage(setup, 'rq1')
    rows = pd.read_csv(output / 'predictions.csv')
    assert len(rows) == 4 * 12 * 11 and set(rows.layer) == set(range(1, 13))
    cover = pd.read_csv(output / 'coverage.csv')
    assert len(cover) == 4 * 12
    assert (cover['L1_k128_patch'] >= cover['L1_k4_patch']).all()
    assert cover.filter(like='_patch').le(1 + 1e-6).all().all()


def test_rq2_all_layer_frontier(setup):
    config = setup[0]
    output = run_stage(setup, 'rq2')
    specs = pd.read_csv(output / 'settings.csv')
    assert len(specs) == len(expand_rq2(config))
    assert len(pd.read_csv(output / 'predictions.csv')) == 4 * len(specs)


def test_noise_and_mita_stages(setup):
    noise = pd.read_csv(run_stage(setup, 'noise') / 'predictions.csv')
    assert len(noise) == 8 and (noise.agree == 1).all()  # fake shift is the same image
    repro = pd.read_csv(run_stage(setup, 'mita_repro') / 'predictions.csv')
    assert len(repro) == 4
    anatomy = pd.read_csv(run_stage(setup, 'mita_anatomy') / 'predictions.csv')
    assert (anatomy.query("setting == 'L4_frozen'").agree == 1).all()
    assert len(anatomy) == 4 * (4 + 2 * 2)


def test_pet_composition_examples_and_deletion(setup):
    output = run_stage(setup, 'pet')
    comp = pd.read_csv(output / 'composition.csv')
    assert len(comp) == 2 * 4 * 12 * 3
    totals = comp[['cls', 'sink', 'foreground', 'background', 'unknown']].sum(axis=1)
    assert (totals - 1).abs().max() < 1e-5
    examples = pd.read_csv(output / 'examples.csv', dtype={'tokens': str})
    assert set(examples['index']) == {2335, 989} and examples.tokens.str.len().eq(196).all()
    deletion = pd.read_csv(output / 'deletion.csv')
    assert len(deletion) == 2 * 4 * (1 + 3 * 3)
    assert (deletion.query("model == 'mita' and kind == 'baseline'").agree == 1).all()


def test_batches_resume_skips_completed(setup, tmp_path):
    config = setup[0]
    data = TensorDataset(torch.arange(6).float(), torch.arange(6))
    batches = runner.make_batches(config, {'x': 2}, tmp_path / 'progress.json', time.perf_counter())
    starts = [start for start, _, _ in batches('x', data, list(range(6)))]
    assert starts == [2, 4]
