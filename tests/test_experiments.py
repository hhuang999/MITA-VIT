"""Every experiment end to end on CPU with random weights and fake data."""

import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import timm
import torch
from torch.utils.data import Dataset, TensorDataset

from mita_probe.experiments import anatomy as runner
from mita_probe.experiments import common
from mita_probe.experiments.center_control import centre_weights
from mita_probe.experiments.matched_deletion import deletion
from mita_probe.io import load_config
from mita_probe.models.deit import load_deit
from mita_probe.models.dino import load_dino_model
from mita_probe.models.mita import install, native_plan
from mita_probe.probe import capture_input_norms, capture_qk, expand_rq2

ROOT = Path(__file__).resolve().parents[1]


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
    common.DEVICE = 'cpu'
    config = load_config(ROOT / 'configs/anatomy_deit.yaml')
    config.update(batch_size=2, num_workers=0, random_seeds=[0, 1])
    deit = load_deit(pretrained=False)
    mita = timm.create_model('deit_small_patch16_224', pretrained=False).eval()
    for block in mita.blocks:
        install(block.attn, native_plan(), mita.patch_embed.grid_size, 1)
    images = TensorDataset(torch.randn(4, 3, 224, 224), torch.randint(0, 1000, (4,)))
    data = SimpleNamespace(imagenet=lambda kind, n=None, shift=0: (images, [2, 0, 3, 1]),
                           pet=lambda: (FakePet(), [2335, 0, 7, 989]))
    return config, deit, mita, data, tmp_path_factory


@pytest.fixture(scope='module')
def dino_setup(setup):
    config = load_config(ROOT / 'configs/anatomy_dino.yaml')
    config.update(batch_size=2, num_workers=0, random_seeds=[0, 1])
    torch.manual_seed(0)
    dino = load_dino_model(None, pretrained=False)
    return config, dino, None, setup[3], setup[4]


def run_stage(setup, name, smoke=4, model='deit'):
    config, deit, mita, data, factory = setup
    output = factory.mktemp(f'{model}_{name}')
    batches = runner.make_batches(config, {}, output / 'progress.json', time.perf_counter())
    args = SimpleNamespace(smoke=smoke, model=model)
    with torch.inference_mode():
        runner.STAGE_FUNCTIONS[name](args, config, output, deit, mita, data, batches)
    return output


def test_rq1_single_layer_anatomy(setup):
    output = run_stage(setup, 'rq1')
    rows = pd.read_csv(output / 'predictions.csv')
    assert len(rows) == 4 * (12 * 11 + 2) and set(rows.layer) == set(range(1, 13))
    assert set(rows.query('layer == 12').setting) >= {'L3_k8_v1_cls_routed', 'L3_k8_cls_own'}
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
    union = pd.read_csv(output / 'union_composition.csv')
    assert len(union) == len(comp) and union.union_size.between(1, 196).all()
    parts = union[['sink', 'foreground', 'background', 'unknown']].sum(axis=1)
    assert (parts - 1).abs().max() < 1e-5


def test_dino_stages_without_mita(dino_setup):
    config = dino_setup[0]
    noise = pd.read_csv(run_stage(dino_setup, 'noise', model='dino') / 'predictions.csv')
    assert set(noise.model) == {'dino'} and (noise.agree == 1).all()
    rq2 = run_stage(dino_setup, 'rq2', model='dino')
    assert len(pd.read_csv(rq2 / 'settings.csv')) == len(expand_rq2(config)) == 7 + 2 * 2 + 1
    repro = pd.read_csv(run_stage(dino_setup, 'repro', model='dino') / 'predictions.csv')
    assert len(repro) == 4
    pet = run_stage(dino_setup, 'pet', model='dino')
    assert set(pd.read_csv(pet / 'composition.csv').model) == {'dino'}
    assert len(pd.read_csv(pet / 'deletion.csv')) == 4 * (1 + 3 * 3)
    assert len(pd.read_csv(pet / 'union_composition.csv')) == 4 * 12 * 3


def test_dino_rq1_single_layer_anatomy(dino_setup):
    rows = pd.read_csv(run_stage(dino_setup, 'rq1', model='dino') / 'predictions.csv')
    assert len(rows) == 4 * (12 * 11 + 2)


def test_batches_resume_skips_completed(setup, tmp_path):
    config = setup[0]
    data = TensorDataset(torch.arange(6).float(), torch.arange(6))
    batches = runner.make_batches(config, {'x': 2}, tmp_path / 'progress.json', time.perf_counter())
    starts = [start for start, _, _ in batches('x', data, list(range(6)))]
    assert starts == [2, 4]


def test_matched_deletion_rows_and_doses(setup, tmp_path):
    config = setup[0]
    deit = setup[1]
    images = torch.randn(3, 3, 224, 224)
    valid = torch.rand(3, 196) < .9
    fg = (torch.rand(3, 196) < .5) & valid
    with torch.inference_mode(), capture_qk(deit) as qk, capture_input_norms(deit) as norms:
        original = deit(images)
    with torch.inference_mode():
        rows = pd.DataFrame(deletion(deit, 'deit', qk, norms, images, fg, valid, [5, 6, 7], original, 0, config,
                                     'cpu', tmp_path))
    assert len(rows) == 3 * (1 + 2 * (1 + 2 + 2))
    assert set(rows.kind) == {'baseline', 'target', 'random', 'centre_matched'}
    per = rows.query("kind != 'baseline'").groupby(['index', 'category'])['mean_removed_per_set'].nunique()
    assert (per == 1).all()


def test_centre_weights_take_whole_rings_then_share_the_next():
    weights = centre_weights()
    assert weights.shape == (197, 196)
    assert (abs(weights.sum(1) - range(197)) < 1e-9).all() and (weights <= 1).all()
    assert set(weights[4].nonzero()[0]) == {90, 91, 104, 105}
    assert sorted(set(weights[5][weights[5] > 0].round(3))) == [0.125, 1.0]


@pytest.fixture(scope='module')
def deit_b_setup(setup):
    config = load_config(ROOT / 'configs/anatomy_deit_b.yaml')
    config.update(batch_size=2, num_workers=0, random_seeds=[0, 1])
    torch.manual_seed(0)
    return config, load_deit(pretrained=False, base=True), None, setup[3], setup[4]


def test_deit_b_rq1_rq2_and_maps(deit_b_setup):
    config, model = deit_b_setup[:2]
    assert model.embed_dim == 768 and model.blocks[0].attn.num_heads == 12
    rq1 = pd.read_csv(run_stage(deit_b_setup, 'rq1', smoke=2, model='deit_b') / 'predictions.csv')
    assert len(rq1) == 4 * (12 * 11 + 2)  # the fake ImageNet always yields its 4 images
    rq2 = pd.read_csv(run_stage(deit_b_setup, 'rq2', smoke=2, model='deit_b') / 'settings.csv')
    assert len(rq2) == len(expand_rq2(config))
    from mita_probe import explain
    from mita_probe.probe import resolve
    images = torch.randn(2, 3, 224, 224)
    spec = resolve({'mode': 'L3'}, config['main'])
    for scores in (explain.probe(model, images, spec, [9, 10, 11, 12]), explain.gae(model, images),
                   explain.rollout(model, images)):
        assert scores.shape == (2, 196) and torch.isfinite(scores).all()
