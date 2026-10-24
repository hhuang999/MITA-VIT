from pathlib import Path

import numpy as np
import torch

from mita_probe.data import IMAGENET_IDS, Imagenette1k, subset_indices
from mita_probe.metrics import agreement, kl_divergence, paired_bootstrap_difference


def test_agreement_compares_predictions_not_labels_and_kl_direction():
    original = torch.tensor([[2., 0.], [0., 2.], [0., 2.]])
    executed = torch.tensor([[1., 0.], [2., 0.], [0., 2.]])
    assert agreement(original, executed).tolist() == [True, False, True]
    p, q = original.double().softmax(-1), executed.double().softmax(-1)
    expected = (p * (p.log() - q.log())).sum(-1)
    torch.testing.assert_close(kl_divergence(original, executed).double(), expected)
    torch.testing.assert_close(kl_divergence(original, original), torch.zeros(3))


def test_paired_bootstrap_and_seed_averaging():
    ours = np.array([1, 0, 1, 0], dtype=bool)
    identical = np.stack([ours] * 5)
    assert paired_bootstrap_difference(ours, identical) == (0., 0., 0.)
    random = np.array([[0, 0, 0, 0], [1, 0, 1, 0]])
    actual = paired_bootstrap_difference(ours, random, seed=7, n_resamples=20)
    rng = np.random.default_rng(7)
    differences = ours.astype(float) - random.mean(0)
    means = [np.mean(differences[rng.integers(4, size=4)]) for _ in range(20)]
    low, high = np.quantile(means, [0.025, 0.975])
    np.testing.assert_allclose(actual, [0.25, low, high])


def test_subset_sampling_is_shared_reproducible_and_without_replacement():
    indices = subset_indices(3925, 500)
    assert indices == subset_indices(3925, 500, seed=0)
    assert len(set(indices)) == 500
    assert indices != subset_indices(3925, 500, seed=1)


def test_imagenette_target_remapping_uses_wordnet_ids(monkeypatch):
    class FakeImagenette:
        wnids = ["n03888257", "n01440764", "n02102040"]

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.zeros(3, 224, 224), index

    monkeypatch.setattr("mita_probe.data.Imagenette", FakeImagenette)
    monkeypatch.setattr("mita_probe.data.resolve_model_data_config", lambda model: {})
    monkeypatch.setattr("mita_probe.data.create_transform", lambda **kwargs: None)
    data = Imagenette1k(Path("unused"), model=None)
    assert [data[i][1] for i in range(len(data))] == [701, 0, 217]
    assert len(IMAGENET_IDS) == 10
