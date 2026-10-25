import json
from pathlib import Path

import pytest

from mita_probe.data import experiment_indices, subset_indices
from mita_probe.io import load_config
from mita_probe.rq1 import settings
from mita_probe.runtime import project_path


def test_hpc_smoke_uses_formal_sample_prefix_and_full_sweep(tmp_path):
    root = Path(__file__).resolve().parents[1]
    formal = load_config(root / "configs/rq1.yaml")
    smoke = load_config(root / "configs/rq1_hpc_smoke.yaml")
    assert {key for key in smoke if smoke[key] != formal.get(key)} == {"n_images", "purpose"}
    assert list(settings(smoke)) == list(settings(formal))
    assert len(list(settings(smoke))) == 35
    saved = tmp_path / "indices.json"
    indices = subset_indices(50000, 5000, 0)
    saved.write_text(json.dumps(indices))
    assert experiment_indices(50000, 50, 0, saved) == indices[:50]
    assert experiment_indices(50000, 5000, 0, saved) == indices
    with pytest.raises(ValueError):
        experiment_indices(50000, 5001, 0, saved)
    saved.write_text(json.dumps(indices[::-1]))
    with pytest.raises(ValueError, match="Saved indices"):
        experiment_indices(50000, 50, 0, saved)


def test_persistent_data_volume_applies_to_images_and_saved_indices(tmp_path, monkeypatch):
    monkeypatch.setenv("MITA_DATA_ROOT", str(tmp_path))
    assert project_path("data/imagenet_val") == tmp_path / "imagenet_val"
    assert project_path("data/imagenet_val_indices_seed0.json") == tmp_path / "imagenet_val_indices_seed0.json"
    assert project_path(str(tmp_path / "external.json")) == tmp_path / "external.json"
    assert project_path("configs/rq1.yaml").name == "rq1.yaml"
