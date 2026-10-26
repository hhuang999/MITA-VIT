from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mita_probe.io import load_config
from mita_probe.rq1 import aggregate_predictions, settings


def test_pilot_inherits_complete_sweep_and_only_overrides_dataset():
    root = Path(__file__).resolve().parents[1]
    pilot = load_config(root / "configs/rq1_pilot.yaml")
    formal = load_config(root / "configs/rq1.yaml")
    assert len(list(settings(pilot))) == 35
    assert pilot["n_images"] == 500 and pilot["dataset"] == "imagenette"
    assert formal["n_images"] == 5000 and formal["dataset"] == "imagenet_val"
    assert {key for key in pilot if pilot[key] != formal[key]} == {"n_images", "dataset", "indices"}


def example_predictions():
    config = {"k": [4], "methods": ["grid", "random", "kmeans"], "random_seeds": [0, 1],
              "subset_seed": 0, "n_images": 2, "bootstrap_resamples": 1000, "confidence_level": .95}
    rows = []
    for method, seed, values in [("grid", 0, [1, 0]), ("kmeans", 0, [0, 1]),
                                 ("random", 0, [0, 0]), ("random", 1, [1, 0])]:
        for index, agree in zip([17, 3], values):
            rows.append({"method": method, "seed": seed, "k": 4, "index": index,
                         "agree": agree, "kl": 0., "original_correct": 1,
                         "executed_correct": agree})
    return pd.DataFrame(rows).sample(frac=1, random_state=9), config


def test_aggregation_pairs_image_ids_and_separates_seed_variation():
    frame, config = example_predictions()
    summary, contrast = aggregate_predictions(frame, config)
    random = next(row for row in summary if row["method"] == "random")
    assert random["agree"] == .25
    assert random["seed_std"] == pytest.approx(np.std([0, .5], ddof=1))
    assert contrast[0]["difference"] == .25
    assert contrast[0]["ci_low"] == 0 and contrast[0]["ci_high"] == .5


@pytest.mark.parametrize("problem", ["duplicate", "missing", "different_ids"])
def test_aggregation_rejects_invalid_pairing(problem):
    frame, config = example_predictions()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "missing":
        frame = frame.iloc[1:]
    else:
        frame.loc[(frame.method == "random") & (frame["index"] == 17), "index"] = 99
    with pytest.raises(ValueError):
        aggregate_predictions(frame, config)


def test_continuation_accepts_only_complete_identical_ordered_settings():
    from mita_probe.rq1 import validate_completed_settings
    identity = pd.DataFrame({'index': [17, 3], 'target': [2, 4], 'original': [2, 5]})
    predictions = identity.assign(method='grid', k=4, seed=0, executed=[2, 5], agree=1, kl=0.)
    results = [{'grouping': 'grid', 'readset': 'ours', 'k': 4, 'seed': 0}]
    config = {'k': [4], 'methods': ['grid', 'random', 'kmeans'], 'random_seeds': [0], 'n_images': 2}
    assert validate_completed_settings(predictions, results, identity, config) == {('grid', 4, 0)}
    for broken in [predictions.iloc[:1], predictions.iloc[::-1], predictions.assign(original=99),
                   pd.concat([predictions, predictions]), predictions.assign(agree=0)]:
        with pytest.raises(ValueError):
            validate_completed_settings(broken, results, identity, config)
    with pytest.raises(ValueError):
        validate_completed_settings(predictions, [], identity, config)
