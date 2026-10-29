"""CPU check of the A15 centre-matched deletion on random weights and fake Pet masks."""

import importlib.util
from pathlib import Path

import pandas as pd
import torch

from mita_probe.anatomy import capture_qk
from mita_probe.io import load_config
from mita_probe.models import load_model
from mita_probe.rq3_extension import capture_input_norms

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_cmd', ROOT / 'scripts/run_center_matched_deletion.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_rows_and_doses(tmp_path):
    config = load_config(ROOT / 'configs/anatomy_v2.yaml')
    config.update(random_seeds=[0, 1])
    torch.manual_seed(0)
    model = load_model(pretrained=False)
    images = torch.randn(3, 3, 224, 224)
    valid = torch.rand(3, 196) < .9
    fg = (torch.rand(3, 196) < .5) & valid
    with torch.inference_mode(), capture_qk(model) as qk, capture_input_norms(model) as norms:
        original = model(images)
    with torch.inference_mode():
        rows = pd.DataFrame(script.deletion(model, 'deit', qk, norms, images, fg, valid, [5, 6, 7], original, 0,
                                            config, 'cpu', tmp_path))
    assert len(rows) == 3 * (1 + 2 * (1 + 2 + 2))
    assert set(rows.kind) == {'baseline', 'target', 'random', 'centre_matched'}
    per = rows.query("kind != 'baseline'").groupby(['index', 'category'])['mean_removed_per_set'].nunique()
    assert (per == 1).all()
