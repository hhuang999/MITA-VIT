"""CPU checks on metric semantics and experiment run validation."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from fastweight_vit.data.masks import token_foreground
from fastweight_vit.evidence import compare_csv, rebuild, validate_completed_run

ROOT = Path(__file__).resolve().parents[1]


def test_foreground_denominator_and_valid_patch_rule():
    # One ignored pixel means two foreground pixels are a majority of the three valid pixels.
    masks = torch.tensor([[1,1,3,3], [2,3,3,3], [1,2,3,3], [1,2,3,3]])
    foreground, valid = token_foreground(masks, patch_size=2)
    assert foreground.tolist() == [True, False, True, False]
    assert valid.tolist() == [True, False, True, False]
    # The exact half boundary is included; background still counts in the denominator.
    foreground, valid = token_foreground(torch.tensor([[1,2], [1,2]]), patch_size=2)
    assert foreground.item() and valid.item()


def test_imported_marker_cannot_skip_future_run(tmp_path):
    (tmp_path/'checks.json').write_text('{"complete":true,"imported":true}')
    with pytest.raises(ValueError,match='Imported marker'):
        validate_completed_run(tmp_path,{},0)


@pytest.mark.parametrize('damage', [None,'missing_method','missing_image','different_config'])
def test_completed_runs_require_full_method_image_coverage(tmp_path, damage):
    config = {'experiment':'correct','batch_size':64,'correct':{'dataset':'clickme','methods':[{'name':'R0'},{'name':'C3'}]}}
    (tmp_path/'checks.json').write_text('{"complete":true}')
    (tmp_path/'metadata.json').write_text(json.dumps({'git_commit':'a'*40,'images':3,'smoke_images':0}))
    effective = dict(config)
    if damage == 'different_config':
        effective['batch_size'] = 128
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(effective))
    frame = pd.DataFrame([{'index':i,'method':m,'spearman_pixel':.1} for i in range(3) for m in ['R0','C3']])
    if damage == 'missing_method':
        frame = frame[frame.method == 'R0']
    elif damage == 'missing_image':
        frame = frame.iloc[:-1]
    frame.to_csv(tmp_path/'alignment.csv',index=False)
    if damage:
        with pytest.raises(ValueError):
            validate_completed_run(tmp_path,config)
    else:
        validate_completed_run(tmp_path,config)


def test_comparison_checks_verdicts_intervals_and_missing_values(tmp_path):
    reference = tmp_path/'reference.csv'
    reference.write_text('model,n,mean,low,high,verdict\ndeit,12,0.038,0.036,0.039,higher\n')
    changed = tmp_path/'changed.csv'
    for content in [reference.read_text().replace('higher','comparable'),
                    reference.read_text().replace('0.036','0.026'),
                    reference.read_text().replace('deit,12','deit,11')]:
        changed.write_text(content)
        with pytest.raises(ValueError):
            compare_csv(reference,changed)


@pytest.mark.parametrize('folder', ['outputs','results','figures','paper'])
def test_rebuild_refuses_original_destinations(folder):
    with pytest.raises(ValueError,match='outside original'):
        rebuild(ROOT,ROOT/folder/'would-overwrite')
