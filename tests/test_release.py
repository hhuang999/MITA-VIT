"""Scientific protocol, immutable evidence and distributable-release regressions."""

import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from fastweight_vit.data.masks import token_foreground
from fastweight_vit.gallery import SELECTED, load_gallery
from fastweight_vit.release import (archive_check, compare_csv, csv_digest, csv_profile, rebuild,
                                   scientific_invariants, semantic_config, sha256, validate,
                                   validate_artifact, validate_completed_run)

ROOT = Path(__file__).resolve().parents[1]


def test_paper_invariants_and_registered_outcomes():
    observed = scientific_invariants(ROOT)
    assert observed['R2'] == '1/4 higher: deit_b/clickme'
    assert observed['relocation'] == '9/9 lower'
    assert observed['interpretation'] == 'post hoc'
    tests = pd.read_csv(ROOT/'results/correct_tests.csv')
    dino = tests[(tests.model == 'dino') & (tests.a == 'C3') & (tests.b == 'R0')
                 & (tests.test == 'correction - R0')]
    assert len(dino) == 3 and set(dino.verdict) == {'comparable'}
    mlp = pd.read_csv(ROOT/'results/supplementary/c2_variants.csv')
    cells = mlp[(mlp.test == 'C2MLP - R0') & mlp.metric.isin(['enrichment','clickme'])]
    assert len(cells) == 6 and set(cells.verdict) == {'lower'}


def test_foreground_denominator_and_valid_patch_rule():
    # One ignored pixel means two foreground pixels are a majority of the three valid pixels.
    masks = torch.tensor([[1,1,3,3], [2,3,3,3], [1,2,3,3], [1,2,3,3]])
    foreground, valid = token_foreground(masks, patch_size=2)
    assert foreground.tolist() == [True, False, True, False]
    assert valid.tolist() == [True, False, True, False]
    # The exact half boundary is included; background still counts in the denominator.
    foreground, valid = token_foreground(torch.tensor([[1,2], [1,2]]), patch_size=2)
    assert foreground.item() and valid.item()


def test_original_protocol_fingerprints_and_preprocessing():
    manifest = json.loads((ROOT/'docs/provenance/artifacts.json').read_text())
    for name, expected in manifest['configs'].items():
        assert semantic_config(ROOT/f'configs/{name}.yaml') == expected
    table = yaml.safe_load((ROOT/'configs/table1.yaml').read_text())
    assert table['models'] == ['deit','deit_b','dino']
    assert table['datasets'] == ['imagenet_s','clickme']
    assert table['batch_size'] == 64
    assert table['correct']['subset_seed'] == table['correct']['tie_seed'] == 0
    assert next(m for m in table['correct']['methods'] if m['name'] == 'R0')['layers'] == [9,10,11,12]
    for name in ('value_norm','attention_off','register'):
        assert yaml.safe_load((ROOT/f'configs/{name}.yaml').read_text())['preprocessing'] == 'uniform'
    layers = yaml.safe_load((ROOT/'configs/clickme_layers.yaml').read_text())
    assert [layers['models'][m]['batch_size'] for m in ('deit','deit_b','dino')] == [128,64,128]
    for name, count in [('heldout_4000.json',4000),('imagenet_val_5000.json',5000),('imagenet_s_srg_2000.json',2000)]:
        data = json.loads((ROOT/'data/splits'/name).read_text())
        if isinstance(data, dict):
            assert data['probe_images'] == data['mean_images'] == 2000
            data = data['rows']
        assert len(data) == count and len(set(data)) == count


def test_seeded_v8_gallery_selection():
    data = load_gallery(ROOT/'outputs/table1/clickme/deit')
    assert data['index'][:2].tolist() == SELECTED['figure_1b']
    assert data['index'][2:6].tolist() == SELECTED['figure_D1']
    assert data['map_C2'].shape == (24,196)
    record = json.loads((ROOT/'figures/gallery_selection.json').read_text())
    assert record['all_indices'] == data['index'].tolist()


def test_main_table_has_original_sample_counts_and_complete_method_grid():
    values = pd.read_csv(ROOT/'results/table1_long.csv')
    for row in values.itertuples():
        expected = 17408 if row.metric == 'clickme' else 2000 if row.metric == 'srg' else (
            12144 if row.model == 'dino' else 12150)
        assert row.n == expected
    expected = {entry['name'] for entry in yaml.safe_load((ROOT/'configs/table1.yaml').read_text())['correct']['methods']}
    manifest = json.loads((ROOT/'docs/provenance/artifacts.json').read_text())
    for item in manifest['artifacts']:
        if item['path'].startswith('outputs/table1/clickme/') and 'alignment.csv' in item['path']:
            assert item['csv']['unique_indices'] == 17408
            assert set(item['csv']['labels']['method']) == expected
            assert item['csv']['rows'] == 17408 * len(expected)


@pytest.mark.parametrize('kind', ['missing','duplicate','wrong_indices','missing_map','nonfinite'])
def test_gallery_fails_on_incomplete_or_incompatible_input(tmp_path, kind):
    if kind != 'missing':
        data = load_gallery(ROOT/'outputs/table1/clickme/deit')
        if kind == 'duplicate':
            data['index'][1] = data['index'][0]
        elif kind == 'wrong_indices':
            data['index'][0] = 0
        elif kind == 'missing_map':
            del data['map_C3']
        else:
            data['map_C3'][0,0] = np.nan
        np.savez_compressed(tmp_path/'gallery_000000.npz', **data)
    with pytest.raises(ValueError):
        load_gallery(tmp_path)


def test_all_required_artifacts_and_authentic_records():
    report = validate(ROOT)
    assert report['passed'] and report['run_combinations'] == 42


@pytest.mark.parametrize('damage', ['missing','truncated','duplicate','bad_gzip','both_variants'])
def test_release_audit_rejects_damaged_original_data(tmp_path, damage):
    source = ROOT/'outputs/table1_mita/clickme/mita/alignment.csv'
    item = {'path':'alignment.csv','sha256':sha256(source),'csv':csv_profile(source),'csv_sha256':csv_digest(source)}
    if damage != 'missing':
        target = tmp_path/'alignment.csv'
        shutil.copyfile(source,target)
        if damage == 'truncated':
            target.write_text(','.join(pd.read_csv(source).columns)+'\n')
        elif damage == 'duplicate':
            frame = pd.read_csv(source)
            pd.concat([frame,frame.iloc[:1]]).to_csv(target,index=False)
        elif damage == 'bad_gzip':
            target.unlink()
            (tmp_path/'alignment.csv.gz').write_bytes(b'corrupt gzip')
        elif damage == 'both_variants':
            (tmp_path/'alignment.csv.gz').write_bytes(gzip.compress(source.read_bytes()))
    with pytest.raises((ValueError,FileNotFoundError,gzip.BadGzipFile)):
        validate_artifact(tmp_path,item)


def test_compressed_csv_is_an_equivalent_representation(tmp_path):
    source = ROOT/'outputs/table1_mita/clickme/mita/alignment.csv'
    item = {'path':'alignment.csv','sha256':sha256(source),'csv':csv_profile(source),'csv_sha256':csv_digest(source)}
    (tmp_path/'alignment.csv.gz').write_bytes(gzip.compress(source.read_bytes()))
    assert validate_artifact(tmp_path,item)['byte_identical'] is False


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


def test_real_archive_includes_preregistration_and_revision():
    report = archive_check(ROOT)
    assert report['passed'] and len(report['source_revision']) == 40


def test_cpu_rebuild_keeps_original_artifacts_immutable(tmp_path):
    report = rebuild(ROOT,tmp_path/'rebuilt')
    assert report['original_artifacts_unchanged']
    assert len(report['tables']) == 19
    assert max(table['max_absolute_difference'] for table in report['tables']) <= 1e-12


@pytest.mark.parametrize('folder', ['outputs','results','figures','paper'])
def test_rebuild_refuses_original_destinations(folder):
    with pytest.raises(ValueError,match='outside original'):
        rebuild(ROOT,ROOT/folder/'would-overwrite')
