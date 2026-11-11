"""CPU release integrity checks, separate from exploratory statistical builders.

References describe immutable historical observations, not expected outcomes of new GPU runs.
"""

import ast
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .gallery import SELECTED, load_gallery

ATOL = 1e-12
CONFIGS = ('attention_off', 'c2_analysis', 'clickme_layers', 'diagnose_mean_maps', 'diagnose_probe',
           'high_norm', 'k_sweep', 'register', 'supplementary_eval', 'supplementary_heldout', 'table1',
           'table1_mita', 'table1_register', 'value_norm')
IDENTIFIERS = ('index', 'row', 'model', 'method', 'setting_id', 'layer', 'layers', 'variant', 'source', 'probe',
               'condition', 'budget', 'start', 'head', 'seed', 'family', 'geometry', 'quantity', 'metric',
               'a', 'b', 'test', 'panel', 'k', 'sets', 'reading', 'improvement')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def scientific_ast(path):
    """Ignore prose only; every executable expression remains in the fingerprint."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                if isinstance(node.body[0].value.value, str):
                    node.body.pop(0)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def csv_profile(path):
    table = pd.read_csv(path)
    keys = [name for name in IDENTIFIERS if name in table]
    profile = {'rows': len(table), 'columns': table.columns.tolist(), 'keys': keys,
               'duplicate_keys': int(table.duplicated(keys).sum()) if keys else None}
    identity = 'index' if 'index' in table else 'row' if 'row' in table else None
    if identity:
        ids = sorted(table[identity].dropna().unique().tolist())
        profile['unique_indices'] = len(ids)
        profile['indices_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    profile['labels'] = {key: sorted(table[key].dropna().astype(str).unique().tolist())
                         for key in keys if key not in ('index', 'row', 'start')}
    return profile


def resolve_csv(path):
    """Accept exactly one compressed/uncompressed representation, never silently prefer one."""
    plain = Path(str(path).removesuffix('.gz'))
    candidates = [p for p in (plain, Path(str(plain) + '.gz')) if p.exists()]
    require(len(candidates) == 1, f'Expected exactly one CSV representation for {plain}; found {candidates}')
    return candidates[0]


def csv_digest(path):
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def validate_artifact(root, item):
    path = root / item['path']
    if 'csv' in item:
        path = resolve_csv(path)
        require(csv_digest(path) == item['csv_sha256'], f'Changed original CSV content: {path}')
        observed = csv_profile(path)
        require(observed == item['csv'], f'CSV schema/count/identifier mismatch: {path}')
        require(not observed['duplicate_keys'], f'Duplicate observations: {path}')
    else:
        require(path.is_file(), f'Missing artifact: {path}')
        require(sha256(path) == item['sha256'], f'Changed or corrupt artifact: {path}')
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as arrays:
                for name in arrays.files:
                    value = arrays[name]
                    require(value.dtype != object, f'Object array is not distributable: {path}:{name}')
    return {'path': item['path'], 'byte_identical': sha256(path) == item['sha256']}


def verdict(mean, low, high, metric):
    margin = .05 if metric == 'srg' else .02
    return 'comparable' if low <= 0 <= high or abs(mean) < margin else ('higher' if low > 0 else 'lower')


def scientific_invariants(root):
    tests = pd.read_csv(root / 'results/correct_tests.csv')
    negative = pd.read_csv(root / 'results/negative_controls.csv')
    variants = pd.read_csv(root / 'results/variants.csv')
    values = pd.read_csv(root / 'results/table1_long.csv')
    require(len(tests) == 99 and len(values) == 201 and len(negative) == 9 and len(variants) == 10,
            'Incomplete paper result tables')
    for name, frame in [('correct_tests', tests), ('negative_controls', negative), ('variants', variants)]:
        for row in frame[frame.verdict.notna()].itertuples():
            require(row.verdict == verdict(row.mean, row.low, row.high, row.metric),
                    f'Invalid statistical verdict: {name}, {row}')
    result = {}
    for model, metric, rounded in [('deit','enrichment',.038), ('deit_b','enrichment',.065),
                                   ('deit','clickme',.033), ('deit_b','clickme',.058)]:
        row = tests[(tests.model == model) & (tests.metric == metric) & (tests.a == 'C3') & (tests.b == 'R0')]
        require(len(row) == 1, f'Missing contribution comparison: {model}/{metric}')
        mean = float(row.iloc[0]['mean'])
        require(round(mean, 3) == rounded and row.iloc[0].verdict == 'higher', 'Contribution claim mismatch')
        result[f'C3_{model}_{metric}'] = mean
    deletion = tests[(tests.a == 'C3') & (tests.b == 'R0') & (tests.metric == 'srg')
                     & (tests.test == 'correction - R0')]
    require(len(deletion) == 3 and (deletion.verdict == 'comparable').all(), 'Deletion margin claim mismatch')
    c2 = tests[(tests.a == 'C2') & (tests.b == 'R0')]
    require(len(c2) == 9 and (c2.verdict == 'lower').all(), 'Relocation must be lower in nine deep cells')
    require((negative.verdict == 'comparable').all() and (negative.test == 'P4a').sum() == 3
            and (negative.test == 'P4b').sum() == 6, 'R1 negative controls mismatch')
    r2 = tests[(tests.a == 'R_all') & (tests.b == 'R0') & tests.model.isin(['deit','deit_b'])
               & tests.metric.isin(['enrichment','clickme'])]
    winners = r2[r2.verdict == 'higher']
    require(len(r2) == 4 and len(winners) == 1 and winners.iloc[0].model == 'deit_b'
            and winners.iloc[0].metric == 'clickme', 'R2 must partly hold in 1/4 cells')
    r3 = variants[(variants.test == 'variant - DeiT-S R0') & variants.metric.isin(['enrichment','clickme'])]
    require(len(r3) == 4 and (r3.verdict != 'higher').all(), 'R3 must hold in 4/4 cells')
    for metric, rounded in [('enrichment',.012), ('clickme',.013)]:
        row = r3[(r3.variant == 'register') & (r3.metric == metric)].iloc[0]
        require(round(row['mean'], 3) == rounded and row.verdict == 'comparable', 'Register margin claim mismatch')
        result[f'register_{metric}'] = float(row['mean'])
    gate = pd.read_csv(root / 'results/register/pilot_gate.csv')
    units = yaml.safe_load((root / 'configs/table1_register.yaml').read_text())['correct']['register_units']
    authentic = json.loads(gate[gate.model == 'deit'].iloc[0].neurons)
    require(units == authentic and len(units) == 8, 'Register units differ from authentic pilot evidence')
    result.update(R1='3/3 and 6/6 comparable', R2='1/4 higher: deit_b/clickme', R3='4/4 not higher',
                  relocation='9/9 lower', deletion='3/3 comparable', interpretation='post hoc')
    return result


def validate_evidence_map(root):
    mapping = json.loads((root/'docs/evidence.json').read_text(encoding='utf-8'))
    expected = {'Figure 1a','Figure 1b','Table 1','Figure 2','Table 2','Table A.1','Figure B.1',
                'Table B.1','Table B.2','Table C.1','Figure D.1','Appendix E'}
    require(mapping['paper_version'] == 8 and set(mapping['paper_artifacts']) == expected,
            'Incomplete v8 paper-to-code map')
    for label, paths in mapping['paper_artifacts'].items():
        require(paths and all((root/p).is_file() for p in paths), f'Missing evidence for {label}')
    require(len(mapping['items']) == 26, 'Incomplete empirical-claim map')
    for item in mapping['items']:
        for field in ('config_files','original_artifact_paths','reference_files'):
            require(all((root/p).is_file() for p in item[field]), f'Broken {field}: {item["paper_item"]}')
        require(item['cpu_commands'] and item['source_artifact_revision'], f'Unmapped claim: {item["paper_item"]}')
    require(mapping['manual_tables'] == {'Table 1':'paper/main.tex', 'Table A.1':'paper/appendix.tex',
                                         'Table C.1':'paper/tables_own/relocation_explanations.tex'},
            'Authored tables must be distinguished from generated numeric tables')
    return {'paper_items':len(expected),'empirical_claim_rows':len(mapping['items'])}


def validate(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'docs/provenance/artifacts.json').read_text(encoding='utf-8'))
    for name in CONFIGS:
        path = root / f'configs/{name}.yaml'
        require(path.is_file(), f'Missing paper configuration: {path}')
        require(semantic_config(path) == manifest['configs'][name], f'Changed protocol: {path}')
    for path, digest in manifest['scientific_ast'].items():
        require(scientific_ast(root / path) == digest, f'Changed scientific implementation: {path}')
    artifacts = [validate_artifact(root, item) for item in manifest['artifacts']]
    for path in manifest['required_documents']:
        require((root / path).is_file(), f'Missing release evidence: {path}')
    runs = json.loads((root / 'docs/provenance/runs.json').read_text(encoding='utf-8'))
    historical = (root / 'docs/preregistration.md').read_bytes()[:runs['historical_preregistration_bytes']]
    require(hashlib.sha256(historical).hexdigest() == runs['historical_preregistration_sha256'],
            'Historical pre-registration was rewritten or removed')
    prereg = (root / 'docs/preregistration.md').read_text(encoding='utf-8')
    for term in ('P4a / P4b', 'P3', 'P6', 'Post hoc', '2026-11-05', '94bf426'):
        require(term in prereg, f'Missing pre-registration information: {term}')
    expected_runs = set()
    for name in CONFIGS:
        if name == 'register':  # Historical pilot evidence is distributed in results/register, not a new run.
            continue
        config = yaml.safe_load((root / f'configs/{name}.yaml').read_text())
        for model in config['models']:
            for dataset in config.get('datasets', [None]):
                expected_runs.add('/'.join(filter(None, [name, dataset, model])))
    require({run['run'] for run in runs['runs']} == expected_runs, 'Incomplete model/dataset provenance mapping')
    for run in runs['runs']:
        require(run['sources'], f'Unknown provenance status: {run["run"]}')
        for source in run['sources']:
            require(source['status'] and source['source_revision'], 'Unspecified historical source')
            for record in source['records']:
                require(sha256(root / record['path']) == record['sha256'], f'Changed historical record: {record["path"]}')
    for record in runs['register_evidence_records']:
        require(sha256(root / record['path']) == record['sha256'], f'Changed register record: {record["path"]}')
    gallery = load_gallery(root / 'outputs/table1/clickme/deit')
    selection = json.loads((root / 'figures/gallery_selection.json').read_text())
    require(selection == {'all_indices': gallery['index'].tolist(), **SELECTED}, 'Incorrect figure selection record')
    invariants = scientific_invariants(root)
    return {'passed': True, 'artifact_count': len(artifacts), 'artifacts': artifacts,
            'run_combinations': len(expected_runs), 'scientific_invariants': invariants,
            'new_gpu_experiments': False, 'absolute_tolerance': ATOL, 'evidence_map': validate_evidence_map(root)}


def compare_csv(reference, rebuilt):
    a, b = pd.read_csv(reference), pd.read_csv(rebuilt)
    require(a.shape == b.shape and a.columns.tolist() == b.columns.tolist(), f'Incomplete rebuilt table: {rebuilt}')
    largest = 0.
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            x, y = a[col].to_numpy(float), b[col].to_numpy(float)
            require(np.array_equal(np.isnan(x), np.isnan(y)), f'Changed missing values: {rebuilt}:{col}')
            require(np.allclose(x, y, atol=ATOL, rtol=0, equal_nan=True), f'Numerical mismatch: {rebuilt}:{col}')
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                largest = max(largest, float(np.abs(x[finite] - y[finite]).max()))
        else:
            require(a[col].fillna('<NA>').equals(b[col].fillna('<NA>')), f'Changed labels/verdicts: {rebuilt}:{col}')
    return {'path': str(reference.name), 'rows': len(a), 'max_absolute_difference': largest,
            'byte_identical': reference.read_bytes() == rebuilt.read_bytes()}


def rebuild(root, destination, outputs=None, compare_reference=True):
    """Build only into a fresh directory; verify every original artifact remained unchanged."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    outputs = Path(outputs).resolve() if outputs is not None else root/'outputs'
    require(not any(destination == (root/p) or (root/p) in destination.parents
                    for p in ('outputs','results','figures','paper','data','src','configs','tests')),
            'Rebuild destination must be outside original artifact and source directories')
    require(not destination.exists(), f'Rebuild destination already exists: {destination}')
    protected = [p for folder in ('outputs','results','figures','paper') for p in (root/folder).rglob('*') if p.is_file()]
    before = {p: sha256(p) for p in protected}
    destination.mkdir(parents=True)
    commands = [
        ['scripts/make_tables.py','--outputs',str(outputs),'--results',str(destination/'results')],
        ['scripts/make_figures.py','--outputs',str(outputs),'--results',str(destination/'results'),'--figures',str(destination/'figures')],
        ['scripts/make_latex_tables.py','--results',str(destination/'results'),'--out',str(destination/'paper/tables')],
        ['-I','paper/make/make_main_table.py',str(destination/'results'),str(destination/'paper/tables_own/main_results.tex')],
        ['-I','paper/make/diag_table.py',str(destination/'results/diagnose.csv'),str(destination/'paper/tables_own/diagnostics_per_layer.tex')],
        ['-I','paper/make/make_paired_table.py',str(destination/'paper/tables'),str(destination/'paper/tables_own/paired_differences.tex')],
        ['-I','paper/make/make_diagnosis_figure.py',str(destination/'results'),str(destination/'paper/figures_own/diagnosis.pdf')],
    ]
    (destination/'paper/tables_own').mkdir(parents=True)
    (destination/'paper/figures_own').mkdir(parents=True)
    try:
        for command in commands:
            subprocess.run([sys.executable, *command], cwd=root, check=True)
        if not compare_reference:
            return {'passed': True, 'commands': commands, 'reference_comparison': 'not requested for optional new observations'}
        comparisons = []
        originals = sorted((root/'results').rglob('*.csv'))
        for source in originals:
            if 'register' not in source.parts:
                target = destination/source.relative_to(root)
                require(target.is_file(), f'Builder omitted required table: {target}')
                row = compare_csv(source, target)
                row['path'] = source.relative_to(root).as_posix()
                comparisons.append(row)
        for source in (root/'paper/tables').rglob('*.tex'):
            require(source.read_bytes() == (destination/source.relative_to(root)).read_bytes(), f'Changed paired source: {source}')
        for name in ('main_results','diagnostics_per_layer','paired_differences'):
            source = root/f'paper/tables_own/{name}.tex'
            require(source.read_bytes() == (destination/source.relative_to(root)).read_bytes(), f'Changed paper table: {source}')
        require(json.loads((destination/'figures/gallery_selection.json').read_text()) ==
                json.loads((root/'figures/gallery_selection.json').read_text()), 'Rebuilt gallery selection changed')
        return {'passed': True, 'commands': commands, 'tables': comparisons, 'original_artifacts_unchanged': True,
                'paper_tables': '3 generated tables and paired source byte-identical; manual tables preserved',
                'figures': 'Built from verified original inputs; font rendering may differ across systems'}
    finally:
        after = {p for folder in ('outputs','results','figures','paper') for p in (root/folder).rglob('*') if p.is_file()}
        require(after == set(before) and all(p.is_file() and sha256(p) == h for p,h in before.items()),
                'Rebuild modified original artifacts')


def archive_check(root):
    root = Path(root)
    if not (root/'.git').exists():
        revision = (root/'.git_archival.txt').read_text().strip()
        require(re.fullmatch('[0-9a-f]{40}', revision), 'Export lacks its immutable source revision')
        return {'passed': True, 'source_revision': revision, 'context': 'extracted archive'}
    with tempfile.TemporaryDirectory(prefix='vit-archive-') as temp:
        temp = Path(temp)
        subprocess.run(['git','archive','--format=zip','-o',str(temp/'code.zip'),'HEAD'],cwd=root,check=True)
        with zipfile.ZipFile(temp/'code.zip') as z:
            names = z.namelist()
            require(not any(n.startswith(('archive/','.git/','checkpoints/','local-paper-materials/')) or
                            n.endswith(('.pth','.pt','.safetensors','.env','.sty')) for n in names), 'Private/external files in ZIP')
            z.extractall(temp/'code')
        report = validate(temp/'code')
        revision = (temp/'code/.git_archival.txt').read_text().strip()
        require(revision == subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(), 'Archive revision substitution failed')
        return {'passed': True, 'source_revision': revision, 'files': sum(not n.endswith('/') for n in names),
                'artifacts_checked': report['artifact_count']}


def validate_completed_run(output, config, smoke=0):
    """A marker alone cannot certify a future run. Check its data before skipping it."""
    output = Path(output)
    checks = json.loads((output/'checks.json').read_text())
    require(checks.get('complete') is True, f'Incomplete run marker: {output}')
    require(not checks.get('imported'), f'Imported marker is not a future execution record: {output}')
    metadata = json.loads((output/'metadata.json').read_text())
    effective = yaml.safe_load((output/'config.yaml').read_text())
    require(all(effective.get(k) == v for k,v in config.items()), f'Completed run used a different config: {output}')
    require(metadata.get('smoke_images', 0) == smoke, f'Smoke/full run mismatch: {output}')
    require(re.fullmatch('[0-9a-f]{40}', metadata.get('git_commit','')), f'Missing source revision: {output}')
    experiment, stage = config['experiment'], config.get('stage')
    if experiment == 'correct':
        settings = config['correct']
        names = {entry['name'] for entry in settings['methods']}
        clickme = settings['dataset'] == 'clickme'
        tables = ['alignment.csv'] if clickme else ['plausibility.csv']
        if not clickme and settings.get('variant') != 'mita':
            tables.append('faithfulness.csv')
    else:
        tables = {('execution','k_sweep'): ['predictions.csv','settings.csv'],
                  ('execution','attention_off'): ['attention_off.csv'],
                  ('diagnose','probe'): ['probe_accuracy.csv','probe_accuracy_images.csv'],
                  ('diagnose','mean_maps'): ['m1.csv','radial.csv'],
                  ('diagnose','high_norm'): ['high_norm.csv'],
                  ('register_search',None): [],
                  ('c2_analysis',None): ['probe_errors.csv'],
                  ('supplementary','heldout'): ['mlp_probe.csv','global.csv'],
                  ('supplementary','eval'): (['alignment.csv'] if config.get('supplementary',{}).get('dataset')=='clickme'
                                                        else ['plausibility.csv','faithfulness.csv'])}[(experiment,stage)]
        names = None
    for name in tables:
        path = resolve_csv(output/name)
        frame = pd.read_csv(path)
        profile = csv_profile(path)
        require(len(frame) > 0 and not profile['duplicate_keys'], f'Empty/duplicate completed run: {path}')
        if names and name != 'faithfulness.csv':
            require(set(frame.method) == names, f'Missing methods: {path}')
            expected = smoke or metadata.get('images')
            require(expected is not None and all(n == expected for n in frame.groupby('method')['index'].nunique()),
                    f'Incomplete image coverage: {path}')
        if name == 'faithfulness.csv' and experiment == 'correct':
            expected_names = names - set(settings.get('no_srg',[]))
            require(set(frame.method) == expected_names, f'Missing deletion methods: {path}')
            require(set(frame.budget) == set(settings['budgets']) and set(frame.condition) == set(settings['conditions']),
                    f'Incomplete deletion grid: {path}')
            expected = metadata.get('faithfulness_images')
            require(expected is not None and (frame.groupby(['method','condition','budget'])['index'].nunique() == expected).all(),
                    f'Incomplete deletion samples: {path}')
    arrays = {('diagnose','probe'): ['probes.npz','heldout.json'],
              ('diagnose','mean_maps'): ['means_crop.npz','means_clickme.npz'],
              ('register_search',None): ['gate.json'],
              ('c2_analysis',None): ['mean_prediction.npz']}.get((experiment,stage), [])
    for name in arrays:
        path = output/name
        require(path.is_file(), f'Missing completed run artifact: {path}')
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as data:
                require(bool(data.files), f'Empty array artifact: {path}')
