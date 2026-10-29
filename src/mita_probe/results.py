"""Read result CSVs back for analysis: deduplicate resumed batches and shape per-image tables."""

import pandas as pd

KEYS = ['model', 'index', 'layer', 'setting', 'setting_id', 'seed', 'spec', 'category', 'kind']


def load(directory, name):
    """Read name (or name.gz); drop rows repeated by a crash between a batch write and its progress record."""
    path = directory / name
    frame = pd.read_csv(path if path.exists() else path.with_name(name + '.gz'))
    keys = [c for c in KEYS if c in frame]
    duplicated = frame.duplicated(keys).sum()
    return frame.drop_duplicates(keys), int(duplicated)


def rows_of(table, **labels):
    return [dict(labels, quantity=q, mean=r['mean'], low=r['low'], high=r['high']) for q, r in table.iterrows()]


def seed_mean(frame, by):
    """One row per image, one column per setting; random seeds (suffix _s<n>) averaged per image first."""
    base = frame.assign(setting=frame['setting'].str.replace(r'_s\d+$', '', regex=True))
    return base.groupby(['index'] + by)['agree'].mean().unstack(by).sort_index()
