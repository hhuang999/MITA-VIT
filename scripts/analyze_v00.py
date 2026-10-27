"""Evaluate preregistered maximum-tie sensitivity from existing RQ3 caches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mita_probe.io import source_commit, write_json
from mita_probe.rq3_extension import bootstrap, numpy_heatmap_metrics
from mita_probe.validation import tie_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    commit = source_commit(ROOT)
    output = ROOT / 'experiments/2026-10-27_v00_ties'
    output.mkdir(exist_ok=False)
    summaries, rows, hashes = [], [], {}
    for alias, run in [('deit', 'diagnosis'), ('dino', 'dino'), ('register', 'register')]:
        source = ROOT / f'experiments/2026-10-26_rq3_{run}'
        cache = source / 'raw/observations.npz'
        digest = hashlib.sha256(cache.read_bytes()).hexdigest()
        assert digest == json.loads((source / 'checks.json').read_text())['output_sha256']
        hashes[alias] = digest
        data = dict(np.load(cache))
        assert len(data['norms']) == 3669
        prefix = json.loads((source / 'metadata.json').read_text())['num_prefix_tokens']
        sinks = data['norms'] > 3 * np.median(data['norms'], -1, keepdims=True)
        fg, valid = data['foreground'], data['valid']
        for ki, k in enumerate([16, 32]):
            selection = data['counts'][:, ki, 8:12, prefix:].mean(1)
            maps = [('selection', selection, valid), ('cls', data['cls'], valid),
                    ('selection_no_sink', selection, valid & ~sinks[:, 8:12].any(1)),
                    ('cls_no_sink', data['cls'], valid & ~sinks[:, -1])]
            for method, heatmap, mask in maps:
                count, first, expected = tie_metrics(heatmap, fg, mask)
                mass, original_first, _ = numpy_heatmap_metrics(heatmap, fg, mask)
                np.testing.assert_allclose(first, original_first, atol=0, rtol=0, equal_nan=True)
                values = {'first_pointing': first, 'uniform_tie_pointing': expected,
                          'difference': expected-first, 'tie_count': count,
                          'tie_fraction': np.where(np.isfinite(count), count > 1, np.nan),
                          'fg_mass': mass}
                for metric, array in values.items():
                    summaries.append(dict(model=alias, k=k, method=method, metric=metric, **bootstrap(array)))
                for index in range(3669):
                    rows.append(dict(model=alias, k=k, method=method, index=index,
                                     **{metric: array[index] for metric, array in values.items()}))
    pd.DataFrame(rows).to_csv(output / 'per_image.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'summary.csv', index=False)
    write_json(output / 'checks.json', dict(git_commit=commit, cache_sha256=hashes,
               n_images=3669, settings=24, first_argmax_matches_original=True))
    report = '# V00：热图最高点并列敏感性\n\n原指标保留；新增指标是在所有有效最高点中均匀选择的期望命中率。\n\n'
    report += '| 模型 | k | 热图 | 原pointing % [95% CI] | 并列期望 % [95% CI] | 差值 pp [95% CI] | 并列图片 % [95% CI] |\n| --- | --- | --- | --- | --- | --- | --- |\n'
    for keys, part in summary.groupby(['model', 'k', 'method'], sort=False):
        report += '| ' + ' | '.join(map(str, keys))
        for metric in ['first_pointing', 'uniform_tie_pointing', 'difference', 'tie_fraction']:
            r = part[part.metric == metric].iloc[0]
            report += f' | {100*r["mean"]:.2f} [{100*r.low:.2f}, {100*r.high:.2f}]'
        report += ' |\n'
    report += '\n图片级bootstrap1000次、seed0；所有3669张均处理，缺失数由CSV的n列报告。去汇聚点版本的候选掩码遵循E09。缓存SHA256与原运行一致；未进行模型前向。\n'
    (ROOT / 'docs/analysis/v00_ties.md').write_text(report, encoding='utf-8')
    (output / 'log.txt').write_text('CPU cache analysis completed; original argmax reproduced exactly.\n')
    print(summary[summary.metric == 'difference'].to_string(index=False))


if __name__ == '__main__':
    main()
