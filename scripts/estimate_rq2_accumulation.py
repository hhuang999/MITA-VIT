"""Compute the requested independence reference from existing RQ2 and RQ1 CSV."""

from pathlib import Path
import numpy as np
import pandas as pd

from mita_probe.io import source_commit, write_json

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'


def main():
    commit = source_commit(ROOT)
    output = EXP / '2026-10-26_rq2_independence'
    output.mkdir(exist_ok=False)
    single = pd.read_csv(EXP / '2026-10-26_rq2_imagenet/predictions.csv')
    joint = pd.read_csv(EXP / '2026-10-26_rq1_imagenet/predictions.csv')
    rows = []
    for k in [8, 32]:
        x = single[single.k == k].pivot(index='index', columns='layer', values='agree').sort_index()
        y = joint[(joint.k == k) & (joint.method == 'grid')].set_index('index').sort_index()
        assert x.shape == (5000, 12) and x.index.equals(y.index)
        assert x.notna().all().all()
        assert single[single.k == k].groupby('index').original.first().sort_index().equals(y.original)
        x, y = x.to_numpy(), y.agree.to_numpy()
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(1000):
            indices = rng.integers(5000, size=5000)
            product, observed = x[indices].mean(0).prod(), y[indices].mean()
            samples.append([product, observed, observed-product])
        low, high = np.quantile(samples, [.025, .975], axis=0)
        values = [x.mean(0).prod(), y.mean(), y.mean()-x.mean(0).prod()]
        for j, metric in enumerate(['independent_product', 'observed_all_layers', 'observed_minus_product']):
            rows.append(dict(k=k, metric=metric, mean=values[j], low=low[j], high=high[j], n=5000))
    pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
    write_json(output / 'metadata.json', dict(git_commit=commit,
        rq2_source='2026-10-26_rq2_imagenet', rq1_source='2026-10-26_rq1_imagenet',
        bootstrap_resamples=1000, seed=0, confidence=.95, no_new_model_inference=True))
    report = '''# RQ2 补充：独立累积参照

**真实全层一致率高于十二个单层一致率的乘积。** 简单的逐层独立失败模型夸大了损失，不能把它当作已验证的误差累积机制。

按用户要求在RQ3扩展完成后计算，不改原RQ1/RQ2数据。设aₗ是只替换第l层的一致率，独立性参照为∏ₗaₗ。使用原5000张图片，bootstrap1000次、seed0、95%；每次同步重采样图片，重新计算十二层均值、乘积及全层真实一致率。这样保留同一图片在不同层的相关结构用于区间计算，并不假设层是独立样本。

| k | 数量 | 估计 %（差值为pp）[95% CI] |
| --- | --- | --- |
'''
    for r in rows:
        report += f"| {r['k']} | {r['metric']} | {r['mean']*100:.2f} [{r['low']*100:.2f}, {r['high']*100:.2f}] |\n"
    report += '''
独立参照还假设单层干预行为可以代表级联中每一层的条件行为，这也不成立于一般神经网络。差距可能来自困难图片在不同层重叠、预测变化之间的依赖、层间补偿或冻结说明书交互；仅凭乘积不能区分这些解释。结论应写成“多层替换带来明显的组合损失，简单独立乘积不足以解释其规模”，而不是“已经证明逐层独立累积”。

这里观测全层基准来自原A800结果，单层来自4060；正在进行的同硬件RQ1补充将再次检查这两个全层点。全部原文件保留，本计算没有新增模型前向。
'''
    (ROOT / 'docs/analysis/rq2_independence.md').write_text(report, encoding='utf-8')
    (output / 'log.txt').write_text('Completed paired-image independence reference; no inference; original files unchanged.\n', encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
