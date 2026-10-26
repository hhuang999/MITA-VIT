"""Add same-hardware RQ1 checks and an explicitly hypothetical RQ2 product baseline."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from mita_probe.evaluate import compare_runs
from mita_probe.io import write_json, load_config
from mita_probe.rq1 import mean_interval

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
LOCAL_RUN = EXP / '2026-10-26_rq1_rtx4060_continued'


def estimate(value, low, high):
    return f'{value*100:.2f} [{low*100:.2f}, {high*100:.2f}]'


def main():
    output = EXP / '2026-10-26_hardware_accumulation'
    output.mkdir(exist_ok=False)
    old = pd.read_csv(EXP / '2026-10-26_rq1_imagenet/predictions.csv')
    new = pd.read_csv(LOCAL_RUN / 'predictions.csv')
    identity = pd.read_csv(LOCAL_RUN / 'identity_predictions.csv').sort_values('index')
    expected = pd.read_csv(EXP / '2026-10-26_rq1_imagenet/identity_predictions.csv').sort_values('index')
    pd.testing.assert_frame_equal(identity.reset_index(drop=True), expected.reset_index(drop=True))
    assert len(new) == 175000 and new.groupby(['method', 'k', 'seed']).ngroups == 35
    assert not new.duplicated(['index', 'method', 'k', 'seed']).any()
    assert np.isfinite(new.kl).all() and identity.agree.eq(1).all()
    config = load_config(LOCAL_RUN / 'config.yaml')
    settings = pd.read_csv(LOCAL_RUN / 'results.csv')
    settings['method'] = np.where(settings.readset == 'random', 'random', settings.grouping)
    checks = []
    rows = []
    for (method, k, seed), frame in new.groupby(['method', 'k', 'seed']):
        reference = old[(old.method == method) & (old.k == k) & (old.seed == seed)]
        rows.append(dict(method=method, k=k, seed=seed, **compare_runs(frame, reference)))
        assert len(frame) == 5000 and set(frame['index']) == set(identity['index'])
        assert np.array_equal(frame.agree, (frame.original == frame.executed).astype(int))
        assert frame.kl.min() > -1e-10
        row = settings[(settings.method == method) & (settings.k == k) & (settings.seed == seed)].iloc[0]
        low, high = mean_interval(frame.agree.to_numpy(), config)
        assert np.allclose([frame.agree.mean(), low, high, frame.kl.mean()],
                           [row.agree, row.agree_ci_low, row.agree_ci_high, row.kl], atol=1e-12)
        checks.append(dict(method=method, k=k, seed=seed, samples=5000,
                           exact_reference_pairing=True, agreement_recomputed=True,
                           agreement_ci_recomputed=True, finite_nonnegative_kl=True))
    write_json(output / 'rq1_setting_audit.json', dict(all_passed=True, settings=checks))
    pd.DataFrame(rows).to_csv(output / 'hardware_paired_differences.csv', index=False)
    single = pd.read_csv(EXP / '2026-10-26_rq2_imagenet/predictions.csv')
    products = []
    for k in [8, 32]:
        matrix = single[single.k == k].pivot(index='index', columns='layer', values='agree').sort_index()
        all_layers = new[(new.k == k) & (new.method == 'grid') & (new.seed == 0)].set_index('index').sort_index()
        assert matrix.index.equals(all_layers.index) and matrix.shape == (5000, 12)
        assert matrix.notna().all().all()
        assert single[single.k == k].groupby('index').original.first().sort_index().equals(all_layers.original)
        x, actual = matrix.to_numpy(), all_layers.agree.to_numpy()
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(1000):
            indices = rng.integers(5000, size=5000)
            product = x[indices].mean(0).prod()
            observed = actual[indices].mean()
            samples.append([product, observed, observed-product])
        low, high = np.quantile(samples, [.025, .975], axis=0)
        product, observed = x.mean(0).prod(), actual.mean()
        for j, (metric, value) in enumerate([('independent_product', product), ('observed_all_layers', observed),
                                           ('observed_minus_product', observed-product)]):
            products.append(dict(k=k, metric=metric, mean=value, low=low[j], high=high[j], n=5000))
    pd.DataFrame(products).to_csv(output / 'independent_accumulation.csv', index=False)
    summary = pd.read_csv(EXP / '2026-10-26_rq1_same_hardware/summary.csv')
    report = '''# 补充核对：RQ1统一硬件与RQ2独立累积估算

## 结论与范围

按用户要求，在RQ3扩展之后完成。原RQ1/RQ2文件与设置不改，新增对照另存。原k4–64的35组在RTX4060重新运行，同原k128 RTX4060结果合并为同硬件六点曲线。模型、5000张索引、FP32、batch32、冻结说明书、方法和种子完全保持。

统一硬件结果如下，所有区间为图片bootstrap1000次、seed0、95%。随机先逐图平均五种子；同seed跨GPU不保证生成相同随机集合，因此A800→4060的差值包含随机集合变化，不能全部叫作浮点误差。

| k | 方法 | 一致率 % [95% CI] | KL [95% CI] |
| --- | --- | --- | --- |
'''
    for _, r in summary.iterrows():
        report += f'| {r.k} | {r.method} | {estimate(r.agree, r.agree_ci_low, r.agree_ci_high)} | {r.kl:.4f} [{r.kl_ci_low:.4f}, {r.kl_ci_high:.4f}] |\n'
    report += '\n![同硬件RQ1曲线](../../experiments/2026-10-26_rq1_same_hardware/rq1_agreement.png)\n'
    report += '\n## 与原A800结果逐设置配对比较\n\n差值=4060−A800，单位pp。所有图片ID、标签、原预测与k197恒等逐项一致。\n\n| k | 方法 | seed | 差值 pp [95% CI] |\n| --- | --- | --- | --- |\n'
    for r in rows:
        report += f"| {r['k']} | {r['method']} | {r['seed']} | {estimate(r['difference'], r['ci_low'], r['ci_high'])} |\n"
    report += '''
## 独立累积估算

独立性参照定义为十二个单层一致率的乘积 ∏ₗaₗ；它假设各层保留预测的事件独立，且可用单层干预代替级联中的条件行为。两条假设都不是已证实事实。每次bootstrap同步重采样5000张图、重新算十二层均值和乘积，并与同一批图的真实全层一致率比较；没有将5000×12当独立样本。

| k | 数量 | 估计 %（差值为pp）[95% CI] |
| --- | --- | --- |
'''
    for r in products:
        report += f"| {r['k']} | {r['metric']} | {estimate(r['mean'], r['low'], r['high'])} |\n"
    report += '''
这是独立性假设下的参照，不是对实际多层误差来源的因果分解。如果真实全层一致率高于乘积，说明简单独立累积会夸大损失，可能与图片难度相关、不同层错误重叠、级联交互或误差修正有关；这些机制不能由乘积唯一识别。仍可确认多层替换比单层替换损失更大，但不应把“逐层独立失败概率相乘”当作已验证解释。

## 复现与交付

同硬件35组完整结果在 `2026-10-26_rq1_rtx4060_continued`：首次进程在24组完成后随会话中断停止，原目录 `2026-10-26_rq1_rtx4060` 完整保留。接续前重新做5000张恒等检查，严格验证配置/软件/硬件/权重/图片顺序/原预测，再复制24个完整设置并只计算剩余11组；每个(k,seed)独立初始化随机生成器，未续用半截随机流。results.csv保留各组自己的原git commit与运行来源。

六点合并在 `2026-10-26_rq1_same_hardware`，硬件差值/乘积/检查在 `2026-10-26_hardware_accumulation`。原报告保留，是否将新同硬件曲线替换为论文主图由用户审阅后决定。没有启动HPC。
'''
    (ROOT / 'docs/analysis/hardware_and_accumulation.md').write_text(report, encoding='utf-8')
    write_json(output / 'checks.json', dict(same_images_labels_original=True, identity_agreement=1,
        n_settings=35, n_predictions=175000, rq2_paired_images=5000, old_results_unchanged=True))
    (output / 'log.txt').write_text('Same-hardware pairing and image-paired independence baseline completed.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
