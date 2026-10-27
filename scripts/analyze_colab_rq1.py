"""Audit the complete Colab replication and report paired hardware comparisons."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mita_probe.evaluate import compare_runs
from mita_probe.io import load_config, source_commit, write_json
from mita_probe.rq1 import mean_interval

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
BASE = EXP / '2026-10-27_rq1_colab'
EXTRA = EXP / '2026-10-27_rq1_colab_k128'
MERGED = EXP / '2026-10-27_rq1_colab_combined'
OUTPUT = EXP / '2026-10-27_colab_hardware_audit'
OLD_BASE = EXP / '2026-10-26_rq1_imagenet'
OLD_EXTRA = EXP / '2026-10-26_rq1_k128_retry'


def interval(value, low, high, scale=100):
    return f'{value * scale:.2f} [{low * scale:.2f}, {high * scale:.2f}]'


def main():
    commit = source_commit(ROOT)
    OUTPUT.mkdir(exist_ok=False)
    metadata = [json.loads((p / 'metadata.json').read_text()) for p in [BASE, EXTRA]]
    for field in ['torch', 'timm', 'device', 'dtype', 'preprocessing', 'indices_sha256',
                  'checkpoint_revision', 'explanation_source', 'random_rng', 'git_commit']:
        assert metadata[0][field] == metadata[1][field], field
    assert metadata[0]['device'].startswith('NVIDIA ')
    identity = pd.read_csv(BASE / 'identity_predictions.csv').sort_values('index').reset_index(drop=True)
    for path in [EXTRA, OLD_BASE, OLD_EXTRA]:
        other = pd.read_csv(path / 'identity_predictions.csv').sort_values('index').reset_index(drop=True)
        pd.testing.assert_frame_equal(identity, other)
    assert len(identity) == 5000 and identity.agree.eq(1).all()
    new = pd.concat([pd.read_csv(p / 'predictions.csv') for p in [BASE, EXTRA]], ignore_index=True)
    old = pd.concat([pd.read_csv(p / 'predictions.csv') for p in [OLD_BASE, OLD_EXTRA]], ignore_index=True)
    assert len(new) == 210000 and new.groupby(['method', 'k', 'seed']).ngroups == 42
    assert not new.duplicated(['method', 'k', 'seed', 'index']).any()
    summaries = pd.concat([pd.read_csv(p / 'results.csv') for p in [BASE, EXTRA]], ignore_index=True)
    summaries['method'] = np.where(summaries.readset == 'random', 'random', summaries.grouping)
    config = load_config(BASE / 'config.yaml')
    rows, checks = [], []
    for (method, k, seed), frame in new.groupby(['method', 'k', 'seed']):
        assert len(frame) == 5000 and set(frame['index']) == set(identity['index'])
        ordered = frame.set_index('index').sort_index()
        assert np.array_equal(ordered.original, identity.original)
        assert np.array_equal(ordered.target, identity.target)
        assert np.array_equal(frame.agree, (frame.original == frame.executed).astype(int))
        assert np.isfinite(frame.kl).all() and frame.kl.min() > -1e-10
        summary = summaries[(summaries.method == method) & (summaries.k == k)
                            & (summaries.seed == seed)].iloc[0]
        low, high = mean_interval(frame.agree.to_numpy(), config)
        assert np.allclose([frame.agree.mean(), low, high, frame.kl.mean()],
                           [summary.agree, summary.agree_ci_low, summary.agree_ci_high, summary.kl],
                           atol=1e-12, rtol=0)
        reference = old[(old.method == method) & (old.k == k) & (old.seed == seed)]
        rows.append({'method': method, 'k': k, 'seed': seed,
                     'reference_device': 'RTX 4060' if k == 128 else 'A800-SXM4-80GB',
                     **compare_runs(frame, reference)})
        checks.append({'method': method, 'k': k, 'seed': seed, 'passed': True})
    pd.DataFrame(rows).to_csv(OUTPUT / 'hardware_paired_differences.csv', index=False)
    single = pd.read_csv(EXP / '2026-10-26_rq2_imagenet/predictions.csv')
    products = []
    for k in [8, 32]:
        matrix = single[single.k == k].pivot(index='index', columns='layer', values='agree').sort_index()
        joint = new[(new.k == k) & (new.method == 'grid') & (new.seed == 0)].set_index('index').sort_index()
        assert matrix.shape == (5000, 12) and matrix.index.equals(joint.index)
        assert matrix.notna().all().all()
        assert single[single.k == k].groupby('index').original.first().sort_index().equals(joint.original)
        x, y = matrix.to_numpy(), joint.agree.to_numpy()
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(1000):
            indices = rng.integers(5000, size=5000)
            product, observed = x[indices].mean(0).prod(), y[indices].mean()
            samples.append([product, observed, observed - product])
        low, high = np.quantile(samples, [.025, .975], axis=0)
        values = [x.mean(0).prod(), y.mean(), y.mean() - x.mean(0).prod()]
        for j, metric in enumerate(['independent_product', 'observed_all_layers', 'observed_minus_product']):
            products.append({'k': k, 'metric': metric, 'mean': values[j],
                             'low': low[j], 'high': high[j], 'n': 5000})
    pd.DataFrame(products).to_csv(OUTPUT / 'independent_accumulation.csv', index=False)
    write_json(OUTPUT / 'checks.json', {'git_commit': commit, 'all_passed': True,
               'settings': checks, 'n_predictions': len(new), 'same_images_labels_original': True,
               'identity_agreement': 1, 'same_hardware_and_software': True})
    write_report(identity, rows, products, metadata[0])
    (OUTPUT / 'log.txt').write_text('All 42 settings audited; paired image bootstrap and RQ2 product checked.\n')
    print('All 42 Colab settings audited; report and paired differences saved.')


def write_report(identity, differences, products, metadata):
    summary = pd.read_csv(MERGED / 'summary.csv')
    provenance = json.loads((MERGED / 'provenance.json').read_text())
    contrasts = pd.read_csv(MERGED / 'paired_contrasts.csv')
    grid128 = summary[(summary.k == 128) & (summary.method == 'grid')].iloc[0]
    km128 = summary[(summary.k == 128) & (summary.method == 'kmeans')].iloc[0]
    random128 = summary[(summary.k == 128) & (summary.method == 'random')].iloc[0]
    report = f'''# RQ1 补充：Colab 统一硬件核对

## 1. 一句话结论

同一张 {metadata['device']} 上完成全部42设置；k=128 时网格一致率 {grid128.agree:.2%}、K-means {km128.agree:.2%}、随机五种子均值 {random128.agree:.2%}。

## 2. 回答的问题

核对研究设计 §6.2 的六点 RQ1 曲线是否依赖此前混用 A800 与 RTX 4060 的计算来源。原数据和报告保留，本报告是完整统一硬件复算；同时以新 RQ1 全层结果复核既有 RQ2 独立累积参照。

## 3. 设置

Colab {metadata['device']}；PyTorch {metadata['torch']}、timm {metadata['timm']}。ImageNet固定5000张、m=16、k=4/8/16/32/64/128、随机种子0–4、FP32、batch32与冻结原模型说明书均不变。先跑原35组，再跑k128七组，两次各做5000张恒等验收。源Parquet、抽样文件和模型SHA-256与原实验一致，使用同一软件和硬件。

区间为1000次图片bootstrap、seed0、95%。随机主曲线先逐图平均五种子；种子标准差单列，不等同于置信区间。同seed跨GPU不保证生成同一随机集合，因此跨硬件随机结果差异不能全部叫作浮点误差。

## 4. 结果

![统一硬件RQ1曲线](../../experiments/2026-10-27_rq1_colab_combined/rq1_agreement.png)

| k | 方法 | 一致率 % [95% CI] | KL [95% CI] | 随机种子SD pp |
| --- | --- | --- | --- | --- |
'''
    for _, row in summary.iterrows():
        report += (f'| {int(row.k)} | {row.method} | {interval(row.agree, row.agree_ci_low, row.agree_ci_high)} | '
                   f'{row.kl:.4f} [{row.kl_ci_low:.4f}, {row.kl_ci_high:.4f}] | {row.seed_std * 100:.3f} |\n')
    report += f"\n测试点中达到90%所需最小k：网格 {provenance['minimum_k_at_90_percent']['grid']}；K-means {provenance['minimum_k_at_90_percent']['kmeans']}。这不是未测试预算上的精确阈值。\n"
    near = summary[(summary.method != 'random') & (summary.agree_ci_low <= .9)
                   & (summary.agree_ci_high >= .9)]
    if not near.empty:
        labels = ', '.join(f'{r.method} k={int(r.k)}' for _, r in near.iterrows())
        report += (f'\n{labels} 的95%区间跨过90%。上述最小k按样本均值定义；'
                   '门槛附近少数预测变化即可改变该离散统计，不代表总体门槛已被精确确定。\n')
    report += '\n### 方法配对差值\n\n差值=网格−对照，随机取seed0；使用 `evaluate.compare_runs`。\n\n| k | 对比 | 差值 pp [95% CI] |\n| --- | --- | --- |\n'
    for _, row in contrasts.iterrows():
        report += f'| {int(row.k)} | {row.contrast} | {interval(row.difference, row.ci_low, row.ci_high)} |\n'
    groups = pd.read_csv(MERGED / 'group_size_summary.csv')
    report += ('\n### K-means每组query数量\n\n'
               '组大小从原模型Q统计，与k无关；本次在k128运行中收集。'
               '欧氏聚类的patch分配与最终点积路由（含CLS）分别列出。\n\n'
               '| 层 | 分配方式 | 最小 | 中位数 | 最大 | 空组比例 % |\n'
               '| --- | --- | --- | --- | --- | --- |\n')
    for _, row in groups.iterrows():
        kind = '点积路由（含CLS）' if row.kind == 'dot_product_routes_including_cls' else '欧氏patch聚类'
        report += (f'| {int(row.layer)} | {kind} | {int(row.minimum)} | {row["median"]:.1f} | '
                   f'{int(row.maximum)} | {row.empty_fraction * 100:.2f} |\n')
    report += ('\n较大的组必须共享同一读取集合；在k=4时这可能限制对不同query的表达。'
               '欧氏聚类与点积路由目标不一致也可能产生不均衡；这些是与组大小相容的解释，'
               '并非本实验已经分离证明的原因。表中是全体组的描述性分布。\n')
    report += f"\n### 跨硬件复核\n\n差值=新{metadata['device']}−旧结果；k≤64旧结果来自A800，k128来自4060。\n\n| k | 方法 | seed | 旧硬件 | 差值 pp [95% CI] |\n| --- | --- | --- | --- | --- |\n"
    for row in differences:
        report += (f"| {row['k']} | {row['method']} | {row['seed']} | {row['reference_device']} | "
                   f"{interval(row['difference'], row['ci_low'], row['ci_high'])} |\n")
    report += '\n### RQ2 独立累积参照\n\n| k | 数量 | 估计 %（差值为pp）[95% CI] |\n| --- | --- | --- |\n'
    for row in products:
        report += f"| {row['k']} | {row['metric']} | {interval(row['mean'], row['low'], row['high'])} |\n"
    report += f'''
RQ2十二层的单层结果仍使用原RTX4060结果，没有重跑或改设置；本表仅更新用于比较的RQ1全层观测点。乘积在每次图片配对重采样后重算，它是独立性假设参照，不是实测联合干预或误差的因果分解。

## 5. 和预期对照

同硬件复算用于核对RQ1忠实性结论，不改变RQ3空间定位的独立证据。短预算的全层忠实性有限、单层差异温和，因此结合既有RQ2，仍最接近预案C，但不是“部分层可读、部分层完全不可读”的强分化。应结合RQ3 v2中前景干预的重要性、汇聚点干预的负结果和寄存器对照讨论整体研究叙事，不能用RQ1高一致率直接推导热图是物体解释。

## 6. 可信度检查

两次k=197在5000张上均为100%；原模型准确率为 {(identity.original == identity.target).mean():.2%}。两次与此前结果的图片ID、标签、原预测均逐项一致；42组各5000条，无重复/缺失，重算一致率/CI/KL通过。远程pytest与环境记录见 `2026-10-27_colab_environment_retry`；此前A100只完成环境验收和部分传输，未进入正式推理。

## 7. 异常与可能原因

跨GPU若出现随机结果变化，首先考虑CUDA随机采样实现与算术路径差异；验证方法是事先固定并导出读取集合，再跨GPU执行同一集合，此项未运行。独立乘积与真实全层结果的差距可能来自图片难度相关、错误重叠和层间补偿；若要区分，需要另行设计联合干预，当前乘积不能唯一识别机制。

## 8. 下一步选项

1. 推荐审阅后将本次完整同硬件曲线作为论文图2；只更新图与来源说明，约30分钟，减少计算来源对读者判断的干扰。
2. 沿用原图，在附录放本次复核；约20分钟，保留原叙事结构并提供稳健性证据。
3. 讨论RQ3图4的“成分—因果—寄存器”布局；约30–60分钟，突出空间峰值与功能重要性并不等价。

## 9. 可写进论文的句子

“在同一张{metadata['device']}上使用固定5000张ImageNet图片复算全部读取预算后，k=128时网格与K-means说明书分别保持 {grid128.agree:.2%} 和 {km128.agree:.2%} 的原模型预测；随机读取集合为 {random128.agree:.2%}。”

“十二个单层一致率的独立乘积低于实际全层一致率，表明简单的独立失败累积模型不足以解释多层替换损失；该参照不构成误差来源的因果分解。”
'''
    destination = ROOT / 'docs/analysis/rq1_colab_verification.md'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
