"""Render Chinese RQ3 extension reports from saved result tables."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
D = EXP / '2026-10-26_rq3_diagnosis'


def estimate(row, percent=True):
    scale, unit = (100, '%') if percent else (1, '')
    return f"{row['mean']*scale:.2f}{unit} [{row['low']*scale:.2f}, {row['high']*scale:.2f}]"


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def diagnosis():
    a = D / 'A'
    orientation = pd.read_csv(a / 'orientation.csv')
    sink = pd.read_csv(a / 'sink_summary.csv', dtype={'layer': str})
    heat = pd.read_csv(a / 'heatmaps.csv')
    peaks = pd.read_csv(a / 'peak_top10.csv')
    text = '''# RQ3 诊断：背景峰值是 bug 还是高范数 token？

## 结论

**现有证据支持高范数 token 现象，没有发现方向、空间对齐或指标实现 bug。** 主 k=16 的第 9–12 层选中次数图，最高点属于这四层高范数 token 并集的比例为 **98.66% [98.31, 99.02]**。这个结果确认了事先定义的“汇聚点”与热图强峰重合，但其是否承担重要计算，要由 B3 检验，不能由范数命名直接推出。

全部区间为图片 bootstrap 1,000 次、seed=0、95%；全量 Pet test 3,669 张。定义在提交 a03c99c 中事先登记，主阈值严格大于每图每层 patch 范数中位数的 3 倍；2 倍和 5 倍为完整敏感性检查。范数取 block 输入、norm1 之前。模型/预处理/图像顺序均沿用原 RQ3，网格 m=16，k16/32；本地 RTX4060，FP32。

## A1 方向与叠图

重新采集的原方向指标与原 RQ3 逐图对比，最大绝对差异 1.84×10⁻⁷，差异来自 CPU/GPU 浮点归约；pointing 完全一致。Observe 与原模型输出的最大误差为 0。检查五张预先 seed=0 固定图像（2335、1874、989、1129、3117），image/mask/token grid 的几何和行优先方向一致。

转置只是诊断，不修改原标签。转置后 selection 的 pointing 增加，但仍远低于机会水平；selection 的前景质量反而下降。转置提高一个分数并不证明转置正确，尤其是四角/边缘热点与 Pet 形状存在系统性位置关系时。

'''
    text += table(['k', '方法', '转置标签', '指标', '估计 [95% CI]'],
                  [[r.k, r.method, r.transposed, r.metric, estimate(r)] for _, r in orientation.iterrows()])
    text += '\n![五张固定叠图](../../experiments/2026-10-26_rq3_diagnosis/A/five_overlays.png)\n'
    text += '\n## A2 最高点位置\n\n最高点并非均匀分布，两个方法都常落在靠近四角的位置。坐标从 0 开始，token = row×14+col；并列最大值按原规则取首个有效位置。\n'
    for k in [16, 32]:
        for method in ['selection', 'cls']:
            f = peaks[(peaks.k == k) & (peaks.method == method)]
            text += f'\n### {method}, k={k} 的前十位置\n\n' + table(
                ['排名', '(row,col)', 'token', '频率 [95% CI]'],
                [[r['rank'], f'({int(r.row)},{int(r.col)})', int(r.token), estimate(r)] for _, r in f.iterrows()])
    text += '\n![峰值频率](../../experiments/2026-10-26_rq3_diagnosis/A/peak_locations.png)\n'
    text += '''
## A3 范数分布

![逐层 block 输入范数](../../experiments/2026-10-26_rq3_diagnosis/A/norm_histograms.png)

主峰之外，中后层出现稀疏的高范数尾部/小峰；不能把图描述为两个规模相当的明显峰。这里没有进行形式化双峰检验。固定对数分箱覆盖所有观测值（无超界 token），图按层展示每图 patch 比例的均值与区间，完整分箱见 norm_histogram.csv。是否满足汇聚点定义用预注册阈值判断，不依直方图形状改阈值。

## A4 汇聚点出现的层与峰值对应

下面的峰值对应使用各层自己的选中次数图和同层汇聚点定义。最后四层平均图单独用四层并集；它和“最后一层是否为汇聚点”是不同问题。

'''
    for threshold in [3, 2, 5]:
        f = sink[(sink.threshold == threshold) & (sink.k == 16) & (sink.method == 'selection')]
        text += f'\n### 阈值 {threshold} 倍中位数\n\n' + table(
            ['层', '每图数量 [95% CI]', '有汇聚点的图片比例 [95% CI]', '最高点属于汇聚点 [95% CI]'],
            [[layer, estimate(f[(f.layer == layer) & (f.metric == 'count')].iloc[0], False),
              estimate(f[(f.layer == layer) & (f.metric == 'has_sink')].iloc[0]),
              estimate(f[(f.layer == layer) & (f.metric == 'peak_is_sink')].iloc[0])]
             for layer in list(map(str, range(1, 13))) + ['9-12']])
    text += '''
主阈值在第 1–4 层没有检出，第 5 层开始出现，第 6 层迅速普及。最后四层任一层有汇聚点的图片比例为 **99.84% [99.70,99.95]**，超过预定 10% 门槛，因此执行 B2。最后层 CLS 峰值属于同层汇聚点的比例为 **58.38% [56.77,59.88]**，低于跨层 selection 的并集对应比例；不可混用这两种定义。

k=32 的全部逐层峰值对应及三阈值结果保存在 [sink_summary.csv](../../experiments/2026-10-26_rq3_diagnosis/A/sink_summary.csv)，数量/有无汇聚点不依 k，峰值对应依 k。

## A5 随机机会水平

每图每 token 独立 Uniform(0,1)，NumPy seed=0，两个 k 共用；不是所有分数相同的平坦图。理论期望为有效前景面积占比 **51.18% [50.51,51.83]**。随机实现与理论接近，原热图的 pointing 明显更低。

'''
    text += table(['k', '方法', '指标', '估计 [95% CI]'],
                  [[r.k, r.method, r.metric, estimate(r)] for _, r in heat.iterrows()])
    text += '''
## 可信度与下一步

44 项测试通过，包括 pre-norm 取值、寄存器/CLS 排除池化、observe 不变、固定 k 补选、随机删除剂量。原实现的几何对齐、指标和 observe 输出均复验。没有证据需要修复并重跑原 RQ3，因此原 rq3.md 与原结果不改写。

本诊断回答的是“高范数强峰存在”，还没有证明“高范数位置保存了什么信息”。按已授权流程继续 B1–B5，尤其用 B3 的类别删除与等数量随机删除区分读取频繁与预测重要性。
'''
    (ROOT / 'docs/analysis/rq3_diagnosis.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    diagnosis()
