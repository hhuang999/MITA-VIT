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


def extension():
    composition = pd.read_csv(D / 'B1/composition.csv')
    heat = pd.read_csv(D / 'B2/heatmaps.csv')
    causal = pd.read_csv(EXP / '2026-10-26_rq3_causal/B3/causal_summary.csv')
    contrasts = pd.read_csv(EXP / '2026-10-26_rq3_causal/B3/causal_contrasts.csv')
    stratified = pd.read_csv(D / 'B4/stratified_heatmaps.csv')
    registers = pd.read_csv(EXP / '2026-10-26_rq3_register_comparison/heatmaps.csv')
    register_contrasts = pd.read_csv(EXP / '2026-10-26_rq3_register_comparison/heatmap_contrasts.csv')
    text = '''# RQ3 v2：读取位置、预测作用与寄存器对照

## 1. 一句话结论

原 RQ3 的背景峰值与预注册高范数 token 高度重合，但这些位置未显示出更强的预测重要性：k16全层共同样本中，基准一致率73.02% [71.52,74.48]，去汇聚点73.07% [71.69,74.59]，去前景仅7.93% [7.03,8.84]。读取频繁、空间定位与保持预测是不同的性质。

## 2. 这个实验回答什么问题

本轮是在设计书 §6.4 之上、用户明确授权的 A1–A5/B1–B5 扩展：说明书看到了什么、是否真的使用、是否符合寄存器缓解高范数 patch 的机制假设。原 RQ1/RQ2 与原 RQ3 的 CSV、设置和结论原文保留；诊断没有发现需要改写原 RQ3 的方向/对齐 bug，见 [诊断报告](rq3_diagnosis.md)。

## 3. 设置

Pet test 全部 3,669 张；网格 m=16，主 k=16、稳健性 k=32。DeiT-S 的 resize248/center crop224、mask 最近邻、patch 非边界前景像素比例≥0.5保持原样。汇聚点为每图每层 block 输入（norm1前）patch L2 范数严格大于中位数3倍；2/5倍敏感性结果均保留。规则在 a03c99c 的 E01–E07 中运行前固定。

B1 三类空间 token 互斥（汇聚点优先），unknown/CLS/register 单列，以全部读取次数为分母。空间热图排除 prefix 和 unknown。B2 主 selection 排除最后四层汇聚点并集，CLS 排除最后层汇聚点；这改变了空间评价的候选位置，不改变模型输出，也不使改造后的热图自动拥有执行忠实性。

B3 从原模型固定区域打分、读取集合和路由，只在9–12层或全部12层执行。删除目标后依原始排名补入非目标、不重复的 token，保持 k。前景干预包括前景汇聚点，背景干预不含汇聚点；随机对照在每个集合随机删除相同数量（seed0–4），补选原集合外最高分位置。某类非目标位置不足 k 时按预注册记不可执行，绝不改变 k。报告完整基准、各自可执行样本及跨类别共同样本；五个随机种子先逐图平均。这里的因果效应是相对固定预算执行基准的额外变化，不等同于直接消融原模型全部注意力。

B4 使用变换后前景像素质心，距中心除以224，三分位阈值0.073663/0.129310，样本数1221/1220/1221；7张裁剪后无前景，不分组。分组不依模型或热图。

B5 使用 timm DINOv2-S/14 与 reg4，224输入、16×16 patch、相同 resize248/crop224/归一化，observe-only。CLS与4个register不参与区域池化，但可以被选为key并分别计数。checkpoint revision固定在配置与metadata。两个模型是分别训练的权重，因此对照检验一致性，不能表述为在同一模型权重上只开关寄存器。[无寄存器模型](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)、[reg4模型](https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m)、[寄存器原论文](https://arxiv.org/abs/2309.16588)。

所有区间均为图片 bootstrap 1,000次、seed0、95%，差值同步配对图片；不把层/head/区域/随机种子当独立图片。各图独立运行于本地RTX4060、FP32、batch32；HPC未重启。

## 4. 结果

### B1 哪些层开始改变读取成分？

![逐层成分与定位](../../experiments/2026-10-26_rq3_diagnosis/B1/layer_composition.png)

第5层开始检出高范数 patch，第6层开始普遍出现；k16 的 pointing 同时明显下降。第9层总体前景质量较高，但峰值被少数高范数位置主导；第12层普通背景读取重新增加。以下成分表分母包含CLS，因而与排除CLS后的前景质量不同。

需区分时间定义：最后四层平均图峰值有98.66% [98.31,99.02]落入四层汇聚点并集，但第12层单独图只有5.26% [4.55,5.94]的峰值是同层汇聚点。部分位置在更早层高范数、到最后层不再超过阈值；不能把所有背景峰值都统称为同层汇聚点。

'''
    for k in [16, 32]:
        f = composition[(composition.k == k) & (composition.threshold == 3)]
        text += f'\n#### k={k}，主阈值逐层表（均为 % [95% CI]）\n\n' + table(
            ['层', '前景非汇聚', '普通背景', '汇聚点', 'CLS', 'unknown', '前景质量', 'pointing'],
            [[l] + [estimate(f[(f.layer == l) & (f.metric == metric)].iloc[0])
                    for metric in ['foreground', 'background', 'sink', 'cls', 'unknown', 'fg_mass', 'pointing']]
             for l in range(1, 13)])
    text += '''
2/5倍阈值完整结果见 [composition.csv](../../experiments/2026-10-26_rq3_diagnosis/B1/composition.csv)。它们只影响汇聚点分类，不改变原始读取集合。

### B2 汇聚点是否遮住了物体？

主设置去汇聚点后的 selection pointing 为42.71% [41.13,44.26]，高于原图18.21% [16.95,19.38]，但仍低于随机50.94% [49.30,52.58]。k32达到61.22% [59.61,62.61]，说明读取预算会影响去除强峰后留下的空间信息。不能据此将主设置换成k32。

'''
    for t in [3, 2, 5]:
        text += f'\n#### 阈值 {t} 倍；% [95% CI]\n\n' + table(
            ['k', '方法', '前景质量', 'pointing', 'n'],
            [[k, method, estimate(heat[(heat.threshold == t) & (heat.k == k) & (heat.method == method) & (heat.metric == 'fg_mass')].iloc[0]),
              estimate(heat[(heat.threshold == t) & (heat.k == k) & (heat.method == method) & (heat.metric == 'pointing')].iloc[0]), 3669]
             for k in [16, 32] for method in ['selection', 'selection_no_sink', 'cls', 'cls_no_sink', 'gaussian', 'random']])
    text += '''
### B3 哪类读取对保持预测最重要？

![固定预算因果检验](../../figures/rq3_causal_2026-10-26.png)

柱图使用跨干预共同可执行图片。下面同时列“各自可执行”与“共同样本”，一致率参照为原模型在同一Pet图片上的ImageNet类别预测，不把Pet品种标签当1000类准确率。

'''
    for support in ['available', 'common']:
        text += f'\n#### {support} 样本\n\n' + table(
            ['k', '范围', '设置', 'n', '一致率 % [95% CI]', '相对基准下降 pp [95% CI]', '每集合删除数 [95% CI]'],
            [[k, scope, method, int((r := causal[(causal.k == k) & (causal.scope == scope) & (causal.method == method) & (causal.support == support) & (causal.metric == 'agreement')].iloc[0]).n), estimate(r),
              estimate(causal[(causal.k == k) & (causal.scope == scope) & (causal.method == method) & (causal.support == support) & (causal.metric == 'drop_from_baseline')].iloc[0]),
              estimate(causal[(causal.k == k) & (causal.scope == scope) & (causal.method == method) & (causal.support == support) & (causal.metric == 'removed_per_set')].iloc[0], False)]
             for k in [16, 32] for scope in ['last4', 'all12'] for method in ['baseline', 'sink', 'foreground', 'background', 'random']])
    text += '''
在四个 (k,范围) 设置的共同可执行样本上，删除前景造成最大下降：k16最后四层下降47.10pp [45.13,48.99]，全层下降65.08pp [63.31,66.71]；k32分别下降42.51pp [40.64,44.36]和69.26pp [67.59,70.99]。汇聚点删除没有降低一致率的明确证据；k16最后四层反而提高1.21pp [0.48,1.94]。这不等于原模型从不使用高范数信息，只说明在当前冻结读取集合和补选规则下，它们不是不可替代的读取位置。

普通背景删除在k16最后四层提高5.03pp [3.96,6.05]，而k32全层下降1.70pp [0.36,2.92]，因此不能笼统宣称“背景完全无用”。

随机一致率 − 去汇聚点一致率，正数表示去汇聚点比等数量随机删除更有破坏性：

'''
    text += table(['k', '范围', '配对差值 pp [95% CI]'],
                  [[r.k, r.scope, estimate(r)] for _, r in contrasts.iterrows()])
    text += '\n随机五种子逐一结果（% [95% CI]）：\n\n' + table(
        ['k', '范围', '种子', '一致率'], [[r.k, r.scope, r.method, estimate(r)]
        for _, r in causal[causal.method.str.startswith('random_seed')].iterrows()])
    text += '''
以上“下降”允许为负：某种替换也可能比执行基准更接近原模型。前景、背景和汇聚点删除剂量不同，因此跨类别比较是该操作的总体影响，不能直接换算成每个token的固有重要性；只有汇聚点与随机对照明确匹配删除数量。

### B4 中心偏置控制

最偏心三分之一中，高斯的 pointing 仍领先；因此不能仅用“Pet 都在中心”解释原热图低于机会水平。质心偏心不保证物体避开中心，大物体可能质心偏移但仍覆盖中央token。

'''
    for group in ['centered', 'middle', 'offcenter']:
        f = stratified[stratified.group == group]
        text += f'\n#### {group}\n\n' + table(['k', '方法', 'n', '前景质量 % [95% CI]', 'pointing % [95% CI]'],
            [[k, method, int((r := f[(f.k == k) & (f.method == method) & (f.metric == 'fg_mass')].iloc[0]).n), estimate(r),
              estimate(f[(f.k == k) & (f.method == method) & (f.metric == 'pointing')].iloc[0])]
             for k in [16, 32] for method in ['selection', 'selection_no_sink', 'cls', 'cls_no_sink', 'gaussian', 'random']])
    text += '''
### B5 有无寄存器的模型对照

![两个DINOv2模型](../../experiments/2026-10-26_rq3_register_comparison/register_comparison.png)

在主阈值下，无寄存器模型最后四层任一层出现高范数patch的图片比例为12.46% [11.36,13.55]，reg4为0% [0,0]。这里的零是本测试集上的经验值，普通bootstrap在零事件时也给零宽区间，不能据此断言总体发生率严格为零。reg4的浅层仍可能检出高范数patch，不能称整个模型没有高范数token。

k16选中次数图pointing从60.94% [59.42,62.55]升至78.11% [76.75,79.39]，配对增加17.17pp [15.64,18.81]；前景质量增加3.12pp [3.06,3.19]。k32的pointing也提高17.61pp [16.00,19.24]。这些观察与寄存器缓解背景patch强峰的假设一致，但不能把分别训练的checkpoint差异全部归因为寄存器本身。

第9层k16四个寄存器合计占读取次数18.12% [18.08,18.16]，第12层8.76% [8.75,8.76]；它们确实占据读取预算。但预算k包含prefix，因此相同k并不意味着两模型具有相同的有效patch读取数量，这也是该架构对照的一部分。主selection的pointing仍低于中心高斯92.23% [91.44,93.10]；CLS的pointing接近97%，与读取集合热图的含义不同。

'''
    text += table(['模型', 'k', '方法', '前景质量 % [95% CI]', 'pointing % [95% CI]'],
        [[model, k, method, estimate(registers[(registers.model == model) & (registers.k == k) & (registers.method == method) & (registers.metric == 'fg_mass')].iloc[0]),
          estimate(registers[(registers.model == model) & (registers.k == k) & (registers.method == method) & (registers.metric == 'pointing')].iloc[0])]
         for model in ['dino', 'register'] for k in [16, 32] for method in ['selection', 'cls', 'gaussian', 'random']])
    text += '\n寄存器模型 − 无寄存器模型的配对差值：\n\n' + table(
        ['k', '方法', '指标', '差值 pp [95% CI]'],
        [[r.k, r.method, r.metric, estimate(r)] for _, r in register_contrasts.iterrows()])
    text += '''
每个模型的A2位置前十/频率、A3逐层范数、A4三阈值和B1逐层成分分别保存在 [DINOv2](../../experiments/2026-10-26_rq3_dino/) 与 [reg4](../../experiments/2026-10-26_rq3_register/)。寄存器按1–4分别计数，总寄存器读取比例也用逐图之和重新bootstrap。224分辨率为用户指定，checkpoint默认518的绝对位置嵌入由timm插值；不得把此处与DeiT不同patch网格的空间指标差异全归因于训练方式。

## 5. 和预期对照

“最强位置等于物体位置”的预期被否定；高范数位置可以频繁被读取，同时空间定位较差。B2只恢复一部分前景峰值，B3/B5的解释必须依据实际对照。RQ3扩展本身不重新定义设计书A/B/C，它深化空间与功能的区分，原RQ1/RQ2分型保留。不能为了更完整的故事把所有高范数位置都称为已经证实的全局信息汇聚器。

## 6. 可信度检查

44项测试通过；DeiT采集observe误差0，32张B3前置k197恒等误差0；ImageNet此前200/5000张恒等结果保留。正式B3每批输出有限，并检查全量样本数、相同参照预测、缺失与可执行规则一致、随机删除数量匹配。DINOv2只报告observe特征不变，不把384维特征argmax解释成类别。

A的逐图指标与原RQ3最大差异1.84×10⁻⁷；五张叠图视觉核对通过。抽样CI用图片为单位，随机种子均值的CI反映图片抽样不确定性，不包括训练种子或checkpoint选择不确定性。未作多重比较校正。大矩阵缓存与权重留本地ignored目录，CSV/配置/日志/图已提交；源commit、checkpoint revision与缓存SHA256记录在各目录。

## 7. 异常与可能原因

| 观察/限制 | 可能原因 | 如何验证 |
| --- | --- | --- |
| 转置提高pointing但降低selection质量 | 热点位置与物体构图不均匀，单一分数不足以判断方向 | 对照已保存的五张原图/标签/热图及patch行优先索引；本轮未发现方向bug |
| 去汇聚点后主设置仍低于随机pointing | 普通背景强峰仍存在；k16覆盖不足 | 比较B1最后层普通背景、B2两k与B3类别干预；下一步可按层看干预，不在本轮追加 |
| 少数B3干预不可执行 | 非目标token不足k，固定预算约束无法同时满足 | 查看coverage.csv与逐图reason，复算对应mask剩余token数量 |
| B4有7张无质心 | 原eval裁剪后前景全部消失 | 查看groups.csv中group=-1的原始trimap与224裁剪；不补造质心 |
| 不同模型对照不等于纯register开关 | checkpoint训练权重不同，224位置插值与原始训练分辨率不同 | 若需严格训练因果结论，应做相同训练设置的受控模型实验，超出本轮 |

前置代码测试发现stable argsort参数调用错误，正式实验前已修复并通过测试。一次分析过早启动，等待采集结束后重试；没有丢弃负结果、改阈值或重选示例。

## 8. 下一步选项与图4版面

1. **推荐：图4采用“机制证据”三面板。** 左：逐层前景/普通背景/汇聚点成分；中：主k16的B3两执行范围及等数量随机对照；右：DINOv2两模型的空间指标与prefix读取分配。完整k32、敏感性、位置/范数图放附录。目的：在4页内把“看到了什么→是否有用→跨模型一致性”连起来；排版约1–2小时，不新增实验。
2. **图4采用“图像案例”版。** 上：固定三例的原图、原热图、去汇聚点图；下：B3主结果和偏心组三方法小表。DINOv2放附录。目的：优先让读者直观看懂定位与功能不同；约1–2小时，代价是主文的寄存器证据较少。

需要用户决定的是采用哪种主文叙事和版面，以及是否将上述干预约束作为正式方法定义。未改研究叙事稿的【待填】。

## 9. 可以写进论文的句子与三个问题

> 在Oxford-IIIT Pet上，原读取热图的背景强峰与高范数patch显著重合：98.66% [98.31,99.02]的峰值属于最后四层高范数位置的并集。移除这些位置使k16的pointing从18.21% [16.95,19.38]升至42.71% [41.13,44.26]，但仍不足以恢复可靠的物体定位。

> 在保持读取预算不变的执行检验中，移除前景导致k16全层一致率下降65.08pp [63.31,66.71]，而移除高范数位置未带来明确下降；等数量随机对照也未显示这些位置更重要。DINOv2寄存器模型的读取热图pointing比无寄存器模型提高17.17pp [15.64,18.81]，与高范数patch现象减弱相伴。这些结果说明，读取热图可以暴露模型内部的选择偏好，但强峰不是功能重要性的直接度量。

**说明书看到了什么？** 中间层逐渐更多读取前景；少量高范数patch从第5–6层开始突出，并主导跨层平均热图峰值；最后层普通背景读取又增加。读取集合的整体前景覆盖和最高点定位因此可以给出不同结论。

**看到的东西真的被用到了吗？** 在本轮冻结说明书、保持k并按排名补选的检验里，删除前景的影响最大，主k16全层从73.02%降到7.93%（完整CI见B3）；删除高范数位置后的73.07%与基准接近，也不比匹配数量的随机删除更有破坏性。这支持前景读取的重要性，反对把读取频率直接当作不可替代性；结论限于该干预，不否定原模型存在信息混合或冗余。

**为什么会这样？** 空间位置是token的地址，深层内容已经混合，区域平均打分/重复选择也不等于实际query对V的贡献。寄存器模型出现更少的深层高范数patch及更好的空间定位，提供与“非空间计算可由额外token承载”一致的证据；但B3没有证明DeiT的高范数patch是预测必需的草稿纸，B5也不是同权重的纯开关实验。当前能支持的是位置、读取频率与执行作用需要分开测量。
'''
    (ROOT / 'docs/analysis/rq3_v2.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    import sys
    if '--final' in sys.argv:
        extension()
    else:
        diagnosis()
