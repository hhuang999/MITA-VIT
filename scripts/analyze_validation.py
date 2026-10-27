"""Audit and analyze approved V01-V03 results using paired image bootstrap."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mita_probe.evaluate import compare_runs
from mita_probe.io import source_commit, write_json
from mita_probe.rq3_extension import bootstrap
from mita_probe.validation import layer_settings

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
RUNS = {stage: EXP / f'2026-10-27_{stage}_colab' for stage in ['v01', 'v02', 'v03']}


def fmt(row, scale=100):
    precision = 4 if scale == 1 else 2
    return (f'{scale*row["mean"]:.{precision}f} '
            f'[{scale*row["low"]:.{precision}f}, {scale*row["high"]:.{precision}f}]')


def table(frame, keys, metrics):
    result = '| ' + ' | '.join(keys + list(metrics)) + ' |\n'
    result += '| ' + ' | '.join(['---']*(len(keys)+len(metrics))) + ' |\n'
    for key, part in frame.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        result += '| ' + ' | '.join(map(str, key))
        for metric, scale in metrics.items():
            row = part[part.metric == metric].iloc[0]
            result += ' | ' + fmt(row, scale)
        result += ' |\n'
    return result


def save_figure(fig, output, name):
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        path = output / f'{name}.{suffix}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def audit_run(stage):
    run = RUNS[stage]
    frame = pd.read_csv(run/'predictions.csv', dtype={
        'scope':'string', 'category':'string', 'reason':'string'}).fillna({
        'scope':'', 'category':'', 'reason':''})
    identity = pd.read_csv(run/'identity_predictions.csv').sort_values('index').set_index('index')
    n = 3669 if stage == 'v01' else 5000
    expected = {'v01':76, 'v02':57, 'v03':3}[stage]
    keys = ['k', 'setting', 'scope', 'category', 'seed']
    assert len(frame) == n*expected and frame.groupby(keys).ngroups == expected
    assert not frame.duplicated(keys+['index']).any()
    assert len(identity) == n and identity.agree.eq(1).all()
    for _, group in frame.groupby(keys):
        ordered = group.sort_values('index').set_index('index')
        assert ordered.index.equals(identity.index)
        assert ordered.original.equals(identity.original)
        assert ordered.target.equals(identity.target)
        valid = group[group.eligible == 1]
        assert valid.agree.eq(valid.executed.eq(valid.original).astype(float)).all()
        assert np.isfinite(valid.kl).all() and valid.kl.min() > -1e-9
        assert group[group.eligible == 0].agree.isna().all()
    check = json.loads((run/'checks.json').read_text())
    assert check['identity_agreement'] == 1 and check['max_observe_error'] == 0
    assert check['max_identity_error'] < 1e-5 and check['settings_completed'] == expected
    return frame, identity, check


def analyze_v01(frame, identity, output):
    previous = pd.read_csv(EXP/'2026-10-26_rq3_causal/predictions.csv')
    old_original = previous.groupby('index').original.first().sort_index()
    assert identity.original.equals(old_original)
    stats, seed_stats, contrasts, coverage, anchors, dispersion = [], [], [], [], [], []
    for (k, scope, setting), part in frame[frame.setting != 'random_patch'].groupby(['k','scope','setting']):
        old = previous[(previous.k==k)&(previous.scope==scope)&(previous.method==setting)]
        ids = set(part.loc[part.eligible==1,'index']) & set(old.loc[old.eligible==1,'index'])
        anchors.append(dict(k=k,scope=scope,setting=setting,
            **compare_runs(part[part['index'].isin(ids)],old[old['index'].isin(ids)])))
    for k in [16, 32]:
        for scope in ['last4', 'all12']:
            part = frame[(frame.k == k) & (frame.scope == scope)]
            base = part[part.setting == 'baseline'].set_index('index').sort_index()
            targets = {c:part[part.setting == c].set_index('index').sort_index()
                       for c in ['sink', 'foreground', 'background']}
            common = np.logical_and.reduce([t.eligible.eq(1).to_numpy() for t in targets.values()])
            for category, target in targets.items():
                random = part[(part.setting == 'random_patch') & (part.category == category)]
                assert set(random.seed) == set(range(5))
                for _, group in random.groupby('seed'):
                    np.testing.assert_allclose(group.set_index('index').sort_index().mean_removed_per_set,
                                               target.mean_removed_per_set, atol=1e-7, rtol=0)
                random_mean = random.groupby('index')[['agree', 'kl']].mean().sort_index()
                for population, mask in [('available', target.eligible.eq(1).to_numpy()), ('common', common)]:
                    meta = dict(k=k, scope=scope, category=category, population=population)
                    coverage.append({**meta, 'n':int(mask.sum()), 'excluded':int((~mask).sum())})
                    for method, values in [('baseline',base), ('target',target), ('random_patch',random_mean)]:
                        for metric in ['agree', 'kl']:
                            stats.append({**meta, 'method':method, 'metric':metric,
                                          **bootstrap(values[metric].to_numpy()[mask])})
                    for metric, values in {
                        'target_drop':base.agree.to_numpy()-target.agree.to_numpy(),
                        'random_drop':base.agree.to_numpy()-random_mean.agree.to_numpy(),
                        'extra_drop_vs_random':random_mean.agree.to_numpy()-target.agree.to_numpy(),
                        'dose':target.mean_removed_per_set.to_numpy(),
                    }.items():
                        contrasts.append({**meta, 'metric':metric, **bootstrap(values[mask])})
                    for seed, group in random.groupby('seed'):
                        seed_stats.append({**meta, 'seed':int(seed),
                            **bootstrap(group.set_index('index').sort_index().agree.to_numpy()[mask])})
                    seed_means = [group.set_index('index').sort_index().agree.to_numpy()[mask].mean()
                                  for _, group in random.groupby('seed')]
                    dispersion.append({**meta, 'n_seeds':5,
                                       'seed_std':float(np.std(seed_means,ddof=1))})
    stats, contrasts = pd.DataFrame(stats), pd.DataFrame(contrasts)
    stats.to_csv(output/'summary.csv', index=False)
    contrasts.to_csv(output/'paired_contrasts.csv', index=False)
    pd.DataFrame(coverage).to_csv(output/'coverage.csv', index=False)
    pd.DataFrame(seed_stats).to_csv(output/'seed_results.csv', index=False)
    pd.DataFrame(dispersion).to_csv(output/'seed_dispersion.csv', index=False)
    pd.DataFrame(anchors).to_csv(output/'anchor_differences.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, scope in zip(axes, ['last4', 'all12']):
        for j, k in enumerate([16,32]):
            p = contrasts[(contrasts.k == k) & (contrasts.scope == scope) &
                          (contrasts.population == 'common') & (contrasts.metric == 'extra_drop_vs_random')]
            p = p.set_index('category').loc[['sink','foreground','background']]
            y = p['mean'].to_numpy()*100
            ax.bar(np.arange(3)+(j-.5)*.35, y, .35, label=f'k={k}',
                   yerr=np.vstack([y-p.low.to_numpy()*100,p.high.to_numpy()*100-y]), capsize=3)
        ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(range(3), ['High-norm','Foreground','Background'])
        ax.set_title(scope)
        ax.legend()
    axes[0].set_ylabel('Extra agreement loss vs matched random (pp)')
    fig.suptitle('V01 | Patch-only matched-dose random controls | 95% paired CI')
    save_figure(fig,output,'matched_controls')
    main = contrasts[(contrasts.k==16)&(contrasts.scope=='all12')&(contrasts.population=='common')&
                     (contrasts.metric=='extra_drop_vs_random')].set_index('category')
    conclusion = ('k16全层共同样本中，目标删除相对匹配随机删除的额外一致率下降：'
                  + '；'.join(f'{c} {fmt(main.loc[c])} pp' for c in main.index) + '。正值代表目标删除更有破坏性。')
    body = '![匹配对照](../../experiments/2026-10-27_v01_colab/analysis/matched_controls.png)\n\n'
    body += table(contrasts, ['k','scope','category','population'],
                  {'target_drop':100,'random_drop':100,'extra_drop_vs_random':100,'dose':1})
    body += ('\n完整基准、目标、随机的绝对一致率与KL区间见summary.csv；逐种子见seed_results.csv，'
             '种子标准差见seed_dispersion.csv，覆盖见coverage.csv。新旧基准/类别干预在共同可执行图片上的配对核对见anchor_differences.csv。\n')
    return conclusion, body, ('CLS只禁止删除；原集合外的CLS仍可依排名补入。随机与类别补选的候选规则不同，'
        '结果是本次固定预算替换干预的效应，不能当作单token固有重要性或原模型直接消融效应。')


def analyze_v02(frame, identity, output):
    reference = pd.read_csv(EXP/'2026-10-27_rq1_colab/identity_predictions.csv').sort_values('index').set_index('index')
    pd.testing.assert_frame_equal(identity, reference)
    old = pd.read_csv(EXP/'2026-10-26_rq2_imagenet/predictions.csv')
    rq1 = pd.read_csv(EXP/'2026-10-27_rq1_colab/predictions.csv')
    rows, contrasts, products = [], [], []
    for (k, setting), part in frame.groupby(['k','setting']):
        for metric in ['agree','kl']:
            rows.append(dict(k=k,setting=setting,metric=metric,**bootstrap(part[metric])))
        all_layers = frame[(frame.k==k)&(frame.setting=='all12')]
        if setting != 'all12':
            contrasts.append(dict(k=k,setting=setting,comparison='minus_all12',**compare_runs(part,all_layers)))
        if setting.startswith('layer_') and k in [8,32]:
            layer = int(setting.split('_')[1])
            contrasts.append(dict(k=k,setting=setting,comparison='minus_old_4060',
                **compare_runs(part,old[(old.k==k)&(old.layer==layer)])))
        if setting == 'all12':
            contrasts.append(dict(k=k,setting=setting,comparison='minus_previous_L4',
                **compare_runs(part,rq1[(rq1.k==k)&(rq1.method=='grid')])))
    for k in [8,32,64]:
        for depth in [3,6,9]:
            contrasts.append(dict(k=k,setting=f'depth_{depth}',comparison='prefix_minus_suffix',
                **compare_runs(frame[(frame.k==k)&(frame.setting==f'prefix_{depth}')],
                               frame[(frame.k==k)&(frame.setting==f'suffix_{depth}')])) )
        single = frame[(frame.k==k)&frame.setting.str.startswith('layer_')].pivot(index='index',columns='setting',values='agree').sort_index()
        joint = frame[(frame.k==k)&(frame.setting=='all12')].set_index('index').sort_index().agree
        assert single.shape==(5000,12) and single.index.equals(joint.index)
        x,y=single.to_numpy(),joint.to_numpy()
        rng=np.random.default_rng(0)
        samples=[]
        for _ in range(1000):
            idx=rng.integers(len(y),size=len(y))
            p=x[idx].mean(0).prod()
            samples.append([p,y[idx].mean(),y[idx].mean()-p])
        low,high=np.quantile(samples,[.025,.975],axis=0)
        for j,(metric,value) in enumerate(zip(['independent_product','observed_all12','observed_minus_product'],
                [x.mean(0).prod(),y.mean(),y.mean()-x.mean(0).prod()])):
            products.append(dict(k=k,metric=metric,mean=value,low=low[j],high=high[j],n=5000))
    summary=pd.DataFrame(rows)
    summary.to_csv(output/'summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(output/'paired_contrasts.csv',index=False)
    pd.DataFrame(products).to_csv(output/'independent_product.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,k in zip(axes,[8,32,64]):
        for direction,start in [('prefix',1),('suffix',12)]:
            names=[f'layer_{start:02}',f'{direction}_3',f'{direction}_6',f'{direction}_9','all12']
            p=summary[(summary.k==k)&(summary.metric=='agree')].set_index('setting').loc[names]
            ax.plot([1,3,6,9,12],p['mean']*100,'o-',label=direction)
            ax.fill_between([1,3,6,9,12],p.low*100,p.high*100,alpha=.15)
        ax.set_title(f'k={k}')
        ax.set_xticks([1,3,6,9,12])
        ax.set_xlabel('Number of replaced layers')
        ax.legend()
    axes[0].set_ylabel('Prediction agreement (%)')
    fig.suptitle('V02 | Frozen original summaries | ImageNet 5,000 | 95% CI')
    save_figure(fig,output,'layer_paths')
    p=summary[(summary.k==64)&(summary.metric=='agree')&summary.setting.str.startswith('layer_')]
    conclusion=f'k64单层一致率均值范围为{p["mean"].min():.2%}–{p["mean"].max():.2%}；逐层及多层组合的区间见完整表。'
    body='![层组合路径](../../experiments/2026-10-27_v02_colab/analysis/layer_paths.png)\n\n'
    body+=table(summary,['k','setting'],{'agree':100,'kl':1})
    body+='\n### 独立乘积参照\n\n'+table(pd.DataFrame(products),['k'],
        {'independent_product':100,'observed_all12':100,'observed_minus_product':100})
    return conclusion,body,'前缀/后缀是替换层的集合，不改变网络执行顺序；路径差不是各层固定因果贡献。独立乘积仅为参照，不能识别误差机制。'


def analyze_v03(frame, identity, output):
    frozen, reference, _=audit_run('v02')
    pd.testing.assert_frame_equal(identity,reference)
    metadata=[json.loads((RUNS[s]/'metadata.json').read_text()) for s in ['v02','v03']]
    for key in ['torch','timm','device','dtype','checkpoint_revision','git_commit','preprocessing']:
        assert metadata[0][key]==metadata[1][key],key
    equivalence=pd.read_csv(RUNS['v03']/'equivalence.csv')
    assert equivalence.groupby(['k','layer']).n.sum().eq(200).all()
    assert equivalence.groupby(['k','layer']).ngroups==36 and equivalence.max_error.eq(0).all()
    rows,contrasts,kl_contrasts=[],[],[]
    for k in [8,32,64]:
        online=frame[frame.k==k]
        base=frozen[(frozen.k==k)&(frozen.setting=='all12')]
        contrasts.append(dict(k=k,**compare_runs(online,base)))
        kl_contrasts.append(dict(k=k,**compare_runs(online,base,metric='kl')))
        for method,part in [('frozen',base),('online',online)]:
            for metric in ['agree','kl']:
                rows.append(dict(k=k,method=method,metric=metric,**bootstrap(part[metric])))
    pattern=pd.read_csv(RUNS['v03']/'patterns.csv')
    assert len(pattern)==5000*3*12 and not pattern.duplicated(['index','k','layer']).any()
    assert pattern[['readset_overlap','route_changed']].ge(0).all().all()
    assert pattern[['readset_overlap','route_changed']].le(1).all().all()
    first = pattern[pattern.layer == 1]
    assert first.readset_overlap.eq(1).all() and first.route_changed.eq(0).all()
    pstats=[]
    for (k,layer),part in pattern.groupby(['k','layer']):
        for metric in ['readset_overlap','route_changed']:
            pstats.append(dict(k=k,layer=layer,metric=metric,**bootstrap(part[metric])))
    summary=pd.DataFrame(rows)
    summary.to_csv(output/'summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(output/'paired_contrasts.csv',index=False)
    pd.DataFrame(kl_contrasts).to_csv(output/'paired_kl_contrasts.csv',index=False)
    pd.DataFrame(pstats).to_csv(output/'pattern_changes.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,scale,label in zip(axes,['agree','kl'],[100,1],['Prediction agreement (%)','KL(original || executed)']):
        for method in ['frozen','online']:
            p=summary[(summary.method==method)&(summary.metric==metric)].sort_values('k')
            ax.plot(p.k,p['mean']*scale,'o-',label=method)
            ax.fill_between(p.k,p.low*scale,p.high*scale,alpha=.15)
        ax.set_xscale('log',base=2)
        ax.set_xticks([8,32,64],['8','32','64'])
        ax.set_xlabel('k')
        ax.set_ylabel(label)
        ax.legend()
    fig.suptitle('V03 | ImageNet 5,000 | Frozen vs recomputed summaries | 95% CI')
    save_figure(fig,output,'frozen_online')
    conclusion='现场−冻结一致率差：'+'；'.join(f'k={r["k"]} {100*r["difference"]:.2f} pp [{100*r["ci_low"]:.2f}, {100*r["ci_high"]:.2f}]' for r in contrasts)+'。'
    body='![冻结与现场](../../experiments/2026-10-27_v03_colab/analysis/frozen_online.png)\n\n'
    body+=table(summary,['k','method'],{'agree':100,'kl':1})
    body+='\n### 逐层集合重合与路由变化（% [95% CI]）\n\n'+table(pd.DataFrame(pstats),['k','layer'],{'readset_overlap':100,'route_changed':100})
    return conclusion,body,'现场同时适配集合和路由，回答另一种干预问题；差值不能唯一归因于冻结状态失配，也不据此替换冻结主实验。'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['v01','v02','v03'])
    args=parser.parse_args()
    commit=source_commit(ROOT)
    frame,identity,checks=audit_run(args.stage)
    output=RUNS[args.stage]/'analysis'
    output.mkdir(exist_ok=False)
    conclusion,body,limits=globals()[f'analyze_{args.stage}'](frame,identity,output)
    report=f'''# {args.stage.upper()}：夜间补充验证

## 1. 一句话结论

{conclusion}

## 2. 回答的问题

按已批准的[方案](../overnight_validation_plan_2026-10-27.md)与E09执行，检验现有结论的边界，原实验不覆盖。

## 3. 设置

同一Colab GPU、原固定模型和预处理、m16、FP32、batch32。V01用Pet3669张，V02/V03用原ImageNet5000张。设置和源码提交见对应config.yaml与metadata.json。置信区间为图片bootstrap1000次、seed0、95%；随机先逐图平均五种子。未作多重比较校正。

## 4. 结果

{body}

## 5. 与预期对照

结果以实际对照差值及区间为准；区间包含0不能单独证明等价，强度和方向均保留，不按显著性删组。

## 6. 可信度检查

每设置完整固定样本、无重复图片、相同原模型预测；由预测复算agree，KL有限，缺失与可执行性规则一致。远程先完成完整样本k197验收，observe输出不变。V01运行时逐集合检查删除剂量和CLS保留；V03先通过200张×三k×12单层严格等价，再运行正式设置。完整审计见analysis/checks.json。

## 7. 异常与解释边界

{limits}

## 8. 下一步

与原RQ报告合并讨论，优先把直接支持结论的对照放主文，其余放附录；本报告不授权新的实验。

## 9. 叙事草稿

{conclusion} {limits}
'''
    path=ROOT/f'docs/analysis/{args.stage}_validation.md'
    assert not path.exists()
    path.write_text(report,encoding='utf-8')
    write_json(output/'checks.json',dict(all_passed=True,analysis_commit=commit,
        prediction_rows=len(frame),n_images=len(identity),runtime_checks=checks))
    (output/'log.txt').write_text('Completed structural audit, paired bootstrap, plots and report.\n')
    print(conclusion)


if __name__=='__main__':
    main()
