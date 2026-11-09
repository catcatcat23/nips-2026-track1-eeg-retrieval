"""Matched-image, frozen-head Procrustes diagnostic for fixed LOSO checkpoints.

Uses SCORE (2608.19134v1), equations 6--8, for the regularized orthogonal
solver only. This is supervised EEG-to-EEG alignment, not full label-free SCORE.
No test features, labels, or scores enter fit_transform().
"""
import argparse
import csv
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common_target_readout import ROOT, digest, evaluate, load, write
from pooled_residual_calibration import native

METHODS = ["paired_center", "paired_orthogonal", "shuffled_orthogonal"]
NAMES = {"paired_center": "同图参考中心平移", "paired_orthogonal": "同图配对正交对齐",
         "shuffled_orthogonal": "打乱图片配对的正交对齐"}


def fit_transform(x, y, rotate=True, rho=0.1, lambda_override=None):
    """min ||Xc R-Yc||²/n + lambda||R-I||², R.T R=I.

    lambda=rho*||Xc.T Yc/n||_2. rho is fixed before evaluating test data.
    Identity bias prevents arbitrary rotation of the unsupported subspace.
    """
    x, y = x.double(), y.double()
    assert x.shape == y.shape and x.ndim == 2 and len(x) > 1
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    c = xc.T @ yc / len(x)
    spectral = float(torch.linalg.matrix_norm(c, ord=2)) if rotate else 0.0
    lam = rho * spectral if lambda_override is None else float(lambda_override)
    if rotate and spectral > 0:
        u, sv, vh = torch.linalg.svd(c + lam * eye)
        r = u @ vh
        smallest = float(sv[-1])
    else:
        r, smallest = eye, 0.0
    torch.testing.assert_close(r.T @ r, eye, rtol=1e-9, atol=1e-9)
    def objective(q):
        return float(((xc @ q - yc).square().sum() / len(x)) + lam * (q-eye).square().sum())
    before, after = objective(eye), objective(r)
    assert after <= before + 1e-8 * max(1, abs(before))
    return dict(target_mean=mx, source_mean=my, matrix=r, rho=rho,
                lambda_value=lam, cross_covariance_spectral_norm=spectral,
                regularized_smallest_singular_value=smallest,
                objective_before=before, objective_after=after,
                orthogonality_error=float((r.T @ r-eye).abs().max()),
                rotation_from_identity=float((r-eye).norm()/np.sqrt(len(r))))


def transform(fit, x):
    return (x.double()-fit['target_mean']) @ fit['matrix'] + fit['source_mean']


def derangement(n, seed):
    order = np.random.default_rng(seed + 41000).permutation(n)
    permutation = np.empty(n, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    assert np.all(permutation != np.arange(n))
    return permutation


def self_test():
    g = torch.Generator().manual_seed(92)
    x = torch.randn(40, 8, dtype=torch.float64, generator=g)
    r = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64, generator=g)).Q
    y = x @ r + 3
    fit = fit_transform(x, y, rho=0)
    torch.testing.assert_close(transform(fit, x), y)
    # Rank-deficient paired data must preserve an already correct space.
    small = x[:4]
    ident = fit_transform(small, small, rho=.1)
    torch.testing.assert_close(ident['matrix'], torch.eye(8, dtype=x.dtype), atol=1e-10, rtol=1e-10)
    centered = fit_transform(x, y, rotate=False)
    torch.testing.assert_close(transform(centered, x).mean(0), y.mean(0))
    torch.testing.assert_close(torch.pdist(transform(fit, x)), torch.pdist(x))
    # Correct direction and frozen-head composition.
    head = nn.Linear(8, 5).double().requires_grad_(False)
    direct = head(transform(fit, x))
    w = fit['matrix'] @ head.weight.T
    b = (fit['source_mean']-fit['target_mean']@fit['matrix'])@head.weight.T + head.bias
    torch.testing.assert_close(direct, x@w+b)
    derangement(50, 33)
    return dict(state='passed', checks=['known rotation and translation', 'rank-deficient identity recovery',
                'orthogonality and pair-distance preservation', 'objective no worse than identity',
                'frozen-head composition', 'shuffle has no true image pairs'])


def geometry(source, test, labels):
    """Descriptive ratios, not cluster classification or statistical tests."""
    source = source.double()
    # Remove the common response to each image before measuring subject separation.
    residual = source - source.mean(0, keepdim=True)
    centers = residual.mean(1, keepdim=True)
    between = centers.square().sum(-1).mean()
    within = (residual-centers).square().sum(-1).mean()
    x = test.double()
    means = torch.zeros(200, x.shape[1], dtype=x.dtype).index_add_(0, labels, x)
    counts = torch.bincount(labels, minlength=200)
    means /= counts[:, None]
    image_between = (means-x.mean(0)).square().sum(-1).mean()
    trial_within = (x-means[labels]).square().sum(-1).mean()
    return dict(source_subject_between=float(between), source_subject_within=float(within),
                source_subject_between_over_within=float(between/within.clamp_min(1e-15)),
                source_protocol='9 source subjects; same 1000 images; four predictions averaged per source/image; common image mean removed',
                target_image_between=float(image_between), target_trial_within=float(trial_within),
                target_image_between_over_within=float(image_between/trial_within.clamp_min(1e-15)),
                target_protocol='original test single trials; descriptive only, never used to fit or select alignment')


def metric_vector(m):
    return [100*m[group][metric] for group in ['single_trial','subject_aggregated'] for metric in ['top1','top5']]


def summarize(out, rows, zeros, geometries):
    summaries=[]
    for method in METHODS:
        for budget in [50,200]:
            for fold in [1,2]:
                arms={}
                for arm in ['baseline','identity']:
                    selected=sorted([r for r in rows if (r['method'],r['budget'],r['fold'],r['arm'])==(method,budget,fold,arm)],key=lambda r:r['seed'])
                    arms[arm]=np.array([metric_vector(r['metrics']) for r in selected])
                delta=arms['identity'][:,-1]-arms['baseline'][:,-1]
                zero_delta=100*(zeros[f'{fold}_identity']['subject_aggregated']['top5']-zeros[f'{fold}_baseline']['subject_aggregated']['top5'])
                summaries.append(dict(method=method,budget=budget,fold=fold,
                    baseline_mean=arms['baseline'].mean(0).tolist(),identity_mean=arms['identity'].mean(0).tolist(),
                    final_paired_differences_pp=delta.tolist(),final_difference_pp=float(delta.mean()),
                    improvement_difference_pp=float(delta.mean()-zero_delta),positive_seeds=int((delta>1e-5).sum())))
    decisions={}
    for method in METHODS:
        items=[s for s in summaries if s['method']==method]
        by_fold={str(f):float(np.mean([s['final_difference_pp'] for s in items if s['fold']==f])) for f in [1,2]}
        pairs=np.stack([s['final_paired_differences_pp'] for s in items]).reshape(2,2,5).mean(0).flatten()
        mean=float(pairs.mean()); count=int((pairs>1e-5).sum())
        decisions[method]=dict(mean_final_identity_advantage_pp=mean,per_subject_advantage_pp=by_fold,
            positive_subject_seed_pairs_after_budget_average=count,total_pairs=10,
            passes_existing_engineering_gate=bool(mean>=2 and all(v>0 for v in by_fold.values()) and count>=7))
    write(out/'summary.json',dict(rows=summaries,decisions=decisions,zeros=zeros,geometry=geometries))
    lines=['# 同图配对特征对齐：身份监督是否帮助新受试者适配','',
        '使用已固定的 S1、S2 LOSO checkpoint，比较无身份监督与身份监督模型。REVE、原 blur 投影头及图像 adapter 全部冻结。两模型使用完全相同的校准图片、真实试次和五个支持集种子。', '',
        '校准预算为 50／200 张图片，每图一条目标 EEG。每张校准图片的源参考是九名源受试者各四条试次分别编码后，先在人内、再在人间平均的 512 维表示。校准与测试图片无交集。测试仍为原始 200 张图片，单试次推理后平均同人同图全部有效原始预测。', '',
        '对照包括：原模型、只移动中心、同图配对的正交对齐、打乱图片配对的正交对齐。正交求解器参考 SCORE 第 6—8 式，正确配对的 rho 固定为 0.1，lambda = rho × 跨协方差的谱范数。打乱组沿用对应正确配对组的 lambda，只改变图片对应关系，且没有任何正确图片配对。未搜索测试成绩、未修改源 checkpoint。', '',
        '这是一项使用目标图片标签的 EEG→EEG 迁移实验；只复用了正交对齐求解器，并未复现 SCORE 的无标签匹配、视觉空间变换及完整训练流程。', '',
        '## 检索结果','',
        '五个支持集种子的均值，单位为％。每格为单试次 Top-1／Top-5；聚合 Top-1／Top-5。最终差值是身份监督减去无身份监督；增量差还扣除了各自零校准的起点差。','',
        '| 方法 | 图片数 | 受试者 | 无身份监督 | 身份监督 | 最终聚合 Top-5 差值／百分点 | 适配增量差／百分点 | 正向种子数 |',
        '|---|---:|---|---|---|---:|---:|---:|']
    def fmt(v):return f'{v[0]:.2f}／{v[1]:.2f}；{v[2]:.2f}／{v[3]:.2f}'
    for f in [1,2]:
        b,i=[metric_vector(zeros[f'{f}_{a}']) for a in ['baseline','identity']]
        lines.append(f'| 原模型 | 0 | S{f} | {fmt(b)} | {fmt(i)} | {i[-1]-b[-1]:+.2f} | — | — |')
    for s in summaries:
        lines.append(f"| {NAMES[s['method']]} | {s['budget']} | S{s['fold']} | {fmt(s['baseline_mean'])} | {fmt(s['identity_mean'])} | {s['final_difference_pp']:+.2f} | {s['improvement_difference_pp']:+.2f} | {s['positive_seeds']}／5 |")
    lines+=['','## 判断','']
    for method,d in decisions.items():
        lines.append(f"{NAMES[method]}：两个预算平均的身份优势为 {d['mean_final_identity_advantage_pp']:+.2f} 个百分点；S1 为 {d['per_subject_advantage_pp']['1']:+.2f}，S2 为 {d['per_subject_advantage_pp']['2']:+.2f}。按每位受试者每个种子先平均两个预算后，正向配对为 {d['positive_subject_seed_pairs_after_budget_average']}／10。{'达到' if d['passes_existing_engineering_gate'] else '未达到'}既定工程筛选标准。")
        lines.append('')
    lines+=['正确配对是否有额外作用，应在同一个模型内将正交对齐与中心平移、打乱配对比较。身份监督是否帮助适配，则比较两模型最终成绩及适配增量；不能用更低起点带来的较大增量代替最终优势。', '',
        '## 表征诊断','',
        '| 模型 | 源受试者间／受试者内离散比 | 新受试者图片间／同图试次内离散比 |',
        '|---|---:|---:|']
    for key,g in geometries.items():
        lines.append(f"| {key} | {g['source_subject_between_over_within']:.4f} | {g['target_image_between_over_within']:.4f} |")
    lines+=['','源受试者离散比在四次特征平均、并扣除同图共同响应后计算，数值较高表示源身份中心相对更分离。目标图片离散比在单试次测试特征上计算，数值较高表示图片中心差异相对试次波动更大。两者只作描述，未用于选模型或适配参数。源域四次平均与目标域单试次的比值不能相互作数值比较。','',
        '身份聚团与适配效果之间的因果关系仍需独立干预检验。本实验比较两套原生检索系统，各自的图像 adapter 也不同；S1、S2 为开发受试者，五个种子不是五名独立受试者。未自动触发 S3／S4 或新一轮源模型训练。','',
        '参考：[SCORE，式 6—8 与固定 rho=0.1](https://arxiv.org/html/2608.19134v1)。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (out/'all_results.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','budget','fold','arm','seed','single_top1','single_top5','agg_top1','agg_top5'])
        for r in rows:writer.writerow([r[k] for k in ['method','budget','fold','arm','seed']]+metric_vector(r['metrics']))


def main(args):
    torch.set_num_threads(4)
    out=args.out.resolve()
    assert str(out).startswith('/mnt/DATA-4/anteng/eeg-retrieval/experiments/')
    out.mkdir(parents=True,exist_ok=False)
    args.created_out = True
    started=time.time()
    write(out/'status.json',dict(state='preflight',pid=os.getpid(),started=started))
    snapshot=out/'source';snapshot.mkdir()
    for name in ['paired_subject_alignment.py','common_target_readout.py','pooled_residual_calibration.py']:
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
        (snapshot/name).chmod(0o444)
    write(out/'source_sha256.json',{p.name:digest(p) for p in snapshot.iterdir()})
    snapshot.chmod(0o555)
    write(out/'protocol.json',dict(folds=[1,2],arms=['baseline','identity'],budgets=[50,200],seeds=list(range(33,38)),
        methods=METHODS,reference='each source subject four encoded trials averaged, then nine source subjects averaged for same image',
        label_use='same-image correspondences from target labeled support only',rho=.1,
        regularization='lambda = rho * spectral_norm(Xcenter.T @ Ycenter / n)',
        shuffle_regularization='reuse correctly-paired lambda for each fold/arm/budget/seed; only row correspondence changes',
        original_heads='frozen; native image adapter retained; no per-trial output normalization',
        selection='fixed checkpoints and rho; no target test or extra development selection',
        evaluation='200-image native blur gallery; all valid raw trial predictions averaged within image',
        source_root=str(ROOT),paper='https://arxiv.org/html/2608.19134v1 equations6-8; solver only, not full SCORE',
        source_reference_budget='36 source trials per calibration image, fixed equally for both models',
        numpy=np.__version__,torch=torch.__version__,device='cpu',threads=4))
    write(out/'self_test.json',self_test())
    pairs={};zeros={};geometries={};manifest={}
    for fold in [1,2]:
        pairs[fold]={}
        for arm in ['baseline','identity']:
            c,_,_,labels,meta=load(fold,arm)
            p,_,gallery,expected=native(fold,arm,c,meta)
            protocol=json.loads((ROOT/f'fold_{fold:02d}_{arm}'/'protocol.json').read_text())
            assert meta['heldout'] not in protocol['subjects'] and protocol['source_repeats']==4
            original_cfg=json.loads((ROOT/f'fold_{fold:02d}_{arm}'/'config.json').read_text())
            val_ids=json.loads(Path(original_cfg['validation_ids_path']).read_text())
            test_ids=np.load(Path(original_cfg['source_root'])/'data/test_ids.npy')
            assert not np.intersect1d(c['union'],val_ids).size
            assert not np.intersect1d(c['union'],test_ids).size
            meta.update(reference_protocol_sha256=digest(ROOT/f'fold_{fold:02d}_{arm}'/'protocol.json'),
                        reference_config_sha256=digest(ROOT/f'fold_{fold:02d}_{arm}'/'config.json'),
                        source_subjects=protocol['subjects'],source_repeats=protocol['source_repeats'])
            indices=np.searchsorted(c['union'],c['images'])
            assert np.array_equal(c['union'][indices],c['images'])
            source=c['source_means'][:,indices].float()
            assert source.shape==(9,1000,512) and torch.isfinite(source).all()
            zero=evaluate(p,c['etest'].float(),labels,gallery)
            for group in ['single_trial','subject_aggregated']:
                for k in ['top1','top5']:assert abs(zero[group][k]-expected[group][k])<2e-6
            key=f'{fold}_{arm}';zeros[key]=zero;manifest[key]=meta
            geometries[key]=geometry(source,c['etest'],labels)
            pairs[fold][arm]=(c,p,gallery,labels,source.mean(0))
        b,i=[pairs[fold][a][0] for a in ['baseline','identity']]
        assert np.array_equal(b['images'],i['images']) and np.array_equal(b['union'],i['union']) and b['records']==i['records']
    write(out/'manifest.json',manifest);write(out/'zero_metrics.json',zeros);write(out/'geometry.json',geometries)
    write(out/'preflight.json',dict(state='passed',checks=['all four native zero metrics reproduced',
        'same support images and EEG records across model arms','calibration/test/source-validation IDs disjoint via load()',
        'source reference excludes heldout subject','finite 512-D cached features','source labels only on support pairs']))
    print('PREFLIGHT PASSED',flush=True)
    if args.preflight_only:
        write(out/'status.json',dict(state='preflight_passed'));return
    rows=[];done=0
    for fold,pair in pairs.items():
        for budget in [50,200]:
            for seed in range(33,38):
                subset=np.random.default_rng(seed).permutation(1000)[:budget]
                shuffle=derangement(budget,seed)
                for arm,(c,p,gallery,labels,reference) in pair.items():
                    x,y=c['etrain'][subset],reference[subset]
                    frozen={k:v.clone() for k,v in p.state_dict().items()}
                    paired_fit=fit_transform(x,y)
                    for method in METHODS:
                        key=f's{fold}_{arm}_n{budget}_seed{seed}_{method}'
                        write(out/'status.json',dict(state='running',current=key,completed=done,total=120,pid=os.getpid()))
                        if method=='paired_orthogonal':
                            fit=paired_fit
                        elif method=='shuffled_orthogonal':
                            fit=fit_transform(x,y[shuffle],lambda_override=paired_fit['lambda_value'])
                            assert fit['lambda_value']==paired_fit['lambda_value']
                            torch.testing.assert_close(fit['source_mean'],paired_fit['source_mean'])
                        else:
                            fit=fit_transform(x,y,rotate=False)
                        torch.save(fit,out/f'{key}.pt')
                        # Verify the actual saved artifact, then use the reloaded transform.
                        reloaded=torch.load(out/f'{key}.pt',weights_only=True)
                        torch.testing.assert_close(transform(fit,x),transform(reloaded,x),rtol=0,atol=0)
                        adapted=transform(reloaded,c['etest']).float()
                        assert torch.isfinite(adapted).all()
                        metrics=evaluate(p,adapted,labels,gallery)
                        assert all(torch.equal(v,frozen[k]) for k,v in p.state_dict().items())
                        row=dict(method=method,budget=budget,fold=fold,arm=arm,seed=seed,metrics=metrics,
                            subset_indices=subset.tolist(),support_image_ids=c['images'][subset].tolist(),
                            reference_permutation=shuffle.tolist() if method=='shuffled_orthogonal' else list(range(budget)),
                            transform_diagnostics={k:v for k,v in fit.items() if not isinstance(v,torch.Tensor)},
                            relative_feature_shift=float((adapted-c['etest']).norm()/c['etest'].norm()),
                            checkpoint=manifest[f'{fold}_{arm}']['checkpoint'])
                        rows.append(row);write(out/f'{key}.json',row);done+=1
                        print(key,metric_vector(metrics),'completed',done,'/120',flush=True)
    write(out/'all_results.json',rows)
    summarize(out,rows,zeros,geometries)
    write(out/'status.json',dict(state='complete',completed=done,total=120,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        if getattr(args,'created_out',False):
            write(args.out/'status.json',dict(state='failed',error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        raise
