"""Frozen REVE + SATTC/SCORE deployment comparison, without target labels in calibration.

This is a transfer of deployment operators to existing native blur systems. It
does not reproduce either paper's source training or average-EEG test protocol.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from common_target_readout import ROOT, load, write, digest, evaluate
from pooled_residual_calibration import native


def metrics_from_scores(scores, labels):
    """Labels are used ONLY here, after calibration, for retrieval scoring."""
    scores = torch.as_tensor(scores).detach().cpu().float()
    labels = labels.cpu().long()
    n, m = scores.shape
    assert n == len(labels) and torch.isfinite(scores).all()
    assert labels.min() == 0 and labels.max() == m - 1
    counts = torch.bincount(labels, minlength=m)
    assert (counts > 0).all()
    sums = torch.zeros(m, m)
    sums.index_add_(0, labels, scores)
    def hits(s, y):
        rank = s.topk(5, dim=1).indices
        return dict(top1=float((rank[:, 0] == y).float().mean()),
                    top5=float((rank == y[:, None]).any(1).float().mean()), count=len(y))
    return dict(single_trial=hits(scores, labels),
                subject_aggregated=hits(sums / counts[:, None], torch.arange(m)),
                repeats_min=int(counts.min()), repeats_max=int(counts.max()))


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def check_metrics(actual, expected):
    for group in ('single_trial', 'subject_aggregated'):
        assert actual[group]['count'] == expected[group]['count']
        for name in ('top1', 'top5'):
            assert abs(actual[group][name] - expected[group][name]) < 2e-6, (actual, expected)


def evaluator_selftest():
    labels = torch.arange(10).repeat_interleave(3)
    gallery = torch.eye(10)
    scores = gallery[labels] @ gallery.T
    assert metrics_from_scores(scores, labels)['subject_aggregated']['top1'] == 1
    generator = torch.Generator().manual_seed(33)
    q = torch.randn(30, 12, generator=generator)
    g = torch.randn(10, 12, generator=generator)
    sums = torch.zeros(10, 12).index_add_(0, labels, q) / 3
    score_means = torch.zeros(10, 10).index_add_(0, labels, q @ g.T) / 3
    torch.testing.assert_close(score_means, sums @ g.T, atol=2e-6, rtol=2e-6)


@torch.no_grad()
def source_image_reference(fold, arm, meta):
    """Source train images only, for author-style SATTC image-whitening state."""
    cfg=json.loads((ROOT/f'fold_{fold:02d}_{arm}'/'config.json').read_text())
    data=Path(cfg['source_root'])/'data'
    selected=[]
    for split in cfg['training_splits']:
        ids=np.load(data/f'{split}_ids.npy')
        subjects=np.load(data/f'{split}_subjects.npy').astype(str)
        selected.append(ids[subjects != cfg['heldout_subject']])
    validation=np.array(json.loads(Path(cfg['validation_ids_path']).read_text()))
    ids=np.setdiff1d(np.unique(np.concatenate(selected)),validation)
    assert not np.intersect1d(ids,meta['candidate_ids']).size
    assert len(ids)==14886
    state=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['model']
    image=torch.nn.Sequential(torch.nn.Linear(1536,768),torch.nn.ELU(),torch.nn.Dropout(0),torch.nn.Linear(768,1536))
    image.load_state_dict({k.removeprefix('image_adapter.'):v for k,v in state.items() if k.startswith('image_adapter.')},strict=True)
    image.eval().requires_grad_(False)
    path=Path(cfg['source_root'])/'features/blur.npy'
    raw=np.load(path,mmap_mode='r')
    ref=torch.cat([F.normalize(image(torch.from_numpy(raw[block].copy())),dim=-1) for block in np.array_split(ids,30)])
    meta['source_image_reference'] = dict(ids=ids.tolist(),count=len(ids),normalization='native image_adapter then L2',
        weighting='one feature per unique source training image; source validation/test excluded',
        feature_cache=str(path),validation_ids_path=cfg['validation_ids_path'])
    return ref


def report(out, index):
    rows = []
    for record in index['records']:
        for method, info in record['methods'].items():
            s = info['metrics']
            rows.append(dict(fold=record['fold'], arm=record['arm'], method=method,
                             single_top1=100*s['single_trial']['top1'], single_top5=100*s['single_trial']['top5'],
                             aggregate_top1=100*s['subject_aggregated']['top1'], aggregate_top5=100*s['subject_aggregated']['top5']))
    if not rows:
        return
    with (out/'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    text = [
        '# 固定 REVE 模型的 SATTC 与 SCORE 部署对照', '',
        '本实验比较 S1、S2 的固定无身份监督与身份监督 checkpoint。保留原投影头、各自图像 adapter 和 EEG 预处理；只在检索时校准。', '',
        '每位受试者的全部单试次测试预测和相同的 200 张候选图片用于无标签校准。算法不接收图片标签或试次分组，标签只在输出分数后用于计分。两模型候选图片相同，但冻结图像 adapter 不同，所以候选向量不同。', '',
        '本表的聚合方式是同人同图全部有效试次的最终候选分数取均值。原始点积时，这与先平均原始预测再检索等价；加入候选相关校准或逐试次归一化后，不再等价。SATTC、SCORE 结果因此单列为分数聚合。', '',
        '使用整批无标签测试 EEG 的实验属于传导式测试时校准，与此前只使用独立校准 EEG 的适配实验分开。论文原始 EEG 预处理、重复试次平均和源模型训练均未替换；这里验证部署算法在当前 REVE 系统上的作用。', '',
        '单位为百分比，每格为 Top-1／Top-5。raw_dot 为原系统，raw_cosine 用于检查逐试次归一化本身的影响。', '',
        '| 受试者 | 方法 | 无身份监督单试次 | 无身份监督聚合 | 身份监督单试次 | 身份监督聚合 | 身份−基线聚合 Top-5（百分点） |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for fold in (1, 2):
        methods = list(dict.fromkeys(r['method'] for r in rows if r['fold'] == fold))
        for method in methods:
            pair = {r['arm']: r for r in rows if r['fold'] == fold and r['method'] == method}
            if len(pair) != 2:
                continue
            b, i = pair['baseline'], pair['identity']
            text.append(f"| S{fold} | {method} | {b['single_top1']:.2f}／{b['single_top5']:.2f} | {b['aggregate_top1']:.2f}／{b['aggregate_top5']:.2f} | {i['single_top1']:.2f}／{i['single_top5']:.2f} | {i['aggregate_top1']:.2f}／{i['aggregate_top5']:.2f} | {i['aggregate_top5']-b['aggregate_top5']:+.2f} |")
    text += ['', '配置、数值选择及论文对应关系见 config.json、各方法 diagnostics.json 和源码中的说明。失败阶段保留异常，不将回退结果写成完整方法成功。所有分数矩阵保存在各模型子目录，可独立重算指标。', '']
    (out/'REPORT.md').write_text('\n'.join(text))


def main(args):
    from sattc_deployment import calibrate_sattc, author_defaults
    from score_deployment import run_score_deployment, ScoreConfig
    torch.set_num_threads(4)
    torch.manual_seed(33)
    evaluator_selftest()
    args.out.mkdir(parents=True, exist_ok=False)
    args.created_out = True
    started = time.time()
    index = {'records': []}
    config = dict(
        purpose='fixed-checkpoint native blur retrieval; deployment-method transfer',
        folds=[1,2], arms=['baseline','identity'], trainable_parameters=0, labels_used_for_calibration=False,
        target_data='all unlabelled target single-trial test predictions; transductive',
        gallery='same200 candidate images; each arm retains its native frozen image_adapter',
        input_normalization='Q raw blur_head output; G native image_adapter output L2 normalized',
        aggregation='mean final calibrated scores over all valid repetitions; labels in evaluator only',
        selection='no test-based parameter or checkpoint selection; report every stage',
        device='cpu', threads=4, source_dir=str(args.sources),
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        checkpoint_role='existing fixed development checkpoints; not selected anew',
        score_parameters=asdict(ScoreConfig()), sattc_parameters=author_defaults(args.sources),
    )
    write(args.out/'config.json', config)
    write(args.out/'status.json', dict(state='cpu_preflight', pid=os.getpid(), started_unix=started))
    pair_meta = {}
    for fold in (1,2):
        for arm in ('baseline','identity'):
            key=f's{fold}_{arm}'
            folder=args.out/key; folder.mkdir()
            write(args.out/'status.json', dict(state='evaluating', current=key, completed=len(index['records']), total=4, pid=os.getpid()))
            cache, _, raw_gallery, labels, meta=load(fold,arm)
            projection, _, gallery, expected=native(fold,arm,cache,meta)
            with torch.no_grad():
                query=projection(cache['etest'].float()).detach()
            assert not query.requires_grad and not gallery.requires_grad
            assert query.shape == (len(labels),1536) and gallery.shape == (200,1536)
            assert torch.isfinite(query).all() and torch.isfinite(gallery).all()
            before={k:v.clone() for k,v in projection.state_dict().items()}
            checkpoint_hash=meta['checkpoint_sha256']
            meta['test_order_sha256']=hashlib.sha256(labels.numpy().tobytes()).hexdigest()
            meta['query_norm']={'min':float(query.norm(dim=1).min()), 'mean':float(query.norm(dim=1).mean()), 'max':float(query.norm(dim=1).max())}
            meta['test_order_hash_scope']='hash of candidate-index sequence; feature_sha256 additionally binds within-image trial ordering'
            if fold in pair_meta:
                other=pair_meta[fold]
                for k in ('candidate_ids','test_order_sha256','records','image_pool'):
                    assert meta[k] == other[k], k
            else:
                pair_meta[fold]=meta.copy()
            np.save(folder/'labels.npy', labels.numpy())
            record=dict(fold=fold, arm=arm, metadata=meta, labels_file=f'{key}/labels.npy', methods={})
            def save_method(name, score):
                score=torch.as_tensor(score).detach().cpu().float()
                assert score.shape == (len(labels),200) and torch.isfinite(score).all()
                result=metrics_from_scores(score,labels)
                np.save(folder/f'{name}.npy',score.numpy())
                record['methods'][name]=dict(scores_file=f'{key}/{name}.npy', metrics=result)
                print(key,name,json.dumps(result),flush=True)
                return result
            raw=save_method('raw_dot',query@gallery.T)
            check_metrics(raw,expected)
            check_metrics(raw,evaluate(projection,cache['etest'].float(),labels,gallery))
            save_method('raw_cosine',F.normalize(query,dim=1)@gallery.T)
            image_reference=source_image_reference(fold,arm,meta)
            print(key,'CPU PREFLIGHT PASS: original native metrics reproduced; inputs finite; fixed projection',flush=True)
            for method, function in (
                ('sattc',lambda:calibrate_sattc(query.clone(),gallery.clone(),source_dir=args.sources,gallery_reference=image_reference)),
                ('score',lambda:run_score_deployment(query.clone(),gallery.clone(),ScoreConfig())),
            ):
                write(args.out/'status.json',dict(state='evaluating',current=f'{key}/{method}',completed=len(index['records']),total=4,pid=os.getpid()))
                try:
                    t=time.time()
                    with torch.no_grad():
                        result=function()
                    for stage, scores in result['scores'].items():
                        save_method(f'{method}_{stage}',scores)
                    write(folder/f'{method}_diagnostics.json',jsonable(dict(result['diagnostics'],elapsed_seconds=time.time()-t)))
                    if method == 'score':
                        landmark=result['landmarks']
                        qi=landmark['query_indices'].cpu()
                        gi=landmark['gallery_indices'].cpu()
                        correct=labels[qi]==gi
                        write(folder/'score_posthoc_landmark_audit.json',dict(
                            scope='labels inspected only after all transforms and scores were fixed; not used for fit or selection',
                            selected_count=len(qi), correct_count=int(correct.sum()),
                            accuracy=float(correct.float().mean()) if len(qi) else None,
                            weighted_accuracy=float((correct.double()*landmark['weights'].cpu()).sum()/landmark['weights'].sum()) if len(qi) else None))
                    # Transform and landmark indices are preserved for independent checks.
                    extras={k:v for k,v in result.items() if k not in ('scores','diagnostics','features')}
                    if extras:
                        torch.save(extras,folder/f'{method}_state.pt')
                except Exception as exc:
                    write(folder/f'{method}_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
                    print(key,method,'FAILED',repr(exc),flush=True)
            assert all(torch.equal(v,before[k]) for k,v in projection.state_dict().items())
            assert digest(Path(meta['checkpoint'])) == checkpoint_hash
            index['records'].append(record)
            write(args.out/'audit_index.json',index)
            write(folder/'result.json',record)
            report(args.out,index)
    failures=sorted(str(x.relative_to(args.out)) for x in args.out.glob('*/*_failure.json'))
    write(args.out/'status.json',dict(state='complete' if not failures else 'completed_with_failures',completed=4,total=4,failures=failures,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        if getattr(args,'created_out',False):
            write(args.out/'status.json',dict(state='failed',error=repr(exc),traceback=traceback.format_exc(),ended_unix=time.time()))
        raise
