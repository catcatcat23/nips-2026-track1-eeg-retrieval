"""Sequential calibration diagnostics, fixed low-LR control, conditional internal selection."""
import argparse, json, csv, io
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common_target_readout import load, write, evaluate
from pooled_residual_calibration import Residual, native
OLD=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/pooled_residual_calibration_20261108/results')
STEPS=[0,10,25,50,100,200]

def report(folder,dest):
    dest.mkdir(exist_ok=True,parents=True)
    rows=[];points=[]
    for fold in [1,2]:
        for budget in [50,200]:
            group={}
            for arm in ['baseline','identity']:
                vals=[json.loads((folder/f's{fold}_{arm}_n{budget}_seed{seed}.json').read_text()) for seed in range(33,38)]
                group[arm]=vals
                for step in STEPS:
                    for seed,v in zip(range(33,38),vals):
                        s=v['scores'][str(step)]
                        points.append(dict(fold=fold,budget=budget,arm=arm,seed=seed,step=step,agg_top5=100*s['subject_aggregated']['top5']))
                arr=np.array([[v['scores']['200'][g][m]*100 for g in ['single_trial','subject_aggregated'] for m in ['top1','top5']] for v in vals])
                rows.append(dict(fold=fold,budget=budget,arm=arm,zero=np.mean([v['scores']['0']['subject_aggregated']['top5']*100 for v in vals]),final=arr.mean(0).tolist(),loss50=np.mean([v['losses'][49] for v in vals]),loss200=np.mean([v['losses'][-1] for v in vals]),residual_history='available' if 'diagnostics' in vals[0] else 'not recorded'))
            for step in STEPS:
                diff=[100*(i['scores'][str(step)]['subject_aggregated']['top5']-b['scores'][str(step)]['subject_aggregated']['top5']) for b,i in zip(group['baseline'],group['identity'])]
                points.append(dict(fold=fold,budget=budget,arm='paired_identity_minus_baseline',seed='mean',step=step,agg_top5=float(np.mean(diff)),positive_seeds=sum(x>0 for x in diff)))
    write(dest/'summary.json',rows);write(dest/'curves.json',points)
    text=['# 配对校准结果','每格单位为百分比。五种子为配对重复，不是五名受试者。', '|受试者|图片数|模型|零步聚合Top5|最终单试次Top1/5|最终聚合Top1/5|loss50→200|','|---|---|---|---|---|---|---|']
    for r in rows:
        a=r['final'];text.append(f"|S{r['fold']}|{r['budget']}|{r['arm']}|{r['zero']:.2f}|{a[0]:.2f}/{a[1]:.2f}|{a[2]:.2f}/{a[3]:.2f}|{r['loss50']:.3f}→{r['loss200']:.3f}|")
    (dest/'report.md').write_text('\n'.join(text)+'\n历史残差范数未记录的实验不补造。\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for f in [1,2]:
        for j,n in enumerate([50,200]):
            ax=axes[f-1,j]
            for arm,color in [('baseline','#0072B2'),('identity','#D55E00')]:
                ys=[]
                for step in STEPS:
                    vals=[p['agg_top5'] for p in points if p['fold']==f and p['budget']==n and p['arm']==arm and p['step']==step]
                    ys.append(np.mean(vals));ax.scatter([step]*5,vals,s=9,alpha=.25,color=color)
                ax.plot(STEPS,ys,'o-',label=arm,color=color)
            ax.set_title(f'S{f}, {n} labelled images');ax.set_ylabel('Blur aggregate Top-5 (%)');ax.set_xlabel('Updates');ax.legend()
    fig.tight_layout();fig.savefig(dest/'curves.png',dpi=300);fig.savefig(dest/'curves.pdf');plt.close(fig)
    return rows

def gate(folder):
    diffs={};drops=[]
    for fold in [1,2]:
        for seed in range(33,38):
            per=[]
            for n in [50,200]:
                vals={a:json.loads((folder/f's{fold}_{a}_n{n}_seed{seed}.json').read_text()) for a in ['baseline','identity']}
                per.append(100*(vals['identity']['scores']['200']['subject_aggregated']['top5']-vals['baseline']['scores']['200']['subject_aggregated']['top5']))
            diffs[f'{fold}_{seed}']=float(np.mean(per))
    for f in [1,2]:
        for n in [50,200]:
            for a in ['baseline','identity']:
                vals=[json.loads((folder/f's{f}_{a}_n{n}_seed{s}.json').read_text()) for s in range(33,38)]
                drop=np.mean([100*(v['scores']['50']['subject_aggregated']['top5']-v['scores']['200']['subject_aggregated']['top5']) for v in vals])
                if drop>=2:drops.append(dict(fold=f,budget=n,arm=a,drop=float(drop)))
    means={str(f):float(np.mean([v for k,v in diffs.items() if k.startswith(str(f)+'_')])) for f in [1,2]}
    return dict(paired_differences=diffs,subject_means=means,mean=float(np.mean(list(diffs.values()))),positive_pairs=sum(v>0 for v in diffs.values()),passes=bool(np.mean(list(diffs.values()))>=2 and min(means.values())>0 and sum(v>0 for v in diffs.values())>=7),late_degradation=drops)

@torch.no_grad()
def diagnostics(adapter,projection,x,y):
    z=adapter(x);score=projection(z)@y.T
    return dict(residual_input_norm_ratio=float(((z-x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).mean()),support_top1=float((score.argmax(1)==torch.arange(len(x))).float().mean()))

def fit(x,y,projection,seed,steps=200,callback=None):
    torch.manual_seed(seed);a=Residual();opt=torch.optim.AdamW(a.parameters(),lr=3e-5,weight_decay=.01);gen=torch.Generator().manual_seed(seed+10000)
    losses=[];grad=[]
    for step in range(1,steps+1):
        ix=torch.randperm(len(x),generator=gen)[:50];loss=F.cross_entropy(projection(a(x[ix]))@y[ix].T,torch.arange(len(ix)))
        opt.zero_grad();loss.backward();assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
        gn=nn.utils.clip_grad_norm_(a.parameters(),1);opt.step();losses.append(float(loss.detach()));grad.append(float(gn))
        if callback and step in STEPS:callback(step,a)
    return a,losses,grad

def main(root):
    torch.set_num_threads(4)
    write(root/'protocol.json',dict(lr=3e-5,late_degradation='mean step50 minus step200 >=2 percentage points in any subject/budget/arm',selection='fixed200; optional development-only mean rank',gate='mean >=2pp, both subjects positive, >=7/10 pairs positive',stages='stage1 then stage2 then conditional internal selection; decision before LoRA'))
    report(OLD,root/'stage1_report');write(root/'stage1_report/decision.json',dict(next='low_lr_control',historical_residual_norm='missing',gate=gate(OLD)))
    pairs={}
    for f in [1,2]:
        pairs[f]={}
        for arm in ['baseline','identity']:
            cache,_,_,labels,meta=load(f,arm);p,t,g,e=native(f,arm,cache,meta)
            actual=evaluate(p,cache['etest'],labels,g)
            for group in ['single_trial','subject_aggregated']:
                for k in ['top1','top5']:assert abs(actual[group][k]-e[group][k])<2e-6
            pairs[f][arm]=(cache,labels,p,t,g,actual,meta)
        b,i=[pairs[f][arm][0] for arm in ['baseline','identity']]
        assert np.array_equal(b['images'],i['images']) and b['records']==i['records']
    out=root/'stage2_low_lr';out.mkdir()
    write(out/'manifest.json',{str(f):{a:v[-1] for a,v in pair.items()} for f,pair in pairs.items()})
    for f,pair in pairs.items():
        for n in [50,200]:
            for seed in range(33,38):
                subset=np.random.default_rng(seed).permutation(1000)[:n]
                for arm,(c,labels,p,t,g,zero,meta) in pair.items():
                    key=f's{f}_{arm}_n{n}_seed{seed}';write(root/'status.json',dict(state='running',stage='low_lr',current=key))
                    x=c['etrain'][subset].float();y=t[subset];scores={'0':zero};diags={};original={k:v.clone() for k,v in p.state_dict().items()}
                    def cb(step,a):
                        scores[str(step)]=evaluate(nn.Sequential(a,p),c['etest'],labels,g);diags[str(step)]=diagnostics(a,p,x,y)
                    a,loss,grad=fit(x,y,p,seed,callback=cb)
                    assert all(torch.equal(v,original[k]) for k,v in p.state_dict().items())
                    buf=io.BytesIO();torch.save(a.state_dict(),buf);buf.seek(0);b=Residual();b.load_state_dict(torch.load(buf,weights_only=True));torch.testing.assert_close(a(x[:4]),b(x[:4]),rtol=0,atol=0)
                    write(out/f'{key}.json',dict(scores=scores,losses=loss,gradient_norm_before_clip=grad,diagnostics=diags,subset_indices=subset.tolist(),image_ids=c['images'][subset].tolist()))
                    torch.save(dict(adapter=a.state_dict(),checkpoint=meta['checkpoint']),out/f'{key}.pt');print('DONE',key,flush=True)
    report(out,root/'stage2_report');decision=gate(out);write(root/'stage2_report/decision.json',decision)
    if decision['late_degradation']:
        dest=root/'stage2_internal_selection';dest.mkdir()
        for f,pair in pairs.items():
            for n in [50,200]:
                for seed in range(33,38):
                    support=np.random.default_rng(seed).permutation(1000)[:n];fitidx=support[:int(.8*n)];dev=support[int(.8*n):];cand=np.setdiff1d(np.arange(1000),fitidx);devlabels=torch.from_numpy(np.searchsorted(cand,dev))
                    assert not np.intersect1d(fitidx,dev).size
                    for arm,(c,labels,p,t,g,zero,meta) in pair.items():
                        ranks={}
                        @torch.no_grad()
                        def rank_cb(step,a):
                            score=p(a(c['etrain'][dev]))@t[cand].T;correct=score.gather(1,devlabels[:,None]);ranks[step]=float((1+(score>correct).sum(1)+.5*((score==correct).sum(1)-1)).float().mean())
                        fit(c['etrain'][fitidx],t[fitidx],p,seed,callback=rank_cb)
                        chosen=min(ranks,key=lambda s:(ranks[s],s))
                        a,loss,grad=fit(c['etrain'][support],t[support],p,seed,steps=chosen)
                        score=evaluate(nn.Sequential(a,p),c['etest'],labels,g)
                        key=f's{f}_{arm}_n{n}_seed{seed}'
                        write(dest/f'{key}.json',dict(scores={'0':zero,'selected':score},selected_steps=chosen,development_mean_ranks=ranks,fit_indices=fitidx.tolist(),dev_indices=dev.tolist(),candidate_indices=cand.tolist(),losses=loss))
                        torch.save(a.state_dict(),dest/f'{key}.pt')
        chosen_gate=gate_selected(dest);write(dest/'decision.json',chosen_gate)
    else:chosen_gate=None
    final=dict(low_lr=decision,internal_selection=chosen_gate,next='extended_validation' if decision['passes'] or (chosen_gate and chosen_gate['passes']) else 'lora_200',state='stage2_complete')
    write(root/'decision.json',final);write(root/'status.json',final);print(json.dumps(final),flush=True)

def gate_selected(folder):
    diffs={}
    for f in [1,2]:
        for seed in range(33,38):
            vals=[]
            for n in [50,200]:
                b=json.loads((folder/f's{f}_baseline_n{n}_seed{seed}.json').read_text());i=json.loads((folder/f's{f}_identity_n{n}_seed{seed}.json').read_text())
                vals.append(100*(i['scores']['selected']['subject_aggregated']['top5']-b['scores']['selected']['subject_aggregated']['top5']))
            diffs[f'{f}_{seed}']=float(np.mean(vals))
    means=[np.mean([v for k,v in diffs.items() if k.startswith(str(f)+'_')]) for f in [1,2]]
    return dict(paired_differences=diffs,passes=bool(np.mean(list(diffs.values()))>=2 and min(means)>0 and sum(v>0 for v in diffs.values())>=7))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:main(args.root)
    except Exception as exc:
        write(args.root/'failure.json',dict(error=repr(exc)));raise
