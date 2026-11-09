"""Source-only reference statistics and support-only closed-form adaptation.
CORAL: Sun et al., AAAI2016, Section3.2 Algorithm1; reverse-domain deployment
with frozen EEG readout is our transfer, not reproduction of paper benchmarks.
"""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common_target_readout import ROOT,load,write,evaluate,digest
from pooled_residual_calibration import native

def cov(x):
    z=x-x.mean(0);return z.T@z/(len(x)-1)

def power(c,p):
    c=(c+c.T)/2;v,u=torch.linalg.eigh(c);floor=c.diag().mean().clamp_min(1e-12)*1e-4
    return (u*v.clamp_min(floor).pow(p))@u.T

def ridge_fit(x,y,lam):
    # Objective ||XW+b-Y||_F^2/n + lambda||W||_F^2. Unpenalized intercept.
    x=x.double();y=y.double();mu=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-6);z=(x-mu)/scale;ym=y.mean(0);yc=y-ym
    w=z.T@torch.linalg.solve(z@z.T+len(x)*lam*torch.eye(len(x),dtype=z.dtype),yc)
    return dict(mu=mu,scale=scale,w=w,bias=ym)

def ridge_predict(fit,x):return ((x.double()-fit['mu'])/fit['scale'])@fit['w']+fit['bias']

def tests():
    torch.manual_seed(7);x=torch.randn(30,8,dtype=torch.float64);y=torch.randn(30,4,dtype=torch.float64)
    fit=ridge_fit(x,y,10);z=(x-fit['mu'])/fit['scale']
    primal=torch.linalg.solve(z.T@z+300*torch.eye(8,dtype=z.dtype),z.T@(y-y.mean(0)))
    torch.testing.assert_close(primal,fit['w'])
    torch.testing.assert_close(ridge_fit(x,y+3,10)['w'],fit['w']);torch.testing.assert_close(ridge_predict(ridge_fit(x,y+3,10),x),ridge_predict(fit,x)+3)
    a=cov(x)+torch.eye(8);b=cov(x*torch.arange(1,9))+torch.eye(8);mapping=power(a,-.5)@power(b,.5)
    torch.testing.assert_close(mapping.T@a@mapping,b,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(power(a,-.5)@power(a,.5),torch.eye(8,dtype=a.dtype),rtol=1e-5,atol=1e-5)
    return 'PASS dual/primal Ridge; intercept; CORAL covariance identity; identity map'

def sources(root):
    from data import DatasetEEG
    from lora_subject_adaptation import model_for
    assert torch.cuda.is_available()
    for fold in [1,2]:
        common=None
        for arm in ['baseline','identity']:
            key=f's{fold}_{arm}';write(root/'source_status.json',dict(state='extracting',current=key))
            c,_,_,_,meta=load(fold,arm);cfg=json.loads((ROOT/f'fold_{fold:02d}_{arm}'/'config.json').read_text())
            train=[DatasetEEG(cfg['source_root'],s,aux=False) for s in ['train','val']]
            testids=set(map(int,np.load(Path(cfg['source_root'])/'data/test_ids.npy')));val=set(json.loads(Path(cfg['validation_ids_path']).read_text()))
            groups={}
            for d,ds in enumerate(train):
                for i,(s,k) in enumerate(zip(ds.subjects.astype(str),ds.ids)):
                    if s!=cfg['heldout_subject'] and int(k) not in val|testids:groups.setdefault((s,int(k)),[]).append((d,i))
            subjects=sorted(set(s for s,k in groups));eligible=sorted(set.intersection(*[set(k for s,k in groups if s==subject) for subject in subjects]))
            # Source labels only; RNG independent of target support identities.
            images=np.random.default_rng(20261108).choice(eligible,1000,replace=False)
            records=[groups[(s,int(k))][0] for s in subjects for k in images]
            protocol=dict(subjects=subjects,images=images.tolist(),records=records,selection='source-only eligible training IDs; fixed seed20261108; first stored trial; no target image identities used')
            if common is None:common=protocol
            else:assert protocol==common
            model=model_for(cfg,meta);capture=[];handle=model.blur_head.register_forward_pre_hook(lambda m,x:capture.append(x[0].detach()))
            values=[]
            with torch.no_grad():
                for start in range(0,len(records),64):
                    x=torch.stack([torch.from_numpy(train[d].x[i].copy()) for d,i in records[start:start+64]]).cuda();capture.clear();model.encode_eeg(x);values.append(capture[-1].float().cpu())
                    if start%2048==0:print('SOURCE',key,start,len(records),flush=True)
            handle.remove();features=torch.cat(values);assert features.shape==(9000,512) and torch.isfinite(features).all()
            torch.save(dict(features=features,protocol=protocol,checkpoint=meta['checkpoint'],checkpoint_sha256=digest(Path(meta['checkpoint']))),root/f'{key}_source.pt')
            del model;torch.cuda.empty_cache()
    write(root/'source_status.json',dict(state='complete'))

@torch.no_grad()
def evaluate_prediction(pred,labels,gallery):return evaluate(nn.Identity(),pred.float(),labels,gallery)

@torch.no_grad()
def mean_rank(pred,gallery,labels):
    scores=pred.float()@gallery.T;target=scores.gather(1,labels[:,None]);return float((1+(scores>target).sum(1)+.5*((scores==target).sum(1)-1)).mean())

def run(root,mode):
    write(root/f'{mode}_tests.json',dict(result=tests()))
    dest=root/mode;dest.mkdir(exist_ok=False)
    for fold in [1,2]:
        paired=None
        for arm in ['baseline','identity']:
            c,_,_,labels,meta=load(fold,arm);cfg=json.loads((ROOT/f'fold_{fold:02d}_{arm}'/'config.json').read_text());p,targets,gallery,expected=native(fold,arm,c,meta)
            if paired is None:paired=c
            else:assert np.array_equal(paired['images'],c['images']) and paired['records']==c['records']
            xt=c['etest'].double();train=c['etrain'].double()
            with torch.no_grad():
                base_train=F.normalize(p(train.float()),dim=-1).double();base_test=F.normalize(p(xt.float()),dim=-1).double()
                zero=evaluate(p,xt.float(),labels,gallery)
                unit_zero=evaluate_prediction(base_test,labels,gallery)
            if mode=='stats':
                source=torch.load(root/f's{fold}_{arm}_source.pt',map_location='cpu',weights_only=False)
                assert source['checkpoint_sha256']==meta['checkpoint_sha256']
                ref=source['features'].double();mu_s=ref.mean(0);cs=cov(ref);eye=torch.eye(512,dtype=cs.dtype)
                shrink_s=.9*cs+.1*cs.diag().mean()*eye
            for budget in [50,200]:
                for seed in range(33,38):
                    indices=np.random.default_rng(seed).permutation(1000)[:budget];x=train[indices]
                    result=dict(zero=zero,unit_prediction_zero=unit_zero,support_indices=indices.tolist(),checkpoint=meta['checkpoint'],methods={})
                    write(root/f'{mode}_status.json',dict(state='running',fold=fold,arm=arm,budget=budget,seed=seed))
                    if mode=='stats':
                        # Function sees EEG features only; support image IDs are not used.
                        mt=x.mean(0);ct=cov(x);eye=torch.eye(512,dtype=ct.dtype)
                        ratio=(cs.diag().clamp_min(1e-12)/ct.diag().clamp_min(1e-12)).sqrt().clamp(.5,2.)
                        transforms={'mean':eye,'diagonal':torch.diag(ratio),'shrink_coral':power(.9*ct+.1*ct.diag().mean()*eye,-.5)@power(shrink_s,.5),'coral_identity_regularizer':power(ct+eye,-.5)@power(cs+eye,.5)}
                        for method,a in transforms.items():
                            adapted=(xt-mt)@a+mu_s
                            assert torch.isfinite(adapted).all()
                            with torch.no_grad():pred=p(adapted.float())
                            result['methods'][method]=dict(metrics=evaluate_prediction(pred,labels,gallery),relative_shift=float(((adapted-xt).norm(dim=1)/xt.norm(dim=1).clamp_min(1e-12)).mean()),mapping_condition=float(torch.linalg.cond(a)),target_covariance_rank=int(torch.linalg.matrix_rank(ct)))
                            torch.save(dict(target_mean=mt,source_mean=mu_s,mapping=a),dest/f's{fold}_{arm}_n{budget}_seed{seed}_{method}.pt')
                    else:
                        fitidx=indices[:int(.8*budget)];dev=indices[int(.8*budget):];candidates=np.setdiff1d(np.arange(1000),fitidx);devlabels=torch.tensor(np.searchsorted(candidates,dev))
                        assert not np.intersect1d(fitidx,dev).size
                        for method in ['ridge','residual_ridge']:
                            residual=method=='residual_ridge';y=targets.double()-(base_train if residual else 0)
                            ranks={}
                            for lam in [1,10,100]:
                                f=ridge_fit(train[fitidx],y[fitidx],lam);pred=ridge_predict(f,train[dev])+(base_train[dev] if residual else 0)
                                ranks[lam]=mean_rank(pred,targets[candidates],devlabels)
                            lam=min(ranks,key=lambda z:(ranks[z],-z));f=ridge_fit(x,y[indices],lam)
                            pred=ridge_predict(f,xt)+(base_test if residual else 0)
                            assert torch.isfinite(pred).all()
                            result['methods'][method]=dict(metrics=evaluate_prediction(pred,labels,gallery),lambda_selected=lam,development_mean_ranks=ranks,fit_indices=fitidx.tolist(),development_indices=dev.tolist(),candidate_indices=candidates.tolist())
                            torch.save(f,dest/f's{fold}_{arm}_n{budget}_seed{seed}_{method}.pt')
                    write(dest/f's{fold}_{arm}_n{budget}_seed{seed}.json',result);print('DONE',mode,fold,arm,budget,seed,flush=True)
    write(root/f'{mode}_status.json',dict(state='complete'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['source','stats','ridge'],required=True);a=p.parse_args();torch.set_num_threads(4)
    try:
        if a.mode=='source':sources(a.root)
        else:run(a.root,a.mode)
    except Exception as exc:
        write(a.root/f'{a.mode}_status.json',dict(state='failed',error=repr(exc)));raise
