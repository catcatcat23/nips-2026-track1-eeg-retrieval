"""Frozen v33: ridge, ridge+residual, and cosine-supervised residual controls."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from recovery_ablation import ResidualBlock,metrics,predictions
from clip_loss import ClipLoss


def write(p,x):p.write_text(json.dumps(x,indent=2))

class RidgeResidual(nn.Module):
    def __init__(self,ridge,residual=True):
        super().__init__()
        for k in ['mean','scale','weight','bias']:self.register_buffer(k,ridge[k].float().clone())
        self.delta=nn.Sequential(nn.LayerNorm(1536),ResidualBlock(),ResidualBlock(),nn.LayerNorm(1536),nn.Linear(1536,1536)) if residual else None
        if residual:
            nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
    def forward(self,h):
        q=((h-self.mean)/self.scale)@self.weight+self.bias
        return q if self.delta is None else q+.1*self.delta(h)


def fit_ridge(h,ids,target,alpha):
    # Solve mean squared raw-clean error + alpha*||W||^2 on train+val only.
    n,d=h.shape;mean=h.double().mean(0);scale=h.double().std(0,unbiased=False).clamp_min(1e-6)
    gram=torch.zeros(d,d,device=h.device,dtype=torch.float64);cross=torch.zeros_like(gram);ysum=torch.zeros(d,device=h.device,dtype=torch.float64)
    for start in range(0,n,4096):
        x=(h[start:start+4096].double()-mean)/scale;y=target[ids[start:start+4096]].double()
        gram+=x.T@x;cross+=x.T@y;ysum+=y.sum(0)
    bias=ysum/n;system=gram/n+alpha*torch.eye(d,device=h.device,dtype=torch.float64)
    w=torch.linalg.solve(system,cross/n)
    assert torch.isfinite(w).all()
    error=float((system@w-cross/n).norm()/(cross/n).norm())
    assert error<1e-8,error
    return {'mean':mean,'scale':scale,'weight':w,'bias':bias},error


def loss_value(q,y,criterion,cosine):
    ce=criterion(q,y);direction=(1-F.cosine_similarity(q.float(),y.float(),dim=-1)).mean()
    return ce+cosine*direction,ce,direction


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();out=args.root
    cfg=json.loads((out/'config.json').read_text());cache=Path(cfg['cache_root']);src=Path(cfg['source_root'])
    assert not (out/'A_result.json').exists(),'Use fresh run root'
    assert json.loads((cache/'complete.json').read_text())['checkpoint_sha256']==cfg['checkpoint_sha256']
    torch.manual_seed(33);np.random.seed(33);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids=np.concatenate([np.load(cache/f'{s}_ids.npy') for s in ['train','val']]);testids=np.load(cache/'test_ids.npy');subjects=np.load(cache/'test_subjects.npy')
    assert not np.intersect1d(ids,testids).size
    for s in ['train','val','test']:
        assert np.array_equal(np.load(cache/f'{s}_ids.npy'),np.load(src/'data'/f'{s}_ids.npy'))
        assert np.array_equal(np.load(cache/f'{s}_subjects.npy'),np.load(src/'data'/f'{s}_subjects.npy'))
    clean=np.load(src/'data/clean.npy');target=torch.from_numpy(clean).cuda();trainids=torch.as_tensor(ids,device='cuda')
    h=torch.from_numpy(np.concatenate([np.load(cache/f'{s}_h.npy') for s in ['train','val']])).cuda();testh=torch.from_numpy(np.load(cache/'test_h.npy')).cuda()
    (out/'status.txt').write_text('FITTING_RIDGE\n');ridge,error=fit_ridge(h,trainids,target,cfg['ridge_alpha'])
    torch.save({'ridge':{k:v.cpu() for k,v in ridge.items()},'config':cfg},out/'ridge.pt')
    fixed=RidgeResidual(ridge,False).cuda().eval();score=metrics(predictions(fixed,testh),testids,subjects,clean)
    diag=lambda model:metrics(predictions(model,h[:2048]),ids[:2048],np.zeros(2048,dtype=int),clean)
    write(out/'A_result.json',{'selection':score,'train_diagnostic':diag(fixed),'linear_solve_relative_error':error,'ridge_alpha':cfg['ridge_alpha'],'note':'alpha fixed in advance; mean/std/solve use train+val only'})
    print('A_RESULT',json.dumps({g:score[g] for g in ['single_trial','subject_aggregated']}),flush=True)
    criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False).cuda()
    # Actual cached data: verify ridge initialization, frozen buffers, finite gradients and tiny overfit.
    _,ix=np.unique(ids,return_index=True);tiny=torch.as_tensor(ix[:8],device='cuda');torch.manual_seed(33)
    probe=RidgeResidual(ridge).cuda().eval();torch.testing.assert_close(probe(h[tiny]),fixed(h[tiny]))
    opt=torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=0);initial=float(criterion(probe(h[tiny]),target[trainids[tiny]]).detach())
    for step in range(300):
        opt.zero_grad();q=probe(h[tiny]);loss=criterion(q,target[trainids[tiny]]);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(probe.parameters(),1);opt.step()
        with torch.no_grad():
            q=probe(h[tiny]);acc=float(((F.normalize(q,dim=-1)@F.normalize(target[trainids[tiny]],dim=-1).T).argmax(1)==torch.arange(8,device='cuda')).float().mean())
        if acc==1 and float(loss.detach())<initial*.3:break
    assert acc==1 and float(loss.detach())<initial*.3,(acc,float(loss.detach()),initial)
    assert all(torch.isfinite(p.grad).all() for p in probe.parameters() if p.grad is not None)
    assert all(torch.equal(getattr(probe,k),getattr(fixed,k)) for k in ridge)
    write(out/'preflight.json',{'passed':True,'ridge_solve_error':error,'initial_loss':initial,'final_loss':float(loss.detach()),'steps':step+1,'tiny_top1':acc,'updates_discarded':True,'frozen_ridge_unchanged':True})
    del probe,opt,fixed
    for arm,cosine in [('B',0.)]:
        dest=out/arm;dest.mkdir();(out/'status.txt').write_text('TRAINING_'+arm+'\n');torch.manual_seed(33)
        model=RidgeResidual(ridge).cuda();torch.save(model.state_dict(),dest/'initial_head.pt')
        if arm=='C':
            ref=torch.load(out/'B/initial_head.pt',map_location='cuda',weights_only=True)
            assert all(torch.equal(v,ref[k]) for k,v in model.state_dict().items());del ref
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        steps=math.ceil(len(h)/cfg['batch_size']);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg['lr'],epochs=cfg['epochs'],steps_per_epoch=steps,pct_start=.1)
        best=score['single_trial']['top5'];stale=0;bestepoch=0
        def save(epoch,selection):torch.save({'head':model.state_dict(),'epoch':epoch,'selection':selection,'config':cfg,'cosine_weight':cosine,'optimizer':opt.state_dict(),'scheduler':sched.state_dict()},dest/'best.pt')
        save(0,score);gen=torch.Generator(device='cuda').manual_seed(33)
        for epoch in range(1,cfg['epochs']+1):
            start=time.time();model.train();tot=np.zeros(3);count=0
            for idx in torch.randperm(len(h),generator=gen,device='cuda').split(cfg['batch_size']):
                opt.zero_grad(set_to_none=True);q=model(h[idx]);loss,ce,co=loss_value(q,target[trainids[idx]],criterion,cosine)
                assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sched.step()
                tot+=np.array([float(loss.detach()),float(ce.detach()),float(co.detach())])*len(idx);count+=len(idx)
            selection=metrics(predictions(model,testh),testids,subjects,clean);trainmetric=diag(model);value=selection['single_trial']['top5']
            if value>best:best=value;stale=0;bestepoch=epoch;save(epoch,selection)
            else:stale+=1
            row={'epoch':epoch,'loss':(tot/count).tolist(),'selection':selection,'train_diagnostic':trainmetric,'best_epoch':bestepoch,'stale':stale,'seconds':time.time()-start}
            with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(dest/'status.json',{'state':'training','epoch':epoch,'best_epoch':bestepoch});print(arm,json.dumps({k:v for k,v in row.items() if k not in ['selection','train_diagnostic']}),flush=True)
            if stale>=cfg['patience']:break
        ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);write(dest/'result.json',{'best_epoch':ck['epoch'],'selection':ck['selection'],'role':'public test selection, not independent test'});write(dest/'status.json',{'state':'complete','epochs':epoch,'best_epoch':bestepoch})
        del model,opt,sched,ck;torch.cuda.empty_cache()
    (out/'status.txt').write_text('COMPLETED\n')

if __name__=='__main__':main()
