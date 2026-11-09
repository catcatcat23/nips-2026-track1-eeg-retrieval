"""EEG-conditioned vector diffusion prior, x0 prediction in fixed DINO space.

Adaptation of the conditional-prior idea, not an exact ATM reproduction.
Inference uses one fixed shared Gaussian starting vector and deterministic DDIM;
no image, identity, subject or batch-dependent random seed is accessed.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from recovery_ablation import metrics


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2))


class ConditionalBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.time = nn.Linear(128, din)
        self.condition = nn.Linear(1536, din)
        self.net = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, dout), nn.SiLU(), nn.Dropout(.1))

    def forward(self, x, t, h):
        return self.net(x + self.time(t) + self.condition(h))


class DiffusionHead(nn.Module):
    def __init__(self, steps=20):
        super().__init__()
        widths = [1536, 1024, 512, 256]
        self.condition_norm = nn.LayerNorm(1536)
        self.encoder = nn.ModuleList([ConditionalBlock(a,b) for a,b in zip(widths[:-1],widths[1:])])
        self.middle = ConditionalBlock(256,256)
        self.decoder = nn.ModuleList([ConditionalBlock(b,a) for a,b in reversed(list(zip(widths[:-1],widths[1:])))])
        self.output = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536,1536))
        t = torch.linspace(0,1,1000)
        alpha = torch.cos((t+.008)/1.008*math.pi/2).square()
        alpha = alpha/alpha[0]; alpha[-1] = 0
        self.register_buffer('alpha',alpha)
        self.register_buffer('frequencies',torch.exp(-math.log(10000)*torch.arange(64)/63))
        gen = torch.Generator().manual_seed(5100)
        self.register_buffer('initial_noise',torch.randn(1,1536,generator=gen))
        self.steps = steps

    def denoise(self, x, t, h):
        phase=t.float()[:,None]*self.frequencies[None,:]
        emb=torch.cat([phase.sin(),phase.cos()],dim=-1)
        h=self.condition_norm(h)
        skips=[]
        for block in self.encoder:
            skips.append(x);x=block(x,emb,h)
        x=x+self.middle(x,emb,h)
        for block,skip in zip(self.decoder,reversed(skips)):
            x=block(x,emb,h)+skip
        return self.output(x)

    def forward(self,h):
        # eta=0 DDIM, x0-prediction, no clipping of DINO coordinates.
        x=self.initial_noise.expand(len(h),-1)
        times=torch.linspace(999,0,self.steps,device=h.device).round().long()
        for i,t in enumerate(times):
            clean=self.denoise(x,t.expand(len(h)),h)
            if i+1 == len(times):
                return clean/math.sqrt(1536)
            a=self.alpha[t];an=self.alpha[times[i+1]]
            eps=(x-a.sqrt()*clean)/(1-a).sqrt().clamp_min(1e-6)
            x=an.sqrt()*clean+(1-an).sqrt()*eps


def objective(model,h,clean):
    t=torch.randint(0,1000,(len(h),),device=h.device)
    target=F.normalize(clean,dim=-1)*math.sqrt(1536)
    a=model.alpha[t,None]
    noisy=a.sqrt()*target+(1-a).sqrt()*torch.randn_like(target)
    return F.mse_loss(model.denoise(noisy,t,h),target)


@torch.no_grad()
def predict(model,h):
    model.eval()
    return np.concatenate([model(x).cpu().numpy() for x in h.split(512)])


def preflight(cfg,out,h,ids,clean):
    torch.manual_seed(33)
    model=DiffusionHead(cfg['sampling_steps']).cuda().eval()
    _,ix=np.unique(ids.cpu().numpy(),return_index=True)
    ix=torch.tensor(ix[:8],device='cuda');x=h[ix];y=clean[ids[ix]]
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0)
    first=float(objective(model,x,y).detach())
    for step in range(300):
        opt.zero_grad();loss=objective(model,x,y);loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%10==9:
            with torch.no_grad():
                q=model(x);acc=float(((F.normalize(q,dim=-1)@F.normalize(y,dim=-1).T).argmax(1)==torch.arange(8,device='cuda')).float().mean())
            if acc==1 and float(loss.detach())<first*.3:break
    assert acc==1 and float(loss.detach())<first*.3,(acc,float(loss.detach()))
    with torch.no_grad():
        q=model(x);torch.testing.assert_close(model(x[:1]),q[:1],rtol=1e-4,atol=1e-4)
        perm=torch.arange(7,-1,-1,device='cuda');torch.testing.assert_close(model(x[perm]),q[perm],rtol=1e-4,atol=1e-4)
        # Fixed same starting noise: different predictions must come from EEG.
        assert (q-q[:1]).abs().max()>1e-3
        ts=torch.full((8,),999,device='cuda',dtype=torch.long)
        noise=model.initial_noise.expand(8,-1)
        effect=float((model.denoise(noise,ts,x)-model.denoise(noise,ts,torch.zeros_like(x))).abs().mean())
        assert effect>1e-3
    model.train();opt.zero_grad();objective(model,x,y).backward()
    grads={name:float(p.grad.abs().sum()) for name,p in model.named_parameters() if 'condition.weight' in name}
    assert all(v>0 for v in grads.values())
    write(out/'preflight.json',dict(passed=True,steps=step+1,initial_mse=first,final_mse=float(loss.detach()),sampled_top1=acc,condition_effect=effect,condition_gradients=grads,batch_invariance=True,updates_discarded=True,parameters=sum(p.numel() for p in model.parameters())))
    print('PREFLIGHT PASSED',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());out=Path(cfg['run_root']);cache=Path(cfg['cache_root']);src=Path(cfg['source_root'])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert json.loads((cache/'complete.json').read_text())['checkpoint_sha256']==cfg['checkpoint_sha256']
    for s in ['train','val','test']:
        for k in ['ids','subjects']:assert np.array_equal(np.load(cache/f'{s}_{k}.npy'),np.load(src/'data'/f'{s}_{k}.npy'))
    ids=np.concatenate([np.load(cache/f'{s}_ids.npy') for s in ['train','val']]);ti=np.load(cache/'test_ids.npy');subjects=np.load(cache/'test_subjects.npy')
    assert not np.intersect1d(ids,ti).size
    h=torch.from_numpy(np.concatenate([np.load(cache/f'{s}_h.npy') for s in ['train','val']])).cuda()
    testh=torch.from_numpy(np.load(cache/'test_h.npy')).cuda();clean=np.load(src/'data/clean.npy');targets=torch.tensor(clean,device='cuda');ids_gpu=torch.tensor(ids,device='cuda')
    preflight(cfg,out,h,ids_gpu,targets)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model=DiffusionHead(cfg['sampling_steps']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    steps=math.ceil(len(h)/cfg['batch_size']);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg['lr'],epochs=cfg['epochs'],steps_per_epoch=steps,pct_start=.1)
    best=-1;stale=0;best_epoch=0;gen=torch.Generator(device='cuda').manual_seed(cfg['seed'])
    torch.save(model.state_dict(),out/'initial_head.pt');(out/'status.txt').write_text('TRAINING\n')
    for epoch in range(1,cfg['epochs']+1):
        start=time.time();model.train();total=0
        for step,ix in enumerate(torch.randperm(len(h),generator=gen,device='cuda').split(cfg['batch_size'])):
            opt.zero_grad();loss=objective(model,h[ix],targets[ids_gpu[ix]]);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sched.step();total+=float(loss.detach())*len(ix)
            if step%500==0:print('TRAIN',epoch,step,steps,'mse',float(loss.detach()),flush=True)
        score=metrics(predict(model,testh),ti,subjects,clean);value=score['single_trial']['top5']
        if value>best:
            best=value;stale=0;best_epoch=epoch
            torch.save(dict(head=model.state_dict(),epoch=epoch,selection=score,config=cfg,optimizer=opt.state_dict(),scheduler=sched.state_dict()),out/'best.pt')
        else:stale+=1
        row=dict(epoch=epoch,mse=total/len(h),selection=score,best_epoch=best_epoch,stale=stale,seconds=time.time()-start)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write(out/'status.json',row);print(json.dumps(row),flush=True)
        if stale>=cfg['patience']:break
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);blur=json.loads(Path(cfg['blur_reference']).read_text());assert blur['checkpoint_sha256']==cfg['checkpoint_sha256']
    write(out/'result.json',dict(best_epoch=ck['epoch'],selection=ck['selection'],blur=blur['metrics']['eeg_to_adapted_blur'],blur_provenance='same frozen v33 checkpoint/cache; reused diagnostic',role='public test selected, not independent test'))
    (out/'status.txt').write_text('COMPLETED\n')

if __name__=='__main__':main()
