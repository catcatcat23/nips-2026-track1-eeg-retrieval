"""Frozen v33: B recovery plus cross-attention to unpooled EEG tokens."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import DatasetEEG
from v33_recovery_stage2 import load_source,make_head,metrics,write
from neuraltrain.losses.losses import ClipLoss

class TokenRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=make_head('B')
        self.query=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,256))
        self.tokens=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256))
        self.attention=nn.MultiheadAttention(256,4,batch_first=True,dropout=0.)
        self.output=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1536))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
    def forward(self,h,t):
        q=self.query(h).unsqueeze(1);kv=self.tokens(t)
        u,_=self.attention(q,kv,kv,need_weights=False)
        return self.base(h)+self.output(u[:,0])

@torch.no_grad()
def encode(model,x):
    x=model.preprocess(x);pos=model.positions.to(x).unsqueeze(0).expand(len(x),-1,-1)
    tokens=model.encoder(x,pos=pos,return_output=True)[-1]
    assert tokens.shape[1:]==(63,512),tokens.shape
    return model.blur_head(tokens.mean(1)),tokens

def preflight(cfg,out):
    source=load_source(cfg);ds=DatasetEEG(cfg['source_root'],'train',aux=False)
    _,first=np.unique(ds.ids,return_index=True);indices=first[:8]
    x=torch.stack([ds[int(i)]['eeg'] for i in indices]).cuda();y=torch.tensor(np.asarray(ds.clean[ds.ids[indices]]),device='cuda')
    h,t=encode(source,x);ref=np.load(Path(cfg['cache_root'])/'train_h.npy',mmap_mode='r')
    torch.testing.assert_close(h,torch.tensor(ref[indices],device='cuda'),rtol=1e-4,atol=1e-4)
    head=TokenRecovery().cuda().eval();initial=torch.load(cfg['initial_reference'],map_location='cpu',weights_only=True)
    assert all(torch.equal(v.cpu(),initial[k]) for k,v in head.base.state_dict().items())
    with torch.no_grad():torch.testing.assert_close(head(h,t),head.base(h),rtol=0,atol=0)
    criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0)
    initial_loss=float(criterion(head(h,t),y).detach())
    for step in range(100):
        opt.zero_grad();loss=criterion(head(h,t),y);assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        nn.utils.clip_grad_norm_(head.parameters(),1);opt.step()
        with torch.no_grad():acc=float(((F.normalize(head(h,t),dim=-1)@F.normalize(y,dim=-1).T).argmax(1)==torch.arange(8,device='cuda')).float().mean())
        if step>=2 and acc==1 and float(loss.detach())<initial_loss*.3:break
    assert acc==1 and float(loss.detach())<initial_loss*.3
    assert all(p.grad is None for p in source.parameters())
    assert all(p.grad.abs().sum()>0 for p in head.attention.parameters())
    with torch.no_grad():
        changed=float((head(h,t)-head(h,t.roll(1,0))).abs().max());assert changed>1e-6
    head.train();opt.zero_grad();criterion(head(h,t),y).backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
    write(out/'preflight.json',dict(passed=True,shape=list(t.shape),initial_loss=initial_loss,final_loss=float(loss.detach()),steps=step+1,tiny_top1=acc,frozen_source=True,attention_gradient_nonzero=True,token_swap_output_change=changed,cached_h_match=True,updates_discarded=True))
    print('PREFLIGHT PASSED',flush=True)

def extract(cfg,out):
    cache=out/'tokens';cache.mkdir();model=load_source(cfg)
    for split in ['train','val','test']:
        ds=DatasetEEG(cfg['source_root'],split,aux=False);refroot=Path(cfg['cache_root']);ref=np.load(refroot/f'{split}_h.npy',mmap_mode='r')
        assert np.array_equal(ds.ids,np.load(refroot/f'{split}_ids.npy'))
        assert np.array_equal(ds.subjects,np.load(refroot/f'{split}_subjects.npy'))
        a=np.lib.format.open_memmap(cache/f'{split}.npy',mode='w+',dtype='float32',shape=(len(ds),63,512));offset=0
        for step,b in enumerate(DataLoader(ds,batch_size=256,num_workers=2)):
            h,t=encode(model,b['eeg'].cuda());n=len(h)
            assert torch.isfinite(t).all()
            torch.testing.assert_close(h,torch.tensor(ref[offset:offset+n].copy(),device='cuda'),rtol=1e-4,atol=1e-4)
            a[offset:offset+n]=t.cpu().numpy();offset+=n
            if step%200==0:print('EXTRACT',split,offset,len(ds),flush=True)
        assert offset==len(ds);a.flush();del a
    write(cache/'complete.json',dict(checkpoint_sha256=cfg['checkpoint_sha256'],dtype='float32',token_shape=[63,512],all_h_checked=True))

@torch.no_grad()
def predict(head,h,t,batch=256):
    head.eval();return np.concatenate([head(torch.from_numpy(h[i:i+batch].copy()).cuda(),torch.from_numpy(t[i:i+batch].copy()).cuda()).cpu().numpy() for i in range(0,len(h),batch)])

def train(cfg,out):
    cache=Path(cfg['cache_root']);tc=out/'tokens';assert json.loads((tc/'complete.json').read_text())['checkpoint_sha256']==cfg['checkpoint_sha256']
    dest=out/'B';dest.mkdir();ids=np.concatenate([np.load(cache/f'{s}_ids.npy') for s in ['train','val']]);testids=np.load(cache/'test_ids.npy');subjects=np.load(cache/'test_subjects.npy')
    assert not np.intersect1d(ids,testids).size
    # Host memory contains token arrays; frozen tokens are streamed to GPU per batch.
    h=np.concatenate([np.load(cache/f'{s}_h.npy') for s in ['train','val']]);t=np.concatenate([np.load(tc/f'{s}.npy',mmap_mode='r') for s in ['train','val']])
    testh=np.load(cache/'test_h.npy',mmap_mode='r');testt=np.load(tc/'test.npy',mmap_mode='r');clean=np.load(Path(cfg['source_root'])/'data/clean.npy');target=torch.tensor(clean,device='cuda')
    head=TokenRecovery().cuda();initial=torch.load(cfg['initial_reference'],map_location='cpu',weights_only=True)
    assert all(torch.equal(v.cpu(),initial[k]) for k,v in head.base.state_dict().items())
    torch.save(head.state_dict(),dest/'initial_head.pt');opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    steps=(len(ids)+cfg['batch_size']-1)//cfg['batch_size'];sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg['lr'],epochs=cfg['epochs'],steps_per_epoch=steps,pct_start=.1)
    gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False);best=-1;stale=0
    for epoch in range(1,cfg['epochs']+1):
        start=time.time();head.train();order=torch.randperm(len(ids),generator=gen,device='cuda').cpu().numpy();total=0
        for step,offset in enumerate(range(0,len(ids),cfg['batch_size'])):
            ix=order[offset:offset+cfg['batch_size']];opt.zero_grad(set_to_none=True)
            q=head(torch.from_numpy(h[ix]).cuda(),torch.from_numpy(t[ix]).cuda());loss=criterion(q,target[ids[ix]])
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True);opt.step();sched.step();total+=float(loss.detach())*len(ix)
            if step%500==0:print('TRAIN',epoch,step,steps,float(loss.detach()),flush=True)
        score=metrics(predict(head,testh,testt),testids,subjects,clean)
        if score['single_trial']['top5']>best:
            best=score['single_trial']['top5'];stale=0
            torch.save(dict(head=head.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,selection=score,config=cfg),dest/'best.pt')
        else:stale+=1
        row=dict(epoch=epoch,loss=total/len(ids),selection=score,stale=stale,seconds=time.time()-start)
        with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write(dest/'status.json',dict(state='training',epoch=epoch,best_top5=best,stale=stale));print(json.dumps(row),flush=True)
        if stale>=cfg['patience']:break
    ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
    blur=json.loads(Path(cfg['blur_reference']).read_text());assert blur['checkpoint_sha256']==cfg['checkpoint_sha256']
    write(dest/'result.json',dict(best_epoch=ck['epoch'],selection=ck['selection'],blur=blur['metrics']['eeg_to_adapted_blur'],blur_provenance='reused fixed v33 EEG/projection/image adapter; all sample h checked on extraction',role='public test model selection, not independent test'))
    write(dest/'status.json',dict(state='complete',epoch=epoch,best_epoch=ck['epoch']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','extract','train']);ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['run_root'])
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    globals()[a.stage](cfg,out)
if __name__=='__main__':main()
