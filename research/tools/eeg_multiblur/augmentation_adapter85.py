"""Pinned v34 noise comparison: adapter dropout .85, all other computation unchanged.
Based on source 7b3a0a28b72ad2833e2da15b0b0fdb65a95513a3.
"""
import argparse,json,time,hashlib,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,ConcatDataset,DistributedSampler
from data import DatasetEEG
from reference_model import ReferenceModel
from neuraltrain.losses.losses import ClipLoss
from recovery_ablation import metrics, ProjectedHead
from reve_model import optimizer_groups

from train_augmentation import TrainAugmentation

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
DATA=BASE/'reve_v13_reference'

class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__();self.query=nn.Parameter(torch.zeros(1,1,512));nn.init.normal_(self.query,std=.02)
        self.norm=nn.LayerNorm(512);self.attention=nn.MultiheadAttention(512,4,batch_first=True,dropout=0.)
        self.gate=nn.Parameter(torch.tensor(.1))
    def forward(self,tokens):
        x=self.norm(tokens);q=self.query.expand(len(x),-1,-1)
        a,_=self.attention(q,x,x,need_weights=False)
        return tokens.mean(1)+self.gate*a[:,0]

class AblationModel(ReferenceModel):
    def __init__(self,manifest,arm):
        super().__init__(manifest)
        self.recovery=ProjectedHead()
        if arm in ('dropout','noise','feature_dropout'):self.image_adapter[2].p=.2
        if arm=='noise':self.image_adapter[2].p=.85
        self.augmentation=TrainAugmentation(arm)
        self.pool=None
        if arm=='attention':
            state=torch.get_rng_state();self.pool=AttentionPool();torch.set_rng_state(state)
    def forward(self,x,blur=None):
        x=self.augmentation.signal(self.preprocess(x));pos=self.positions.to(x).unsqueeze(0).expand(len(x),-1,-1)
        tokens=self.encoder(x,pos=pos,return_output=True)[-1]
        pooled=tokens.mean(1) if self.pool is None else self.pool(tokens)
        h=self.blur_head(self.augmentation.features(pooled));q=self.recovery(h)
        return q if blur is None else (h,q,F.normalize(self.image_adapter(blur),dim=-1))

class PairedBatches:
    def __init__(self,datasets):
        ids=np.concatenate([d.ids for d in datasets]);subjects=np.concatenate([d.subjects for d in datasets]);_,sid=np.unique(subjects,return_inverse=True)
        order=np.lexsort((ids,sid));keys=np.stack([sid[order],ids[order]],1)
        boundaries=np.r_[0,np.flatnonzero(np.any(keys[1:]!=keys[:-1],1))+1,len(ids)]
        rng=np.random.default_rng(33);self.partner=np.full(len(ids),-1,dtype=np.int64)
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            g=rng.permutation(order[lo:hi]);assert len(g)>1,'No repeat for a subject-image group'
            self.partner[g]=np.roll(g,1)
        assert (self.partner!=np.arange(len(ids))).all()
        assert np.array_equal(ids,ids[self.partner]) and np.array_equal(sid,sid[self.partner])
        self.epoch=0;self.steps=len(ids)//256
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.steps
    def __iter__(self):
        g=torch.Generator().manual_seed(33+self.epoch);order=torch.randperm(len(self.partner),generator=g).numpy()
        for i in range(self.steps):
            a=order[i*128:(i+1)*128];yield np.stack([a,self.partner[a]],1).reshape(-1).tolist()

@torch.no_grad()
def evaluate(model):
    model.eval();ds=DatasetEEG(DATA,'test',aux=False);pred=[]
    for b in DataLoader(ds,batch_size=256,num_workers=2):pred.append(model(b['eeg'].cuda()).cpu().numpy())
    return metrics(np.concatenate(pred),ds.ids,ds.subjects,np.asarray(ds.clean))

def objective(model,b,arm,criterion):
    h,q,a=model(b['eeg'].cuda(),b['blur'].cuda());c=b['clean'].cuda()
    lc=criterion(q,c);lb=criterion(h,a);avg=lc.new_zeros(())
    if arm=='repeat':
        assert torch.equal(b['id'][::2],b['id'][1::2])
        assert list(b['subject'][::2])==list(b['subject'][1::2])
        avg=criterion((q[::2]+q[1::2])*.5,c[::2])
    return lc+lb+.1*avg,{'clean':lc.item(),'blur':lb.item(),'repeat_mean':avg.item()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['noise']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    root=args.root/args.arm;root.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((BASE/'reve_v14_trainval_public_selection/training/config.json').read_text());cfg.update(arm=args.arm,baseline='B ProjectedHead; full-parameter training from pretrained REVE; original diagonal clean+blur loss; matched baseline arm',recovery_head='ProjectedHead; no outer raw-h residual',only_change={'baseline':'B head matched control; image adapter dropout .85','dropout':'image adapter dropout .85 -> .2','paired_control':'128 genuine subject-image pairs per batch, no auxiliary loss','repeat':'paired_control + .1 mean-prediction clean contrastive loss','attention':'mean pooling + gated 4-head learned-query attention','noise':'B head; image adapter dropout=.85; post-zscore Gaussian noise std=.02 trial probability=.5; matched v34 noise comparison','feature_dropout':'B-head dropout=.2 baseline + pooled EEG feature dropout p=.1'}[args.arm],reference_root=str(BASE/'reve_v14_trainval_public_selection'),run_root=str(root),test_role='public test single Top5 model selection, not independent test',runtime='single CUDA process; v14 used one-rank DDP, no cross-rank loss changes')
    random.seed(33);np.random.seed(33);torch.manual_seed(33);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((DATA/'data/manifest.json').read_text());model=AblationModel(manifest,args.arm).cuda()
    datasets=[DatasetEEG(DATA,s,aux=True,local_aux=False) for s in ['train','val']];ds=ConcatDataset(datasets)
    if args.arm in ['repeat','paired_control']:
        sampler=PairedBatches(datasets);loader=DataLoader(ds,batch_sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True)
    else:
        sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=33,drop_last=True)
        loader=DataLoader(ds,batch_size=256,sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True)
    opt=torch.optim.AdamW(optimizer_groups(model,cfg),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[cfg['encoder_lr'],cfg['lr']],total_steps=cfg['epochs']*len(loader),pct_start=.1,anneal_strategy='cos')
    criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False).cuda()
    if args.preflight:
        # Common initialization must match original reference, including adapter.
        torch.manual_seed(33);reference=ReferenceModel(manifest);reference.recovery=ProjectedHead()
        for k,v in reference.state_dict().items():assert torch.equal(v,model.state_dict()[k].cpu()),k
        del reference
        model.train();report=[]
        for step,b in enumerate(loader):
            opt.zero_grad();loss,parts=objective(model,b,args.arm,criterion);assert torch.isfinite(loss);loss.backward()
            grads={}
            for name,module in [('encoder',model.encoder),('projection',model.blur_head),('adapter',model.image_adapter),('recovery',model.recovery)]+([('pool',model.pool)] if model.pool is not None else []):
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
                grads[name]=sum(p.grad.abs().sum().item() for p in module.parameters());assert grads[name]>0
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();report.append({'loss':loss.item(),'parts':parts,'gradient_l1':grads})
            if step==2:break
        write={'passed':True,'initialization_matches_reference':True,'real_steps':report,'training_samples':len(ds),'steps_per_epoch':len(loader),'note':'Preflight updates discarded'}
        (root/'preflight.json').write_text(json.dumps(write,indent=2));print('PREFLIGHT PASSED',flush=True);return
    assert not (root/'history.jsonl').exists();(root/'config.json').write_text(json.dumps(cfg,indent=2))
    best=-1.;stale=0;best_epoch=None
    def save(name,epoch,selection):
        torch.save({'model':model.state_dict(),'epoch':epoch,'manifest':manifest,'config':cfg,'selection':selection,'optimizer':opt.state_dict(),'scheduler':sched.state_dict()},root/name)
    for epoch in range(cfg['epochs']):
        sampler.set_epoch(epoch);model.train();total=np.zeros(4);start=time.time()
        for step,b in enumerate(loader):
            opt.zero_grad(set_to_none=True);loss,parts=objective(model,b,args.arm,criterion);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step()
            total+=np.array([loss.item(),parts['clean'],parts['blur'],parts['repeat_mean']])
            if step%100==0:print(json.dumps({'arm':args.arm,'epoch':epoch+1,'step':step,'steps':len(loader),'parts':parts,'loss':loss.item()}),flush=True)
        selection=evaluate(model);score=selection['single_trial']['top5']
        if score>best:best=score;stale=0;best_epoch=epoch+1;save('best.pt',epoch+1,selection)
        else:stale+=1
        save('last.pt',epoch+1,selection)
        row={'epoch':epoch+1,'loss':(total/len(loader)).tolist(),'selection':selection,'best_epoch':best_epoch,'best_top5':best,'stale':stale,'seconds':time.time()-start}
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        (root/'status.json').write_text(json.dumps({'state':'training','epoch':epoch+1,'best_epoch':best_epoch}));print(json.dumps(row),flush=True)
        if stale>=cfg['patience']:break
    ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    (root/'result.json').write_text(json.dumps({'best_epoch':ck['epoch'],'selection':ck['selection'],'config':cfg},indent=2));(root/'status.json').write_text(json.dumps({'state':'complete','epochs':epoch+1,'best_epoch':best_epoch}))
if __name__=='__main__':main()
