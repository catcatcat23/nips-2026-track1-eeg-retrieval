"""Image-disjoint Alljoined pretraining, matched THINGS transfer/control."""
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from alljoined_train import MixedData
from reference_model import ReferenceModel
from recovery_ablation import ProjectedHead, metrics
from data import DatasetEEG
from neuraltrain.losses.losses import ClipLoss

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v13_reference')
ROOT=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v31_subject_holdout_original_loss')

class Model(ReferenceModel):
    def __init__(self,manifest,channels):
        super().__init__(manifest)
        self.recovery=ProjectedHead()
        for domain,names in channels.items():self.register_buffer('pos_'+domain,self.encoder.get_positions(names))
    def forward(self,x,blur=None,domain='things'):
        x=self.preprocess(x);pos=getattr(self,'pos_'+domain).to(x).unsqueeze(0).expand(len(x),-1,-1)
        h=self.blur_head(self.encoder(x,pos=pos,return_output=True)[-1].mean(1))
        q=self.recovery(h)
        return q if blur is None else (h,q,F.normalize(self.image_adapter(blur),dim=-1))

def write(path,data):path.write_text(json.dumps(data,indent=2))

def partitions(ds):
    ids=np.concatenate([r['ids'] for r in ds.records])
    things=np.unique(ids[ds.domains['things']]);test=np.unique(np.load(BASE/'data/test_ids.npy'))
    rng=np.random.default_rng(33);held=rng.permutation(things)[:int(np.ceil(len(things)*.1))]
    external=np.unique(np.concatenate([ids[ix] for d,ix in ds.domains.items() if d!='things']))
    extra=np.setdiff1d(external,np.union1d(things,test));extraheld=rng.permutation(extra)[:int(np.ceil(len(extra)*.1))]
    valid_ids=np.union1d(held,extraheld)
    train={};valid={}
    for d,ix in ds.domains.items():
        assert not np.isin(ids[ix],test).any()
        mask=np.isin(ids[ix],valid_ids);train[d]=ix[~mask];valid[d]=ix[mask]
        assert len(train[d]) and len(valid[d])
    assert not np.intersect1d(ids[np.concatenate(list(train.values()))],ids[np.concatenate(list(valid.values()))]).size
    # A genuinely unseen-subject audit: train on 16/20 Alljoined-2025 subjects.
    records=json.loads(Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v18_alljoined/eeg_complete.json').read_text())['recordings']
    names=sorted(set(str(r['subject']) for r in records if r['dataset']=='Xu2025Alljoined'))
    held_subjects=sorted(np.random.default_rng(33).choice(names,4,replace=False).tolist())
    held_indices=[];start=sum(len(r['ids']) for r in ds.records[:2])
    for record in records:
        end=start+len(record['events'])
        if str(record['subject']) in held_subjects:held_indices.append(np.arange(start,end))
        start=end
    assert start==len(ids)
    all_held=np.concatenate(held_indices)
    for d in list(train):
        if d=='Xu2024Alljoined':del train[d];del valid[d];continue
        train[d]=np.setdiff1d(train[d],all_held);valid[d]=np.setdiff1d(valid[d],all_held)
    assert not np.intersect1d(train['Xu2025Alljoined'],all_held).size
    # Keep held-out images too for the pretraining audit; stage2 intentionally
    # sees THINGS train+val images, so its audit is unseen-subject only.
    held_test=all_held[np.isin(ids[all_held],valid_ids)]
    assert len(held_test)>0
    audit={'seed':33,'validation_image_ids':valid_ids.tolist(),'things_validation_image_ids':held.tolist(),'held_subjects':held_subjects,'held_audit_trials':len(held_test),'counts':{d:{'train':len(train[d]),'val':len(valid[d])} for d in train},'test_images':len(test),'rule':'16-subject Alljoined2025 pretrain only; other4 subjects never trained; source image split disjoint; stage2 uses full THINGS train+val per user'}
    return ids,train,valid,audit


class DomainBatches:
    def __init__(self,indices,batch,seed):self.indices=indices;self.batch=batch;self.seed=seed;self.epoch=0
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);batches=[]
        for d,ix in self.indices.items():
            order=rng.permutation(ix)
            batches.extend(order[i:i+self.batch].tolist() for i in range(0,len(order),self.batch) if len(order[i:i+self.batch])>1)
        rng.shuffle(batches);yield from batches
    def __len__(self):return sum((len(v)+self.batch-1)//self.batch-(len(v)%self.batch==1) for v in self.indices.values())

@torch.no_grad()
def validate(model,ds,indices,criterion):
    model.eval();result={}
    for d,ix in indices.items():
        preds=[];ids=[];total=np.zeros(2);count=0
        for b in DataLoader(Subset(ds,ix.tolist()),batch_size=256,num_workers=2):
            h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),d)
            lc=criterion(q,b['clean'].cuda());lb=criterion(h,a)
            total+=np.array([lc.item(),lb.item()])*len(q);count+=len(q)
            preds.append(q.cpu().numpy());ids.append(b['id'].numpy())
        ids=np.concatenate(ids);pred=np.concatenate(preds)
        m=metrics(pred,ids,np.zeros(len(ids),dtype=int),ds.clean)
        result[d]={'clean_loss':total[0]/count,'blur_loss':total[1]/count,'single_trial':m['single_trial'],'candidate_images':m['candidate_images']}
    return result

@torch.no_grad()
def test_once(model):
    model.eval();ds=DatasetEEG(BASE,'test',aux=False);pred=[]
    for b in DataLoader(ds,batch_size=256,num_workers=2):pred.append(model(b['eeg'].cuda()).cpu().numpy())
    return metrics(np.concatenate(pred),ds.ids,ds.subjects,np.asarray(ds.clean))


@torch.no_grad()
def evaluate_held_subjects(model,ds,audit):
    model.eval();records=json.loads(Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v18_alljoined/eeg_complete.json').read_text())['recordings']
    indices=[];subjects=[];start=sum(len(r['ids']) for r in ds.records[:2])
    for r in records:
        image_ids=np.array([i for _,i in r['events']]);end=start+len(image_ids)
        if str(r['subject']) in audit['held_subjects']:
            chosen=np.flatnonzero(np.isin(image_ids,audit['validation_image_ids']))
            indices.extend((start+chosen).tolist());subjects.extend([str(r['subject'])]*len(chosen))
        start=end
    pred=[];ids=[]
    for b in DataLoader(Subset(ds,indices),batch_size=256,num_workers=2):
        pred.append(model(b['eeg'].cuda(),domain='Xu2025Alljoined').cpu().numpy());ids.append(b['id'].numpy())
    return metrics(np.concatenate(pred),np.concatenate(ids),np.asarray(subjects),ds.clean)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','pretrain','control','transfer']);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    ROOT.mkdir(exist_ok=True);ds=MixedData();ids,train,val,audit=partitions(ds)
    if args.stage=='audit':write(ROOT/'split.json',audit);print(json.dumps(audit['counts']));return
    assert json.loads((ROOT/'split.json').read_text())==audit
    root=ROOT/args.stage;root.mkdir(exist_ok=True)
    torch.manual_seed(33);np.random.seed(33);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((BASE/'data/manifest.json').read_text());model=Model(manifest,ds.channels).cuda()
    source=None
    if args.stage=='transfer':
        assert json.loads((ROOT/'pretrain/status.json').read_text())['state']=='complete'
        path=ROOT/'pretrain/best.pt';source=torch.load(path,map_location='cpu',weights_only=False)
        assert source['epoch']>0;model.load_state_dict(source['model'],strict=True)
    domains=[d for d in train if d!='things'] if args.stage=='pretrain' else ['things']
    if args.stage!='pretrain':
        train['things']=ds.domains['things']  # User-authorized train+val, public test selection.
    batcher=DomainBatches({d:train[d] for d in domains},256,33)
    loader=DataLoader(ds,batch_sampler=batcher,num_workers=4,pin_memory=True,persistent_workers=True)
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},{'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')],'lr':1e-4}],weight_decay=.05)
    mode='min' if args.stage=='pretrain' else 'max'
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode=mode,factor=.5,patience=5,min_lr=1e-6)
    criterion=ClipLoss(norm_kind='y', temperature=False, symmetric=False).cuda()
    if args.preflight:
        model.train();report={}
        for d in domains:
            b=next(iter(DataLoader(Subset(ds,train[d][:256].tolist()),batch_size=256)))
            opt.zero_grad();h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),d)
            loss=criterion(q,b['clean'].cuda())+criterion(h,a);loss.backward()
            groups={}
            for name,module in [('encoder',model.encoder),('projection',model.blur_head),('adapter',model.image_adapter),('recovery',model.recovery)]:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
                groups[name]=sum(p.grad.abs().sum().item() for p in module.parameters());assert groups[name]>0
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            report[d]={'loss':loss.item(),'shape':list(b['eeg'].shape),'gradient_l1':groups}
        write(root/'preflight.json',report);print('PREFLIGHT PASSED',report,flush=True);return
    assert not (root/'history.jsonl').exists()
    cfg={'seed':33,'batch':256,'epochs':20 if args.stage=='pretrain' else 200,'patience':10 if args.stage=='pretrain' else 20,'lr_encoder':1e-5,'lr_heads':1e-4,'selection':'macro domain validation clean+blur loss' if args.stage=='pretrain' else 'THINGS public test single Top5','loss':'original diagonal one-way ClipLoss; norm_kind=y; temperature=False; clean+blur equal weights','data':audit['counts'],'sampler':'each trial once per epoch; shuffled native-channel homogeneous batches','preprocessing':'unchanged official caches and existing REVE input adaptation','initialization':'original pretrained REVE; fresh heads' if source is None else 'stage1 best full model; optimizer reset','source_epoch':None if source is None else source['epoch'],'test_role':'public test used for stage2/control selection, scheduling and early stopping; not independent evaluation','stage_training_samples':{d:len(train[d]) for d in domains}}
    write(root/'config.json',cfg)
    if source is not None:write(root/'source.json',{'epoch':source['epoch'],'sha256':hashlib.sha256((ROOT/'pretrain/best.pt').read_bytes()).hexdigest()})
    best=float('inf') if mode=='min' else -1.;stale=0;best_epoch=None
    def save(name,epoch,validation):
        torch.save({'model':model.state_dict(),'epoch':epoch,'config':cfg,'manifest':manifest,'channels':ds.channels,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'validation':validation,'best_epoch':best_epoch},root/name)
    for epoch in range(1,cfg['epochs']+1):
        batcher.epoch=epoch;model.train();totals={};start=time.time()
        for step,b in enumerate(loader):
            d=b['domain'][0];assert len(set(b['domain']))==1
            opt.zero_grad(set_to_none=True);h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),d)
            lc=criterion(q,b['clean'].cuda());lb=criterion(h,a);loss=lc+lb
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            t=totals.setdefault(d,[0.,0.,0]);t[0]+=lc.item()*len(q);t[1]+=lb.item()*len(q);t[2]+=len(q)
            if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'domain':d,'clean':lc.item(),'blur':lb.item()}),flush=True)
        validation=validate(model,ds,{d:val[d] for d in domains},criterion) if args.stage=='pretrain' else {'things':test_once(model)}
        score=float(np.mean([v['clean_loss']+v['blur_loss'] for v in validation.values()])) if mode=='min' else validation['things']['single_trial']['top5']
        sched.step(score)
        if (score<best if mode=='min' else score>best):best=score;best_epoch=epoch;stale=0;save('best.pt',epoch,validation)
        else:stale+=1
        save('last.pt',epoch,validation)
        row={'epoch':epoch,'train':{d:{'clean':v[0]/v[2],'blur':v[1]/v[2],'count':v[2]} for d,v in totals.items()},'validation':validation,'score':score,'best':best,'best_epoch':best_epoch,'stale':stale,'seconds':time.time()-start,'lr':[g['lr'] for g in opt.param_groups]}
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write(root/'status.json',{'state':'training','epoch':epoch,'best_epoch':best_epoch});print(json.dumps(row),flush=True)
        if stale>=cfg['patience']:break
    ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    held_metrics=evaluate_held_subjects(model,ds,audit)
    write(root/'held_subjects.json',{'best_epoch':best_epoch,'metrics':held_metrics,'subjects':audit['held_subjects'],'scope':'unseen subjects and source-held images' if args.stage=='pretrain' else 'unseen subjects, image identities may be seen in THINGS adaptation; not cross-stimulus'})
    if args.stage!='pretrain':
        ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);write(root/'test_final.json',{'best_epoch':best_epoch,'metrics':ck['validation']['things'],'role':cfg['test_role']})
    write(root/'status.json',{'state':'complete','epochs':epoch,'best_epoch':best_epoch})
if __name__=='__main__':main()
