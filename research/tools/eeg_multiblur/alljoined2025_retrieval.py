"""Standalone Alljoined2025 retrieval with B head and original contrastive loss."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from neuraltrain.losses.losses import ClipLoss
from transfer_protocol import Model,DomainBatches,partitions,write,BASE
from alljoined_train import MixedData
from recovery_ablation import metrics
ROOT=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v35_alljoined2025_b')
DOMAIN='Xu2025Alljoined'

@torch.no_grad()
def evaluate(model,ds,indices,subjects,criterion):
    model.eval();pred=[];ids=[];total=np.zeros(2);count=0
    for b in DataLoader(Subset(ds,indices.tolist()),batch_size=256,num_workers=2):
        h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),DOMAIN)
        total+=np.array([criterion(q,b['clean'].cuda()).item(),criterion(h,a).item()])*len(q)
        count+=len(q);pred.append(q.cpu().numpy());ids.append(b['id'].numpy())
    result=metrics(np.concatenate(pred),np.concatenate(ids),subjects[indices],ds.clean)
    result['clean_loss'],result['blur_loss']=(total/count).tolist()
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    ROOT.mkdir(exist_ok=True);ds=MixedData();ids,tr,va,audit=partitions(ds);train=tr[DOMAIN];valid=va[DOMAIN]
    records=json.loads((ROOT.parent/'reve_v18_alljoined/eeg_complete.json').read_text())['recordings']
    subjects=np.full(len(ds),'things',dtype='U80');start=sum(len(r['ids']) for r in ds.records[:2])
    for r in records:
        end=start+len(r['events']);subjects[start:end]=str(r['subject']);start=end
    assert start==len(ds)
    assert len(np.unique(subjects[train]))==20
    assert np.array_equal(np.unique(subjects[train]),np.unique(subjects[valid]))
    assert not np.intersect1d(ids[train],ids[valid]).size
    # Fixed training probe, matched candidate count to validation, no optimization on validation.
    rng=np.random.default_rng(33);probe_images=rng.choice(np.unique(ids[train]),len(np.unique(ids[valid])),replace=False)
    probe=train[np.isin(ids[train],probe_images)]
    split={'train_trials':len(train),'val_trials':len(valid),'train_images':len(np.unique(ids[train])),'val_images':len(np.unique(ids[valid])),'subjects':np.unique(subjects[train]).tolist(),'validation_image_ids':np.unique(ids[valid]).tolist(),'training_probe_image_ids':sorted(probe_images.tolist()),'training_probe_trials':len(probe),'image_overlap':0}
    write(ROOT/'split.json',split)
    torch.manual_seed(33);np.random.seed(33);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((BASE/'data/manifest.json').read_text());model=Model(manifest,ds.channels).cuda()
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},{'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')],'lr':1e-4}],weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5,min_lr=1e-6)
    criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False).cuda()
    batches=DomainBatches({DOMAIN:train},256,33)
    loader=DataLoader(ds,batch_sampler=batches,num_workers=4,pin_memory=True,persistent_workers=True)
    cfg={'dataset':DOMAIN,'epochs':200,'patience':20,'batch':256,'seed':33,'encoder_lr':1e-5,'head_lr':1e-4,'weight_decay':.05,'adapter_dropout':.85,'recovery':'B ProjectedHead, no outer skip','loss':'original one-way ClipLoss norm_kind=y temperature=False; clean+blur equal','selection':'full validation single-trial Top5; same metric for scheduler/earlystop/best checkpoint','scheduler':'plateau patience5 factor0.5 min_lr1e-6; no warmup','initialization':'original pretrained REVE, fresh heads; all parameters trained','data':'official caches unchanged; only Alljoined2025 train-image trials in optimizer; no THINGS/2024 batches; existing stim_train eligibility preserved','scope':'20 known subjects, held-out images; validation used for model selection, no independent final test','probe':'fixed training images; same candidate count as validation, eval mode every epoch'}
    write(ROOT/'config.json',cfg)
    if args.preflight:
        report=[];model.train()
        for step,b in enumerate(loader):
            assert set(b['domain'])=={DOMAIN}
            opt.zero_grad();h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),DOMAIN)
            loss=criterion(q,b['clean'].cuda())+criterion(h,a);loss.backward();grads={}
            for name,module in [('encoder',model.encoder),('projection',model.blur_head),('adapter',model.image_adapter),('recovery',model.recovery)]:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
                grads[name]=sum(p.grad.abs().sum().item() for p in module.parameters());assert grads[name]>0
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            report.append({'loss':loss.item(),'gradient_l1':grads})
            if step==2:break
        write(ROOT/'preflight.json',{'passed':True,'steps':report,'updates_discarded':True});print('PREFLIGHT PASSED',flush=True);return
    assert not (ROOT/'history.jsonl').exists();best=-1.;stale=0;best_epoch=0
    for epoch in range(1,201):
        batches.epoch=epoch;model.train();total=np.zeros(2);count=0;start=time.time()
        for step,b in enumerate(loader):
            assert set(b['domain'])=={DOMAIN}
            opt.zero_grad(set_to_none=True);h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),DOMAIN)
            lc=criterion(q,b['clean'].cuda());lb=criterion(h,a);loss=lc+lb
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=np.array([lc.item(),lb.item()])*len(q);count+=len(q)
            if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'clean':lc.item(),'blur':lb.item()}),flush=True)
        val=evaluate(model,ds,valid,subjects,criterion);probe_result=evaluate(model,ds,probe,subjects,criterion)
        score=val['single_trial']['top5'];scheduler.step(score)
        improved=score>best
        if improved:best=score;best_epoch=epoch;stale=0
        else:stale+=1
        ck={'model':model.state_dict(),'epoch':epoch,'manifest':manifest,'config':cfg,'channels':ds.channels,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'validation':val,'training_probe':probe_result,'best_epoch':best_epoch}
        torch.save(ck,ROOT/'last.pt')
        if improved:torch.save(ck,ROOT/'best.pt')
        row={'epoch':epoch,'train_loss':(total/count).tolist(),'validation':val,'training_probe':probe_result,'best_epoch':best_epoch,'stale':stale,'lr':[g['lr'] for g in opt.param_groups],'seconds':time.time()-start}
        with (ROOT/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write(ROOT/'status.json',{'state':'training','epoch':epoch,'best_epoch':best_epoch});print(json.dumps({k:v for k,v in row.items() if k not in ('validation','training_probe')}),flush=True)
        if stale>=20:break
    write(ROOT/'status.json',{'state':'complete','epochs':epoch,'best_epoch':best_epoch})
if __name__=='__main__':main()
