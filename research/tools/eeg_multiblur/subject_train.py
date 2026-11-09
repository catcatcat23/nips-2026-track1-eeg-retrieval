"""Independent subject models, two single-GPU queues; no pooled initialization."""
import argparse,json,math,random,time,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from model import Model
from data import DatasetEEG
from losses import objective
from evaluate import normalize,group_means,retrieval_metrics
from evaluate_parallel import gpu_hits


def split_subject(root,subject,seed):
    train=DatasetEEG(root,'train',local_aux=False)
    val=DatasetEEG(root,'val',aux=False)
    test=DatasetEEG(root,'test',aux=False)
    ti=np.flatnonzero(train.subjects==subject);vi=np.flatnonzero(val.subjects==subject)
    assert len(ti)>0
    if len(vi):
        policy='existing official split; full official validation gallery'
        gallery_ids=np.unique(val.ids)
    else:
        rng=np.random.default_rng(seed)
        ids=np.unique(train.ids[ti]);rng.shuffle(ids)
        held=np.sort(ids[:max(1,round(.2*len(ids)))])
        mask=np.isin(train.ids[ti],held);vi=ti[mask];ti=ti[~mask];val=train
        assert not np.intersect1d(train.ids[ti],val.ids[vi]).size
        gallery_ids=held
        policy='20% training image IDs held out, all repeats grouped; held-out-only gallery'
    test_i=np.flatnonzero(test.subjects==subject)
    assert all(len(x)>0 for x in (ti,vi,test_i))
    assert not np.intersect1d(np.r_[train.ids[ti],val.ids[vi]],test.ids[test_i]).size
    for ds,idx in [(train,ti),(val,vi),(test,test_i)]:assert np.all(ds.subjects[idx]==subject)
    return train,ti,val,vi,test,test_i,gallery_ids,policy


class Evaluator:
    def __init__(self,ds,indices,gallery_ids,subject):
        self.ds=ds;self.indices=indices;self.subject=subject
        candidates,inverse=np.unique(np.asarray(ds.clean[gallery_ids]),axis=0,return_inverse=True)
        mapping=dict(zip(gallery_ids.tolist(),inverse.reshape(-1).tolist()))
        self.labels=np.array([mapping[int(i)] for i in ds.ids[indices]])
        self.candidates=candidates;self.gallery=normalize(candidates)
        self.loader=DataLoader(Subset(ds,indices),batch_size=256,num_workers=2,pin_memory=True)
    @torch.no_grad()
    def run(self,model,reference=False):
        model.eval();pred=[]
        for b in self.loader:pred.append(model(b['eeg'].cuda()).float().cpu().numpy())
        pred=np.concatenate(pred)
        keys,means=group_means(pred,self.labels)
        result={}
        for name,x,labels in [('single_trial',pred,self.labels),('subject_aggregated',means,keys[:,0])]:
            hits=gpu_hits(x,self.gallery,labels,'cuda')
            result[name]={'top1':float(hits[:,0].mean()),'top5':float(hits[:,1].mean()),'count':len(x)}
        result['candidate_images']=len(self.gallery)
        if reference:
            from evaluate import topk_accuracy
            # Use identical full candidate gallery, even if this subject lacks some images.
            for name,x,labels in [('single_trial',pred,self.labels),('subject_aggregated',means,keys[:,0])]:
                totals=np.zeros(2)
                for start in range(0,len(x),256):
                    scores=normalize(x[start:start+256])@self.gallery.T
                    for j,k in enumerate((1,5)):totals[j]+=topk_accuracy(scores,labels[start:start+256],k)*len(scores)
                assert np.allclose(totals/len(x),[result[name]['top1'],result[name]['top5']],rtol=0,atol=1e-12)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    cfg=json.loads(Path(__file__).with_name('subject_config.json').read_text())
    root=Path(cfg['data_root']);out=Path(cfg['run_root'])/('subject_'+args.subject);out.mkdir(exist_ok=True,parents=True)
    subject='Gifford2022Large/'+args.subject
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed'])
    train,ti,val,vi,test,tei,gallery_ids,policy=split_subject(root,subject,cfg['seed'])
    manifest=json.loads((root/'data/manifest.json').read_text())
    np.savez(out/'split_indices.npz',train=ti,val=vi,test=tei,val_gallery_ids=gallery_ids)
    audit={'subject':subject,'train':len(ti),'val':len(vi),'test':len(tei),'val_policy':policy,'val_source':'train' if val is train else 'val','val_candidate_ids':len(gallery_ids),'seed':cfg['seed'],'initialization':'fresh per-subject model; no pooled weights'}
    (out/'split_audit.json').write_text(json.dumps(audit,indent=2));(out/'config.json').write_text(json.dumps(cfg,indent=2))
    model=Model(*manifest['input_shape'],dropout=cfg['dropout'],use_local=False).cuda()
    loader=DataLoader(Subset(train,ti),batch_size=cfg['batch_size'],shuffle=True,drop_last=True,num_workers=2,pin_memory=True,persistent_workers=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    warm=cfg['warmup_epochs']*len(loader);total=cfg['epochs']*len(loader)
    def factor(step):
        if step<warm:return (step+1)/max(1,warm)
        t=(step-warm)/max(1,total-warm);minimum=cfg['min_lr']/cfg['lr']
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*t))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    val_eval=Evaluator(val,vi,gallery_ids,subject)
    best=-1.;bad=0;started=time.time()
    (out/'status.json').write_text(json.dumps({'state':'preflight' if args.preflight else 'training','subject':subject}))
    for epoch in range(cfg['epochs']):
        model.train();sums=np.zeros(4);tick=time.time()
        for step,b in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            eeg=b['eeg'].cuda();blur=b['blur'].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(eeg,blur)
            loss,parts=objective(outputs,b['clean'].cuda(),blur,b['id'].cuda(),None,cfg,distributed=False)
            assert torch.isfinite(loss);loss.backward()
            if args.preflight:
                for name,param in model.named_parameters():
                    if param.requires_grad:assert param.grad is not None and torch.isfinite(param.grad).all(),name
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step()
            sums+=np.array([loss.item(),parts['clean'],parts['blur'],parts['restore']])
            if step==0:print(json.dumps({'subject':subject,'epoch':epoch+1,'step':step,'loss':loss.item()}),flush=True)
            if args.preflight and step==4:
                metrics=val_eval.run(model,reference=True)
                (out/'preflight.json').write_text(json.dumps({'passed':True,'steps':5,'metrics':metrics,'allocated_gib':torch.cuda.max_memory_allocated()/2**30},indent=2))
                print('PREFLIGHT PASSED',metrics,flush=True);return
        metrics=val_eval.run(model);score=metrics[cfg['selection_group']][cfg['selection_metric']]
        row={'epoch':epoch+1,'losses':dict(zip(['total','clean','blur','restore'],(sums/len(loader)).tolist())),'validation':metrics,'epoch_seconds':time.time()-tick}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('VALIDATION',json.dumps(row),flush=True)
        if score>best:
            best=score;bad=0
            torch.save({'model':model.state_dict(),'epoch':epoch+1,'config':cfg,'manifest':manifest,'subject':subject,'validation':metrics},out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
        elif epoch>=cfg['warmup_epochs']:bad+=1
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'bad_epochs':bad,'subject':subject,'config':cfg,'manifest':manifest},out/'last.tmp');os.replace(out/'last.tmp',out/'last.pt')
        (out/'status.json').write_text(json.dumps({'state':'training','epoch':epoch+1,'bad_epochs':bad,'best':best}))
        if bad>=cfg['patience']:break
    c=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(c['model'])
    test_result=Evaluator(test,tei,np.unique(test.ids),subject).run(model)
    (out/'test_metrics.json').write_text(json.dumps(test_result,indent=2))
    (out/'status.json').write_text(json.dumps({'state':'test_complete','best_epoch':c['epoch'],'seconds':time.time()-started}))
    print('TEST COMPLETE',subject,test_result,flush=True)

if __name__=='__main__':main()
