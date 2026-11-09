"""Read-only checkpoint diagnosis on identical public test trials."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import DatasetEEG
from reference_model import ReferenceModel
from recovery_ablation import ProjectedHead,metrics
from evaluate import normalize,group_means
from evaluate_parallel import gpu_hits

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')

def summary(x):
    x=np.asarray(x);return {'count':len(x),'mean':float(x.mean()) if len(x) else None,'p05_median_p95':np.quantile(x,[.05,.5,.95]).tolist() if len(x) else []}

def analyze(pred,ds):
    unique,labels=np.unique(ds.ids,return_inverse=True);gallery=normalize(np.asarray(ds.clean[unique]));subjects,sid=np.unique(ds.subjects,return_inverse=True)
    norms=np.linalg.norm(pred,axis=1);hits=gpu_hits(pred,gallery,labels,torch.device('cuda'))
    margin=[];margin5=[]
    for start in range(0,len(pred),2048):
        scores=normalize(pred[start:start+2048])@gallery.T;idx=labels[start:start+len(scores)];true=scores[np.arange(len(scores)),idx].copy();scores[np.arange(len(scores)),idx]=-np.inf
        margin.extend(true-scores.max(1));margin5.extend(true-np.partition(scores,-5,axis=1)[:,-5])
    keys,means=group_means(pred,np.stack([sid,labels],1));groups=[np.flatnonzero((sid==s)&(labels==i)) for s,i in keys]
    curve={}
    # Five deterministic nested subsets; each draw uses different genuine trials.
    for k in [1,2,4,8,16]:
        values=[]
        for repeat in range(5):
            rng=np.random.default_rng(3300+repeat)
            x=np.stack([np.mean(pred[rng.permutation(g)[:k]],axis=0,dtype=np.float64) for g in groups])
            h=gpu_hits(x,gallery,keys[:,1],torch.device('cuda'));values.append(h.mean(0))
        a=np.asarray(values);curve[str(k)]={'top1':float(a[:,0].mean()),'top5':float(a[:,1].mean()),'draw_min':a.min(0).tolist(),'draw_max':a.max(0).tolist(),'draws':5}
    ha=gpu_hits(means,gallery,keys[:,1],torch.device('cuda'));curve['all']={'top1':float(ha[:,0].mean()),'top5':float(ha[:,1].mean())}
    _,unitmeans=group_means(normalize(pred),np.stack([sid,labels],1));hu=gpu_hits(unitmeans,gallery,keys[:,1],torch.device('cuda'))
    score=normalize(means)@gallery.T;errors=[]
    for j in np.flatnonzero(~ha[:,0]):
        errors.append({'subject':str(subjects[keys[j,0]]),'target_image_id':int(unique[keys[j,1]]),'top_wrong_image_id':int(unique[score[j].argmax()]),'margin':float(score[j,keys[j,1]]-score[j].max()),'top5_correct':bool(ha[j,1])})
    return {'official':metrics(pred,ds.ids,ds.subjects,np.asarray(ds.clean)),'prediction_norm':summary(norms),'norm_correct_top1':summary(norms[hits[:,0]]),'norm_wrong_top1':summary(norms[~hits[:,0]]),'cosine_margin_top1':summary(margin),'cosine_margin_top5':summary(margin5),'aggregation_curve':curve,'repeats_per_group':summary([len(g) for g in groups]),'unit_normalized_before_mean_diagnostic_only':{'top1':float(hu[:,0].mean()),'top5':float(hu[:,1].mean())},'aggregate_top1_errors':errors}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ds=DatasetEEG(BASE/'reve_v13_reference','test',aux=False)
    for name in ['v14','B']:
        path=BASE/('reve_v14_trainval_public_selection/training/best.pt' if name=='v14' else 'recovery_v15_frozen_v13/B/best.pt')
        ck=torch.load(path,map_location='cpu',weights_only=False);dest=out/(name+'_pred.npy');digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if name=='v14':model=ReferenceModel(ck['manifest']).cuda().eval();model.load_state_dict(ck['model'])
        else:model=ProjectedHead().cuda().eval();model.load_state_dict(ck['head'])
        with torch.inference_mode():
            if name=='v14':
                parts=[]
                for step,b in enumerate(DataLoader(ds,batch_size=256,num_workers=2)):
                    parts.append(model(b['eeg'].cuda()).cpu().numpy())
                    if step%100==0:print(name,step,flush=True)
                pred=np.concatenate(parts)
            else:
                cache=BASE/'recovery_v15_frozen_v13/cache'
                assert np.array_equal(np.load(cache/'test_ids.npy'),ds.ids)
                assert np.array_equal(np.load(cache/'test_subjects.npy'),ds.subjects)
                h=np.load(cache/'test_h.npy',mmap_mode='r');pred=np.concatenate([model(torch.from_numpy(h[i:i+1024].copy()).cuda()).cpu().numpy() for i in range(0,len(h),1024)])
        assert np.isfinite(pred).all();np.save(dest,pred)
        result=analyze(pred,ds)
        for group in ['single_trial','subject_aggregated']:
            for key in ['top1','top5']:assert abs(result['official'][group][key]-ck['selection'][group][key])<1e-5,(name,group,key)
        result.update(checkpoint=str(path),sha256=digest,epoch=ck['epoch'],note='Public test descriptive diagnostics; five draws are not independent significance tests')
        (out/(name+'_report.json')).write_text(json.dumps(result,indent=2));print('COMPLETE',name,json.dumps(result['aggregation_curve']),flush=True)
        del model,pred,ck;torch.cuda.empty_cache()
    (out/'status.txt').write_text('COMPLETED\n')
if __name__=='__main__':main()
