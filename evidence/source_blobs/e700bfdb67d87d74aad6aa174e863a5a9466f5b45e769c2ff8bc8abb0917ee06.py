"""Unpadded distributed inference and GPU retrieval with official tie semantics."""
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from data import DatasetEEG
from evaluate import normalize, group_means


def gpu_hits(pred, gallery, labels, device, chunk=256):
    # Preserve NumPy normalization and dtype (FP64 for subject means).
    # Disable TF32 to avoid reduced-precision ranking near the cutoff.
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.as_tensor(gallery,device=device,dtype=torch.from_numpy(pred).dtype)
    hits = np.empty((len(pred),2),dtype=bool)
    for start in range(0,len(pred),chunk):
        x=normalize(pred[start:start+chunk])
        s=torch.as_tensor(x,device=device) @ g.T
        idx=torch.as_tensor(labels[start:start+len(x)],device=device,dtype=torch.long)
        true=s.gather(1,idx[:,None])
        rank=((s>true).sum(1)+(s>=true).sum(1)-1)/2
        rank=torch.where(torch.isnan(true[:,0]),torch.inf,rank)
        h=torch.stack((rank<1,rank<5),1).cpu().numpy()
        # Resolve close-score rows on the original CPU path, preserving ties.
        tol=2e-6 if s.dtype==torch.float32 else 1e-12
        lower=(s>true+tol).sum(1)
        upper=(s>=true-tol).sum(1)-1
        uncertain=(((lower<1)&(upper>=1))|((lower<5)&(upper>=5))).cpu().numpy()
        if uncertain.any():
            cpu=x[uncertain] @ gallery.T
            labs=labels[start:start+len(x)][uncertain]
            t=np.take_along_axis(cpu,labs[:,None],1)
            r=((cpu>t).sum(1)+(cpu>=t).sum(1)-1)/2
            r[np.isnan(t[:,0])]=np.inf
            h[uncertain]=np.stack((r<1,r<5),1)
        hits[start:start+len(x)]=h
    return hits


@torch.no_grad()
def evaluate_parallel(model,root,split,device,check_reference=False,max_per_subject=None,seed=21):
    tick=time.time()
    rank=dist.get_rank();world=dist.get_world_size()
    model.eval()
    # Keep evaluation buffers identical when bypassing DDP.forward.
    for b in model.buffers():dist.broadcast(b,0)
    ds=DatasetEEG(root,split,aux=False)
    selected=np.arange(len(ds))
    if max_per_subject is not None:
        rng=np.random.default_rng(seed)
        selected=np.sort(np.concatenate([rng.choice(np.flatnonzero(ds.subjects==s),size=min(max_per_subject,int((ds.subjects==s).sum())),replace=False) for s in np.unique(ds.subjects)]))
    ids=ds.ids[selected];subject_ids=ds.subjects[selected];n=len(selected)
    indices=selected[rank::world].tolist()
    chunks=[]
    for b in DataLoader(Subset(ds,indices),batch_size=256,num_workers=2):
        chunks.append(model(b['eeg'].to(device)).float())
    local=torch.cat(chunks) if chunks else torch.empty(0,1536,device=device)
    padded=torch.zeros((n+world-1)//world,1536,device=device)
    padded[:len(local)]=local
    gathered=[torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered,padded)
    pred=np.empty((n,1536),dtype=np.float32)
    for r,p in enumerate(gathered):pred[r::world]=p[:len(pred[r::world])].cpu().numpy()
    del gathered,padded,local,chunks
    # Exact target deduplication on unique IDs instead of sorting repeated trials.
    unique_ids,trial_idx=np.unique(ids,return_inverse=True)
    candidates,image_idx=np.unique(np.asarray(ds.clean[unique_ids]),axis=0,return_inverse=True)
    labels=image_idx.reshape(-1)[trial_idx.reshape(-1)]
    gallery=normalize(candidates)
    _,subjects=np.unique(subject_ids,return_inverse=True)
    keys,means=group_means(pred,np.stack((subjects.reshape(-1),labels),1))
    result={'per_subject':{str(s):{} for s in np.unique(subject_ids)}}
    for name,x,y in [('single_trial',pred,labels),('subject_aggregated',means,keys[:,1])]:
        local_hits=gpu_hits(x[rank::world],gallery,y[rank::world],device)
        counts=torch.as_tensor(local_hits.sum(0),device=device,dtype=torch.int64)
        dist.all_reduce(counts)
        result[name]={'top1':counts[0].item()/len(x),'top5':counts[1].item()/len(x),'count':len(x)}
        group_subjects=subjects if name=='single_trial' else keys[:,0]
        for sid,label in enumerate(np.unique(subject_ids)):
            mask=group_subjects[rank::world]==sid
            totals=torch.as_tensor([*local_hits[mask].sum(0),int(mask.sum())],device=device,dtype=torch.int64)
            dist.all_reduce(totals)
            result['per_subject'][str(label)][name]={'top1':totals[0].item()/totals[2].item(),'top5':totals[1].item()/totals[2].item(),'count':totals[2].item()}
    result['candidate_images']=len(candidates)
    if max_per_subject is not None:
        result['sampling']={'seed':seed,'max_per_subject':max_per_subject,'indices_file':'train_diagnostic_indices.npy','gallery':'unique clean targets in fixed subset'}
        if rank==0:np.save(root/'training/train_diagnostic_indices.npy',selected)
    if rank==0:print('PARALLEL EVALUATION',split,'seconds',time.time()-tick,flush=True)
    if check_reference and rank==0:
        from evaluate import retrieval_metrics
        reference=retrieval_metrics(pred,np.asarray(ds.clean[ids]),subject_ids)
        assert result['candidate_images']==reference['candidate_images']
        for group in ('single_trial','subject_aggregated'):
            assert result[group]['count']==reference[group]['count']
            for metric in ('top1','top5'):
                assert abs(result[group][metric]-reference[group][metric])<1e-12,(result,reference)
        print('FULL VALIDATION CPU/GPU METRICS MATCH',result,flush=True)
    return result
