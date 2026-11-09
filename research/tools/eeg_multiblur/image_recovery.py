"""Image-only cached raw multiblur -> official clean recovery diagnostic."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from recovery_ablation import ProjectedHead, objective
from evaluate import normalize
from evaluate_parallel import gpu_hits


def save(path, data):
    path.write_text(json.dumps(data, indent=2))


@torch.no_grad()
def score(q, c):
    pred, target = q.cpu().numpy(), c.cpu().numpy()
    gallery, labels = np.unique(target, axis=0, return_inverse=True)
    hits = gpu_hits(pred, normalize(gallery), labels.reshape(-1), torch.device('cuda'))
    return {'top1': float(hits[:, 0].mean()), 'top5': float(hits[:, 1].mean()),
            'cosine': float((F.normalize(q,dim=-1)*F.normalize(c,dim=-1)).sum(-1).mean()),
            'count':len(q), 'candidate_images': len(gallery)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('image_recovery_config.json')))
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cosine_only = cfg.get('objective') == 'cosine'
    def loss_fn(q, c, ids, arm):
        if not cosine_only:
            return objective(q, c, ids, arm)
        distance = (1 - (F.normalize(q.float(), dim=-1) * F.normalize(c.float(), dim=-1)).sum(-1)).mean()
        return distance, {'direction': distance.detach()}
    root, src = Path(cfg['run_root']), Path(cfg['source_root'])
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/'history.jsonl').exists(), 'Refuse overwriting a run'
    torch.manual_seed(33)
    np.random.seed(33)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((src/'data/manifest.json').read_text())
    metadata = json.loads((src/'features/complete.json').read_text())
    assert metadata['schema'] == 'upstream_per_scale_l2_dino_v1'
    ids = np.unique(np.concatenate([np.load(src/'data'/f'{s}_ids.npy') for s in ['train','val']]))
    testids = np.unique(np.load(src/'data/test_ids.npy'))
    assert len(ids)==16540 and len(testids)==200 and not len(np.intersect1d(ids,testids))
    clean = np.load(src/'data/clean.npy')
    blur = np.load(src/'features/blur.npy')[ids]
    assert np.load(src/'features/available.npy')[ids].all()
    testblur=[]
    for i in testids:
        with np.load(Path(cfg['test_shards'])/f'{i:05d}.npz') as f:
            assert int(f['id']) == i and str(f['filepath']) == manifest['paths'][i]
            assert str(f['schema']) == metadata['schema']
            np.testing.assert_allclose(f['blur'],f['levels'].mean(0),rtol=1e-5,atol=1e-7)
            testblur.append(f['blur'].copy())
    arrays=[blur,clean[ids],np.stack(testblur),clean[testids]]
    assert all(a.shape[1]==1536 and np.isfinite(a).all() for a in arrays)
    save(root/'data_audit.json', {'train_images':len(ids),'selection_images':len(testids),'disjoint_ids':True,
         'inputs':'raw multiscale mean, no adapter, no EEG','feature_metadata':metadata,
         'array_sha256':[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]})
    b,c,tb,tc=[torch.from_numpy(a.copy()).cuda() for a in arrays]
    labels=torch.as_tensor(ids,device='cuda')
    save(root/'raw_blur_baseline.json',score(tb,tc))
    # Real paired eight-image smoke test, distinct from the freshly reset full run.
    probe=ProjectedHead().cuda().eval()
    opt=torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=0)
    initial=float(loss_fn(probe(b[:8]),c[:8],labels[:8],'D')[0].detach())
    for step in range(300):
        opt.zero_grad(set_to_none=True)
        loss,_=loss_fn(probe(b[:8]),c[:8],labels[:8],'D')
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
        torch.nn.utils.clip_grad_norm_(probe.parameters(),1.0,error_if_nonfinite=True)
        opt.step()
        tiny=score(probe(b[:8]),c[:8])
        if tiny['top1']==1 and float(loss.detach())<initial*.3 and (not cosine_only or tiny['cosine']>0.95):break
    assert tiny['top1']==1 and float(loss.detach())<initial*.3 and (not cosine_only or tiny['cosine']>0.95)
    probe.train();opt.zero_grad(set_to_none=True)
    loss,_=loss_fn(probe(b[:256]),c[:256],labels[:256],'D');loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
    save(root/'preflight.json',{'initial_loss':initial,'tiny_top1':tiny['top1'],'tiny_cosine':tiny['cosine'],'final_loss':float(loss.detach()),'steps':step+1,'batch256_gradients':'finite'})
    del probe,opt
    torch.manual_seed(33)
    head=ProjectedHead().cuda()
    opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=.01)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5,min_lr=1e-6,threshold=1e-5,threshold_mode='abs')
    rng=torch.Generator(device='cuda').manual_seed(33)
    best=-1.;stale=0
    for epoch in range(1,cfg['epochs']+1):
        head.train();start=time.time();totals={};n=0
        for idx in torch.randperm(len(b),device='cuda',generator=rng).split(256):
            opt.zero_grad(set_to_none=True)
            loss,parts=loss_fn(head(b[idx]),c[idx],labels[idx],'D')
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            for k,v in {'total':loss.detach(),**parts}.items():totals[k]=totals.get(k,0)+float(v)*len(idx)
            n+=len(idx)
        head.eval()
        with torch.no_grad():
            selection=score(head(tb),tc)
            trainprobe=score(head(b[:1024]),c[:1024])
        # Top5 primary, Top1 secondary, cosine final tie-break once retrieval saturates.
        key=selection['cosine'] if cosine_only else (selection['top5'],selection['top1'],selection['cosine'])
        improved=best == -1. or (key>best+1e-5 if cosine_only else key>best)
        if improved:
            best=key;stale=0
            torch.save({'model':head.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'config':cfg,'selection':selection},root/'best.pt')
        else:stale+=1
        sched.step(selection['cosine'] if cosine_only else selection['top5'])
        row={'epoch':epoch,'loss':{k:v/n for k,v in totals.items()},'selection':selection,'train_probe_1024':trainprobe,'lr':opt.param_groups[0]['lr'],'stale':stale,'seconds':time.time()-start}
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        save(root/'status.json',{'state':'training','epoch':epoch,'best_key':best})
        if stale>=cfg['patience']:break
    ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    save(root/'result.json',{'best_epoch':ck['epoch'],'selection':ck['selection'],'role':'image-only public test model selection; not EEG accuracy or independent test'})
    save(root/'status.json',{'state':'complete','epochs':epoch,'best_epoch':ck['epoch']})

if __name__=='__main__':main()
