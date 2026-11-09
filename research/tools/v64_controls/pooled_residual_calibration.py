"""Frozen native blur heads, sample-dependent pooled512 residual calibration."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common_target_readout import ROOT, load, write, digest, evaluate, self_test

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,64),nn.GELU(),nn.Linear(64,512))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        return x+self.net(x)

def native(fold,arm,cache,meta):
    root=ROOT/f'fold_{fold:02d}_{arm}'
    cfg=json.loads((root/'config.json').read_text())
    checkpoint=Path(meta['checkpoint'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)['model']
    projection=nn.Linear(512,1536)
    projection.load_state_dict({k.removeprefix('blur_head.'):v for k,v in state.items() if k.startswith('blur_head.')},strict=True)
    image=nn.Sequential(nn.Linear(1536,768),nn.ELU(),nn.Dropout(0),nn.Linear(768,1536))
    image.load_state_dict({k.removeprefix('image_adapter.'):v for k,v in state.items() if k.startswith('image_adapter.')},strict=True)
    projection.eval().requires_grad_(False);image.eval().requires_grad_(False)
    raw=np.load(Path(cfg['source_root'])/'features/blur.npy',mmap_mode='r')[cache['images']].copy()
    gallery=[]
    for k in meta['candidate_ids']:
        with np.load(Path(cfg['test_blur_shards'])/f'{k:05d}.npz') as f:gallery.append(f['blur'].copy())
    with torch.no_grad():
        targets=F.normalize(image(torch.from_numpy(raw)),dim=-1)
        gallery=F.normalize(image(torch.from_numpy(np.stack(gallery))),dim=-1)
    expected=json.loads((root/'result.json').read_text())['zero']
    meta.update(checkpoint_sha256=digest(checkpoint),visual_space='native frozen image_adapter of each checkpoint')
    return projection,targets,gallery,expected

def main(a):
    torch.set_num_threads(4);self_test();a.out.mkdir(parents=True,exist_ok=False)
    cfg=dict(folds=[1,2],arms=['baseline','identity'],budgets=[50,200,1000],seeds=[33,34,35,36,37],steps=200,report_steps=[0,10,25,50,100,200],lr=1e-4,weight_decay=.01,batch_size=50,gradient_clip=1,optimizer='AdamW',loss='blur only, one-direction CE; image-side L2; temperature1',adapter='e + Linear64to512(GELU(Linear512to64(LayerNorm(e)))); final linear zero init',frozen='REVE cached pooled512; original blur_head; original image_adapter; no clean/local loss',support='same nested subsets of fixed1000-image pool as common_target_readout; one fixed trial per image',selection='fixed step200, no test-based checkpoint selection',comparison='full native systems; native galleries differ across baseline and identity',device='cpu')
    write(a.out/'config.json',cfg)
    pairs={};manifest={};preflight={}
    for fold in cfg['folds']:
        pairs[fold]={};manifest[str(fold)]={}
        for arm in cfg['arms']:
            cache,_,_,labels,meta=load(fold,arm)
            projection,targets,gallery,expected=native(fold,arm,cache,meta)
            adapter=Residual();x=cache['etest'].float()
            with torch.no_grad():torch.testing.assert_close(adapter(x[:10]),x[:10],rtol=0,atol=0)
            actual=evaluate(nn.Sequential(adapter,projection),x,labels,gallery)
            for group in ['single_trial','subject_aggregated']:
                assert actual[group]['count']==expected[group]['count']
                for metric in ['top1','top5']:
                    assert abs(actual[group][metric]-expected[group][metric]) < 2e-6,(fold,arm,group,metric,actual,expected)
            before={k:v.clone() for k,v in projection.state_dict().items()}
            loss=F.cross_entropy(projection(adapter(cache['etrain'][:50].float()))@targets[:50].T,torch.arange(50))
            loss.backward()
            assert adapter.net[-1].weight.grad.norm()>0
            assert all(p.grad is None for p in projection.parameters())
            opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4);opt.step();opt.zero_grad()
            F.cross_entropy(projection(adapter(cache['etrain'][:50].float()))@targets[:50].T,torch.arange(50)).backward()
            assert adapter.net[1].weight.grad.norm()>0
            assert all(torch.equal(v,before[k]) for k,v in projection.state_dict().items())
            pairs[fold][arm]=(cache,labels,projection,targets,gallery,actual)
            manifest[str(fold)][arm]=meta;preflight[f's{fold}_{arm}']=actual
        b,i=[pairs[fold][arm][0] for arm in cfg['arms']]
        assert np.array_equal(b['images'],i['images']) and b['records']==i['records']
    write(a.out/'manifest.json',manifest);write(a.out/'preflight.json',preflight)
    print('PREFLIGHT PASS: zero-step native metrics reproduced; residual identity exact; residual gradients nonzero; original projection frozen; matched support',flush=True)
    done=0;start=time.time()
    for fold,pair in pairs.items():
        for budget in cfg['budgets']:
            for seed in cfg['seeds']:
                subset=np.random.default_rng(seed).permutation(1000)[:budget]
                for arm,(cache,labels,projection,targets,gallery,zero) in pair.items():
                    key=f's{fold}_{arm}_n{budget}_seed{seed}'
                    write(a.out/'status.json',dict(state='running',current=key,completed=done,total=60))
                    torch.manual_seed(seed);adapter=Residual();model=nn.Sequential(adapter,projection)
                    opt=torch.optim.AdamW(adapter.parameters(),lr=cfg['lr'],weight_decay=.01)
                    generator=torch.Generator().manual_seed(seed+10000)
                    x=cache['etrain'][subset].float();y=targets[subset];scores={'0':zero};losses=[]
                    for step in range(1,201):
                        adapter.train();ix=torch.randperm(budget,generator=generator)[:50]
                        loss=F.cross_entropy(model(x[ix])@y[ix].T,torch.arange(len(ix)))
                        opt.zero_grad();loss.backward()
                        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
                        nn.utils.clip_grad_norm_(adapter.parameters(),1);opt.step();losses.append(float(loss.detach()))
                        if step in cfg['report_steps']:scores[str(step)]=evaluate(model,cache['etest'].float(),labels,gallery)
                    write(a.out/f'{key}.json',dict(scores=scores,losses=losses,subset_indices=subset.tolist(),image_ids=cache['images'][subset].tolist(),parameter_count=sum(p.numel() for p in adapter.parameters())))
                    torch.save(dict(adapter=adapter.state_dict(),checkpoint=manifest[str(fold)][arm]['checkpoint']),a.out/f'{key}.pt')
                    done+=1;print(key,scores['200'],'completed',done,'elapsed_s',round(time.time()-start),flush=True)
    write(a.out/'status.json',dict(state='complete',completed=done,elapsed_seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except Exception as exc:
        if a.out.exists():write(a.out/'failure.json',dict(error=repr(exc)))
        raise
