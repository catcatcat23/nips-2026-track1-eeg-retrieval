"""Matched supervised target calibration with training-image-only labels."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data import DatasetEEG
import extensions as ex
from local_ot_train import metrics
class Adapter(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,64),nn.GELU(),nn.Linear(64,1536));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,h):return h+self.net(h)
def main(config,out):
 out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=json.loads(config.read_text());ex.CFG.update(cfg);fold=Path(cfg['run_root']);res=json.loads((fold/'stage1/result.json').read_text());variant=max(res['best'],key=res['best'].get);epoch=res['best_epoch'][variant]
 cp=fold/'stage1'/(f'epoch_{epoch:03d}.pt' if variant=='raw' else f'{variant}_epoch_{epoch:03d}.pt');ck=torch.load(cp,map_location='cpu',weights_only=False)
 cls=ex.IdentityModel if cfg.get('identity_weight') else ex.AugmentedModel;model=cls(ck['manifest']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
 train=[DatasetEEG(cfg['source_root'],s,aux=False) for s in ['train','val']];test=DatasetEEG(cfg['source_root'],'test',aux=False)
 groups={}
 for d,ds in enumerate(train):
  for i in np.flatnonzero(ds.subjects.astype(str)==cfg['heldout_subject']):groups.setdefault(int(ds.ids[i]),[]).append((d,int(i)))
 rng=np.random.default_rng(910033);images=rng.choice(sorted(groups),1000,replace=False);records=[groups[int(im)][int(rng.integers(len(groups[int(im)])))] for im in images]
 assert not np.intersect1d(images,test.ids).size
 testidx=np.flatnonzero(test.subjects.astype(str)==cfg['heldout_subject']);testids=test.ids[testidx];subjects=test.subjects[testidx];assert len(np.unique(testids))==200
 protocol=dict(checkpoint=str(cp),heldout=cfg['heldout_subject'],budgets=[0,50,200,1000],steps=200,lr=1e-4,batch_size=50,seed=33,adapter='h + Linear64(GELU(Linear1536(LayerNorm(h)))) zero output init',loss='clean CE + blur CE, image-side L2 only, temperature1, original heads frozen',records=records,images=images.tolist(),split='one single trial per calibration training image; original 200 test images entirely disjoint; no test-based stopping')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 @torch.inference_mode()
 def encode(xs):return torch.cat([model.encode_eeg(x.cuda())[0].cpu() for x in xs])
 htrain=encode([torch.stack([torch.from_numpy(train[d].x[i].copy()) for d,i in records[j:j+64]]) for j in range(0,1000,64)]).cuda()
 htest=encode([torch.tensor(test.x[testidx[j:j+64]].copy()) for j in range(0,len(testidx),64)]).cuda()
 clean=torch.tensor(train[0].clean[images].copy(),device='cuda');blurcache=np.load(Path(cfg['source_root'])/'features/blur.npy',mmap_mode='r')
 with torch.no_grad():blur=F.normalize(model.image_adapter(torch.tensor(blurcache[images].copy(),device='cuda')),dim=-1)
 galleryids=np.unique(testids);testblur=[]
 for i in galleryids:
  with np.load(Path(cfg['test_blur_shards'])/f'{i:05d}.npz') as f:testblur.append(f['blur'].copy())
 with torch.no_grad():gallery=F.normalize(model.image_adapter(torch.tensor(np.stack(testblur),device='cuda')),dim=-1).cpu().numpy()
 allblur=np.zeros_like(test.clean);allblur[galleryids]=gallery;results={}
 for budget in [0,50,200,1000]:
  torch.manual_seed(33);np.random.seed(33);random.seed(33);adapter=Adapter().cuda();opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01);losses=[]
  if budget:
   adapter.train()
   for step in range(200):
    ix=torch.randperm(budget,device='cuda')[:50];h=adapter(htrain[ix]);z=model.recovery(h);target=torch.arange(len(ix),device='cuda')
    loss=F.cross_entropy(z@F.normalize(clean[ix],dim=-1).T,target)+F.cross_entropy(h@blur[ix].T,target)
    opt.zero_grad();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters());nn.utils.clip_grad_norm_(adapter.parameters(),1);opt.step();losses.append(float(loss.detach()))
  adapter.eval();hh=[];zz=[]
  with torch.inference_mode():
   for x in htest.split(128):
    h=adapter(x) if budget else x;hh.append(h.cpu().numpy());zz.append(model.recovery(h).cpu().numpy())
  scores=dict(clean=metrics(np.concatenate(zz),testids,subjects,test.clean),blur=metrics(np.concatenate(hh),testids,subjects,allblur))
  if budget==0:
   expected=json.loads((fold/'heldout_test.json').read_text())['metrics']
   for space in ['clean','blur']:
    for group in ['single_trial','subject_aggregated']:
     for k in ['top1','top5']:assert abs(scores[space][group][k]-expected[space][group][k])*scores[space][group]['count']<=2
  else:torch.save(dict(adapter=adapter.state_dict(),budget=budget,source_checkpoint=str(cp)),out/f'adapter_{budget}.pt')
  results[str(budget)]=dict(metrics=scores,first_loss=losses[0] if losses else None,last_loss=losses[-1] if losses else None)
  (out/'result.json').write_text(json.dumps(results,indent=2));print('BUDGET',budget,{s:{g:scores[s][g] for g in ['single_trial','subject_aggregated']} for s in scores},flush=True)
 (out/'status.json').write_text(json.dumps(dict(state='complete')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.config,a.out)
