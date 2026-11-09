"""Paired EMA mechanism audit. Read-only models, public test diagnostics."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from joint_train import JointModel,LocalData,load_alljoined,seed
root=Path(sys.argv[1]);domain=sys.argv[2];out=root/domain;out.mkdir(exist_ok=True)
base=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v69_joint_stage1');cfg=json.loads((base/'config.json').read_text());tc=json.loads(Path(cfg['things_config']).read_text());ac=json.loads(Path(cfg['alljoined_config']).read_text());seed()
class Probe(JointModel):
 condition='none'
 def preprocess(self,x):
  x=super().preprocess(x)
  if self.condition.startswith('noise'):
   x=x+float(self.condition.split('_')[1])*torch.randn_like(x)
  elif self.condition.startswith('mask'):
   n=int(self.condition.split('_')[1]);cols=torch.rand(len(x),x.shape[1],device=x.device).argsort(1)[:,:n];x=x.clone();x.scatter_(1,cols[:,:,None].expand(-1,-1,x.shape[2]),0)
  return x
ck=torch.load(base/'stage1/ema0.9998_epoch_015.pt',map_location='cpu',weights_only=False);model=Probe(ck['manifest'],ck['alljoined_channels']).cuda().eval().select(domain);del ck
if domain=='things':ds=LocalData(tc,'test')
else:
 _,tr,ds=load_alljoined(ac);del tr
ids=np.unique(ds.ids);subs=np.unique(ds.subjects);assert len(ids)==200
labels=np.searchsorted(ids,ds.ids);sid=np.searchsorted(subs,ds.subjects);group=sid*200+labels
rng=np.random.default_rng(101);sample=np.concatenate([rng.choice(np.flatnonzero(group==i),8,replace=False) for i in range(len(subs)*200)])
np.savez(out/'indices.npz',ids=ds.ids,subjects=ds.subjects,sampled_indices=sample)
blur=[]
for i in ids:
 with np.load(Path(tc['test_blur_shards'])/f'{i:05d}.npz') as f:assert int(f['id'])==i;blur.append(f['blur'].copy())
blur=torch.tensor(np.stack(blur),device='cuda');clean=F.normalize(torch.tensor(ds.clean[ids],device='cuda'),dim=-1)
variants={'raw15':['raw_epoch_015.pt'],'ema15':['ema0.9998_epoch_015.pt'],'raw_best_aggregate':['raw_epoch_010.pt' if domain=='things' else 'raw_epoch_012.pt'],'swa13_15':[f'raw_epoch_{e:03d}.pt' for e in [13,14,15]]}
def load_variant(names):
 state=None
 for name in names:
  c=torch.load(base/'stage1'/name,map_location='cpu',weights_only=False)['model']
  if state is None:state={k:v.clone()/len(names) if v.is_floating_point() else v.clone() for k,v in c.items()}
  else:
   for k,v in c.items():
    if v.is_floating_point():state[k]+=v/len(names)
    else:assert torch.equal(state[k],v)
 model.load_state_dict(state,strict=True)
def score_metrics(scores,y):
 top=np.argsort(-scores,axis=1)[:,:5];ok1=top[:,0]==y;ok5=(top==y[:,None]).any(1);wrong=scores.copy();wrong[np.arange(len(y)),y]=-np.inf;gap=scores[np.arange(len(y)),y]-wrong.max(1)
 return dict(top1=float(ok1.mean()),top5=float(ok5.mean()),margin=float(gap.mean()),count=len(y)),ok1,ok5,gap
@torch.inference_mode()
def evaluate(variant,condition,sd,indices):
 model.condition=condition;torch.manual_seed(sd);torch.cuda.manual_seed_all(sd)
 gallery=F.normalize(model.image_adapter(blur),dim=-1);ys=labels[indices];gs=group[indices];ng=len(subs)*200
 sums={s:np.zeros((ng,1536),np.float64) for s in ['clean','blur']};units={s:np.zeros((ng,1536),np.float64) for s in sums};scores={s:[] for s in sums};norms={s:[] for s in sums};pos=0
 for b in DataLoader(Subset(ds,indices.tolist()),batch_size=128,num_workers=2):
  h,z,_=model.encode_eeg(b['eeg'].cuda());gids=gs[pos:pos+len(h)];pos+=len(h)
  for s,p,g in [('clean',z,clean),('blur',h,gallery)]:
   assert torch.isfinite(p).all();pn=F.normalize(p,dim=-1);np.add.at(sums[s],gids,p.cpu().numpy());np.add.at(units[s],gids,pn.cpu().numpy());scores[s].append((pn@g.T).cpu().numpy());norms[s].append(p.norm(dim=-1).cpu().numpy())
 result={};arrays={}
 for s,g in [('clean',clean),('blur',gallery)]:
  sc=np.concatenate(scores[s]);n=np.concatenate(norms[s]);single,ok1,ok5,gap=score_metrics(sc,ys);gg=g.cpu().numpy();agg={}
  for kind,vector in [('raw_mean',sums[s]),('unit_mean',units[s])]:
   vector=vector/np.maximum(np.linalg.norm(vector,axis=1,keepdims=True),1e-12);ag,a1,a5,am=score_metrics(vector@gg.T,np.tile(np.arange(200),len(subs)));agg[kind]=ag;arrays[s+'_'+kind+'_top1']=a1;arrays[s+'_'+kind+'_top5']=a5
  result[s]=dict(single=single,aggregate=agg,norm=dict(mean=float(n.mean()),std=float(n.std()),p05=float(np.quantile(n,.05)),p50=float(np.median(n)),p95=float(np.quantile(n,.95)),correct_mean=float(n[ok1].mean()) if ok1.any() else None,incorrect_mean=float(n[~ok1].mean()) if (~ok1).any() else None))
  arrays.update({s+'_norm':n,s+'_top1':ok1,s+'_top5':ok5,s+'_margin':gap})
 name=f'{variant}_{condition}_seed{sd}';np.savez_compressed(out/(name+'.npz'),**arrays)
 record=dict(variant=variant,condition=condition,seed=sd,full_test=condition=='none' and sd==0,metrics=result)
 with (out/'results.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 print('DONE',domain,name,json.dumps(result),flush=True)
for variant,names in variants.items():
 load_variant(names);evaluate(variant,'none',0,np.arange(len(ds)))
 if variant in ['raw15','ema15']:
  evaluate(variant,'none',101,sample)
  for sd in [101,102,103]:
   for condition in ['noise_0.02','noise_0.05','noise_0.1','mask_1','mask_4']:evaluate(variant,condition,sd,sample)
(out/'status.txt').write_text('COMPLETE\n')
