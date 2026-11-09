"""Three/five clean-output ensemble; fixed image tuning/holdout split."""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from alljoined_bayesian_soup import BASE,EEGOnly
from alljoined_v67 import load,LocalOTModel
OLD=BASE/'alljoined_bayesian_soup_v69_v85_20261106'
MEMBERS=[('v69','reve_v69_joint_stage1',15),('v85','reve_v85_alljoined_identity_global',8),('v86','reve_v86_alljoined_identity_image',8),('v87','reve_v87_alljoined_identity_concept',8),('v89','reve_v89_alljoined_identity_blur',8)]
def main(root):
 root.mkdir(exist_ok=True);start=time.time();torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def write(n,x):(root/n).write_text(json.dumps(x,indent=2))
 cfg=json.loads((BASE/MEMBERS[1][1]/'config.json').read_text());meta,tr,ds=load(cfg);del tr
 ids,y=np.unique(ds.ids,return_inverse=True);_,sid=np.unique(ds.subjects,return_inverse=True);group=sid*len(ids)+y;count=np.bincount(group)
 split=json.loads((OLD/'split.json').read_text());tune=np.flatnonzero(np.isin(ids,split['tune_image_ids']));hold=np.flatnonzero(np.isin(ids,split['holdout_image_ids']))
 assert len(ids)==200 and len(tune)==len(hold)==100 and len(count)==4000 and (count>0).all()
 write('protocol.json',dict(members=MEMBERS,split=split,weight_grid='0.1 increments, every selected member >=0.1; original two-model 0.75/0.25 baseline',selection='clean subject-aggregated Top5 on tuning images only; third member selected from v86/v87/v89',normalization='raw outputs; no per-model L2',limitation='Public test previously used for checkpoint selection; holdout applies only to ensemble search',samples=len(ds)))
 model=LocalOTModel({'channels':meta['channels']}).cuda().eval();gallery=F.normalize(torch.tensor(ds.clean[ids],device='cuda'),dim=-1)
 loader=DataLoader(EEGOnly(ds),batch_size=256,num_workers=2,pin_memory=True)
 scores=[];aggregates=[]
 for name,run,epoch in MEMBERS:
  path=(OLD if name in ['v69','v85'] else root)/f'{name}.npy';done=path.with_suffix('.done')
  if not (path.exists() and done.exists()):
   ck=torch.load(BASE/run/'stage1'/f'ema0.9998_epoch_{epoch:03d}.pt',map_location='cpu',weights_only=False);state=ck['model']
   if 'alljoined_positions' in state:state=dict(state);state['positions']=state.pop('alljoined_positions')
   model.load_state_dict(state,strict=True);model.eval()
   out=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(ds),200));off=0
   with torch.inference_mode():
    for x in loader:
     z=model(x.cuda());s=(z@gallery.T).cpu().numpy();assert np.isfinite(s).all();out[off:off+len(x)]=s;off+=len(x)
     if off%25600==0:write('status.json',dict(state='inference',model=name,samples=off,total=len(ds)));print(name,off,flush=True)
   assert off==len(ds);out.flush();done.write_text('complete')
  s=np.load(path,mmap_mode='r');scores.append(s);agg=np.zeros((4000,200),np.float64);np.add.at(agg,group,s);agg/=count[:,None];aggregates.append(agg)
 aggregates=np.stack(aggregates);labels=np.tile(np.arange(200),20);tm=np.isin(labels,tune)
 def metric(s,l):
  top=np.argsort(-s,axis=1)[:,:5];return dict(top1=float((top[:,0]==l).mean()),top5=float((top==l[:,None]).any(1).mean()),count=len(l))
 def evaluate(w):
  agg=np.einsum('i,ijk->jk',w,aggregates);hits={k:np.zeros(2,np.int64) for k in ['tune','holdout','full']}
  for off in range(0,len(ds),4096):
   s=sum(float(a)*v[off:off+4096] for a,v in zip(w,scores) if a);l=y[off:off+4096];top=np.argsort(-s,axis=1)[:,:5];h=np.stack([top[:,0]==l,(top==l[:,None]).any(1)],axis=1)
   for key,which in [('tune',tune),('holdout',hold),('full',np.arange(200))]:hits[key]+=h[np.isin(l,which)].sum(0)
  result={}
  for key,which in [('tune',tune),('holdout',hold),('full',np.arange(200))]:
   n=int(np.isin(y,which).sum());gm=np.isin(labels,which);result[key]=dict(single_trial=dict(top1=float(hits[key][0]/n),top5=float(hits[key][1]/n),count=n),subject_aggregated=metric(agg[gm],labels[gm]))
  return dict(weights={MEMBERS[i][0]:float(v) for i,v in enumerate(w) if v},metrics=result)
 results={'two_model_baseline':evaluate([.75,.25,0,0,0])};search=[]
 for n in [3,5]:
  subsets=[(0,1,k) for k in [2,3,4]] if n==3 else [(0,1,2,3,4)]
  best=None
  for subset in subsets:
   equal=np.zeros(5);equal[list(subset)]=1/n
   results['equal_'+ '_'.join(MEMBERS[i][0] for i in subset)]=evaluate(equal)
   for cuts in itertools.combinations(range(1,10),n-1):
    parts=np.diff([0,*cuts,10])/10.;w=np.zeros(5);w[list(subset)]=parts
    agg=np.einsum('i,ijk->jk',w,aggregates[:,tm]);score=metric(agg,labels[tm])['top5'];search.append(dict(n=n,weights=w.tolist(),score=score))
    if best is None or score>best[0]:best=(score,w.copy())
  results[f'weighted_{n}']=evaluate(best[1]);write('results.json',results);print(n,results[f'weighted_{n}'],flush=True)
 write('search.json',search);write('results.json',results);write('status.json',dict(state='complete',seconds=time.time()-start))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
