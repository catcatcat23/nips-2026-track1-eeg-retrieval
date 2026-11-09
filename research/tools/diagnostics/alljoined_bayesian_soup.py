"""Alljoined clean-output ensemble and GP/EI scalar parameter-soup search."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,ConstantKernel,WhiteKernel
from scipy.stats import norm
from alljoined_v67 import load,LocalOTModel
BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
CHOICES=[('reve_v69_joint_stage1','ema0.9998_epoch_015.pt'),('reve_v85_alljoined_identity_global','ema0.9998_epoch_008.pt')]
class EEGOnly(Dataset):
 def __init__(self,ds):self.ds=ds
 def __len__(self):return len(self.ds)
 def __getitem__(self,i):return torch.tensor(np.array(self.ds.eeg(i),dtype=np.float32))
def main(root):
 root.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(33)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def write(n,d): (root/n).write_text(json.dumps(d,indent=2))
 cfg=json.loads((BASE/CHOICES[1][0]/'config.json').read_text());meta,tr,ds=load(cfg);del tr
 ids,y=np.unique(ds.ids,return_inverse=True);_,sid=np.unique(ds.subjects,return_inverse=True);group=sid*len(ids)+y
 count=np.bincount(group,minlength=4000);assert np.all(count>0) and len(ids)==200
 tune=np.sort(np.random.default_rng(910033).choice(200,100,replace=False));hold=np.setdiff1d(np.arange(200),tune)
 write('split.json',dict(tune_image_ids=ids[tune].tolist(),holdout_image_ids=ids[hold].tolist(),gallery=ids.tolist(),limitation='Public test historically used for checkpoint selection; split protects only this weight search.'))
 states=[];history=[]
 for name,file in CHOICES:
  cp=BASE/name/'stage1'/file;s=torch.load(cp,map_location='cpu',weights_only=False);sd=s['model']
  if 'alljoined_positions' in sd:sd=dict(sd);sd['positions']=sd.pop('alljoined_positions')
  states.append(sd);m=s['metrics'];history.append(m.get('alljoined',m))
 assert states[0].keys()==states[1].keys()
 for k in states[0]:assert states[0][k].shape==states[1][k].shape,k
 torch.testing.assert_close(states[0]['positions'],states[1]['positions'],rtol=0,atol=0)
 model=LocalOTModel({'channels':meta['channels']}).cuda().eval().requires_grad_(False)
 gallery=F.normalize(torch.tensor(ds.clean[ids],device='cuda'),dim=-1)
 loader=DataLoader(EEGOnly(ds),batch_size=256,num_workers=2,pin_memory=True,persistent_workers=True)
 def blend(a):
  return {k:(a*v+(1-a)*states[1][k] if v.is_floating_point() and k!='positions' else v.clone()) for k,v in states[0].items()}
 def scores_metrics(scores,which):
  mask=np.isin(y,which);agg=np.zeros((4000,200),np.float64)
  np.add.at(agg,group,scores);agg/=count[:,None]
  gm=np.isin(np.tile(np.arange(200),20),which);labels=np.tile(np.arange(200),20)
  def metric(s,l):
   top=np.argsort(-s,axis=1)[:,:5];return dict(top1=float((top[:,0]==l).mean()),top5=float((top==l[:,None]).any(1).mean()),count=len(l))
  return dict(single_trial=metric(scores[mask],y[mask]),subject_aggregated=metric(agg[gm],labels[gm]))
 @torch.inference_mode()
 def predict(a,tag,unit=False):
  path=root/f'{tag}.npy'
  if path.exists() and (root/f'{tag}.done').exists():return np.load(path,mmap_mode='r')
  model.load_state_dict(blend(a),strict=True);model.eval()
  out=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(ds),200));off=0
  for x in loader:
   x=x.cuda();h,z,_=model.encode_eeg(x)
   assert torch.isfinite(z).all()
   if unit:z=F.normalize(z,dim=-1)
   out[off:off+len(x)]=(z@gallery.T).cpu().numpy();off+=len(x)
   if off%25600==0:write('status.json',dict(state='inference',candidate=tag,trials=off,total=len(ds)))
  out.flush();assert off==len(ds);(root/f'{tag}.done').write_text('complete');return out
 # Cached scores are sufficient: cosine denominator is common to all gallery candidates;
 # average raw scores by subject-image is exactly ranking average output embeddings.
 endpoints={}
 for a,name,ix in [(1.,'v69',0),(0.,'v85',1)]:
  raw=predict(a,name);actual=scores_metrics(raw,np.arange(200))
  for g in actual:
   for k in ['top1','top5']:assert abs(actual[g][k]-history[ix]['clean'][g][k])*actual[g]['count']<=10,(name,g,k)
  endpoints[name]=(raw,predict(a,name+'_unit',True))
 write('endpoint_replay.json',{'passed':True})
 comparisons=[]
 for kind,col in [('raw',0),('unit',1)]:
  for a in [0.,.25,.5,.75,1.]:
   s=a*endpoints['v69'][col]+(1-a)*endpoints['v85'][col]
   comparisons.append(dict(kind=kind,alpha=a,tune=scores_metrics(s,tune)))
 selected=max(comparisons,key=lambda r:r['tune']['subject_aggregated']['top5'])
 col=0 if selected['kind']=='raw' else 1;a=selected['alpha'];s=a*endpoints['v69'][col]+(1-a)*endpoints['v85'][col]
 write('output_ensemble.json',dict(search=comparisons,selected=selected,holdout=scores_metrics(s,hold),full_public_diagnostic=scores_metrics(s,np.arange(200))))
 trials=[]
 for step in range(12):
  if step<5:a=[0.,1.,.5,.25,.75][step]
  else:
   gp=GaussianProcessRegressor(kernel=ConstantKernel(1.)*Matern(length_scale=.2,nu=2.5)+WhiteKernel(1e-5),normalize_y=True,random_state=33,n_restarts_optimizer=2)
   gp.fit(np.array([t['alpha'] for t in trials])[:,None],np.array([t['score'] for t in trials]))
   grid=np.linspace(0,1,2001);mu,sd=gp.predict(grid[:,None],return_std=True);improve=mu-max(t['score'] for t in trials)-.0001;z=improve/np.maximum(sd,1e-12);ei=improve*norm.cdf(z)+sd*norm.pdf(z)
   for t in trials:ei[np.abs(grid-t['alpha'])<.002]=-np.inf
   a=float(grid[np.argmax(ei)])
  tag='v69' if a==1 else 'v85' if a==0 else f'soup_{a:.5f}'
  s=predict(a,tag);m=scores_metrics(s,tune);trials.append(dict(alpha=a,score=m['subject_aggregated']['top5'],tune=m,tag=tag));write('bayes_trials.json',trials);print('TRIAL',step,trials[-1],flush=True)
 best=max(trials,key=lambda r:r['score']);sd=blend(best['alpha']);model.load_state_dict(sd,strict=True)
 cp=root/'best_soup.pt';torch.save(dict(model={k:v.cpu() for k,v in sd.items()},manifest={'channels':meta['channels']},config=cfg,ensemble_members=CHOICES,alpha=best['alpha'],selection='100-image tuning clean subject-aggregated Top5; GP expected improvement; 12 evaluations'),cp)
 # Verify the saved standalone model reloads and reproduces the in-memory model.
 x=next(iter(loader))[:8].cuda()
 with torch.inference_mode():expected=model(x);saved=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);torch.testing.assert_close(model(x),expected)
 s=np.load(root/f"{best['tag']}.npy",mmap_mode='r')
 write('result.json',dict(best=best,holdout=scores_metrics(s,hold),full_public_diagnostic=scores_metrics(s,np.arange(200)),baseline_holdout={n:scores_metrics(v[0],hold) for n,v in endpoints.items()},single_model=str(cp),reload_passed=True,interpretation='Endpoint optimum is allowed: parameter mixing may fail despite output-ensemble gains. No hidden-test claim.'))
 write('status.json',dict(state='complete'))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();main(a.root)
