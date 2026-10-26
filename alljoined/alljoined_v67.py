"""v67: v64 stage1 recipe (imports v57/v64 model, loss, augmentation, EMA unchanged) on Alljoined2025.
Only new code: record-based Alljoined dataset, Alljoined subject predictability weights, blur+clean evaluation."""
import argparse,json,time,random,hashlib
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset,DistributedSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from local_ot_train import LocalOTModel,MixedTrials,loss_fn,optimizer_groups,write
from local_ot_train_ema import EMA,variants
from local_ot_weights import correlation
from v33_recovery_stage2 import metrics

class Alljoined(Dataset):
 """One item per official event; same fields as v64 LocalData (eeg, id, subject, clean, blur, local)."""
 def __init__(self,records,clean,blur,local):
  self.records=records;self.clean=clean;self.blur=blur;self.local=local;self.maps=OrderedDict()
  self.ends=np.cumsum([len(r['events']) for r in records])
  self.ids=np.concatenate([np.array([i for _,i in r['events']]) for r in records])
  self.subjects=np.concatenate([np.repeat(r['subject'],len(r['events'])) for r in records])
  assert all(r['channels']==records[0]['channels'] for r in records)
 def __len__(self):return len(self.ids)
 def eeg(self,i):
  ri=int(np.searchsorted(self.ends,i,side='right'));j=i-(int(self.ends[ri-1]) if ri else 0);key=self.records[ri]['array']
  if key not in self.maps:
   if len(self.maps)>=64:self.maps.popitem(last=False)
   self.maps[key]=np.load(key,mmap_mode='r');assert self.maps[key].shape==(len(self.records[ri]['events']),32,120)
  self.maps.move_to_end(key);return self.maps[key][j]
 def __getitem__(self,i):
  k=int(self.ids[i])
  return {'eeg':torch.from_numpy(np.array(self.eeg(i),dtype=np.float32)),'id':k,'subject':str(self.subjects[i]),'clean':torch.from_numpy(self.clean[k].copy()),
          'blur':torch.from_numpy(self.blur[k].copy()),'local':torch.from_numpy(np.array(self.local[k]))}

def load(cfg):
 meta=json.loads(Path(cfg['data_manifest']).read_text());clean=np.load(cfg['clean_features']);blur=np.load(cfg['blur_features']).copy()
 testids=sorted({i for r in meta['test'] for _,i in r['events']})
 for i in testids:
  with np.load(Path(cfg['test_blur_shards'])/f'{i:05d}.npz') as f:
   assert int(f['id'])==i and str(f['schema'])=='upstream_per_scale_l2_dino_v1';blur[i]=f['blur']
 local=np.load(cfg['local_features'],mmap_mode='r')
 tr=Alljoined(meta['train'],clean,blur,local);te=Alljoined(meta['test'],clean,blur,local)
 assert len(np.unique(tr.ids))==16540 and len(np.unique(te.ids))==200 and not np.intersect1d(tr.ids,te.ids).size
 assert len(tr)==meta['train_trials'] and len(te)==meta['test_trials']
 return meta,tr,te

def predictability(cfg,tr,dest):
 """Same procedure as v64 local_ot_weights.py, on Alljoined training trials (32 channels); no test EEG."""
 dest.mkdir(exist_ok=True);result={}
 for subject in np.unique(tr.subjects):
  idx=np.flatnonzero(tr.subjects==subject);targetids=np.unique(tr.ids[idx])
  x=np.zeros((len(targetids),32,24),np.float64);count=np.zeros(len(targetids),np.int64)
  for i in idx:
   m=np.searchsorted(targetids,tr.ids[i]);x[m]+=np.asarray(tr.eeg(int(i)),np.float64).reshape(32,24,5).mean(-1);count[m]+=1
  assert (count>0).all();x=(x/count[:,None,None]).reshape(len(x),-1).astype('float32');y=np.asarray(tr.local[targetids]);assert np.isfinite(y).all()
  folds=[]
  for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=50).split(x)):
   scale=StandardScaler().fit(x[a]);px=PCA(n_components=128,svd_solver='randomized',random_state=33);xt=px.fit_transform(scale.transform(x[a]));xv=px.transform(scale.transform(x[b]))
   key=hashlib.sha256(targetids[a].tobytes()).hexdigest()[:16];vp=dest/f'visual_{key}.npz'
   if vp.exists():
    with np.load(vp) as f:mean=f['mean'];components=f['components']
   else:
    py=PCA(n_components=64,svd_solver='randomized',random_state=33).fit(y[a].reshape(-1,1536));mean=py.mean_;components=py.components_;np.savez(vp,mean=mean,components=components)
   yt=((y[a]-mean)@components.T).reshape(len(a),-1);yv=((y[b]-mean)@components.T).reshape(len(b),-1)
   score=np.median(np.maximum(correlation(Ridge(alpha=100.).fit(xt,yt).predict(xv),yv).reshape(9,64),0),axis=1);folds.append(score)
  scores=np.mean(folds,axis=0);weights=np.clip(scores/max(float(np.percentile(scores,90)),1e-6),.1,1.)**2
  result[str(subject)]={'weights':weights.tolist(),'scores':scores.tolist(),'fold_scores':[f.tolist() for f in folds],'images':len(targetids),'mean_training_repeats':float(count.mean())}
  print('WEIGHTS',subject,'images',len(targetids),'weights',np.round(weights,3).tolist(),flush=True)
  write(dest/'subjects.json',result)
 write(dest/'complete.json',dict(subjects=list(result),procedure='v64 local_ot_weights.py on Alljoined2025 training trials, 32 channels',test_used=False))

@torch.no_grad()
def evaluate(model,ds):
 model.eval();hp=[];zp=[]
 for b in DataLoader(ds,batch_size=512,num_workers=4):
  h,z,_=model.encode_eeg(b['eeg'].cuda());hp.append(h.cpu().numpy());zp.append(z.cpu().numpy())
 ids=np.unique(ds.ids);gallery=np.zeros_like(ds.blur)
 gallery[ids]=F.normalize(model.image_adapter(torch.from_numpy(ds.blur[ids]).cuda()),dim=-1).cpu().numpy()
 return {'blur':metrics(np.concatenate(hp),ds.ids,ds.subjects,gallery),'clean':metrics(np.concatenate(zp),ds.ids,ds.subjects,ds.clean)}

def preflight(cfg,model,tr,weights,root):
 b=next(iter(DataLoader(MixedTrials(ConcatDataset([tr])),batch_size=64,shuffle=True,num_workers=2)))
 model.train();w=torch.tensor([weights[s]['weights'] for s in b['subject']],device='cuda');loss,parts=loss_fn(model,b,w);loss.backward();grads={}
 for name,module in [('encoder',model.encoder),('global_projection',model.blur_head),('global_adapter',model.image_adapter),('recovery',model.recovery),('queries_attention',model.attention),('local_image',model.image_projection)]:
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters() if p.requires_grad),name
  grads[name]=sum(p.grad.abs().sum().item() for p in module.parameters());assert grads[name]>0,name
 model.zero_grad(set_to_none=True)
 _,ix=np.unique(tr.ids,return_index=True);items=[tr[int(i)] for i in ix[:8]]
 b={k:(torch.stack([s[k] for s in items]) if torch.is_tensor(items[0][k]) else torch.tensor([s[k] for s in items]) if k=='id' else [s[k] for s in items]) for k in items[0]}
 w=torch.tensor([weights[s]['weights'] for s in b['subject']],device='cuda');opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0);model.eval();initial=None
 for step in range(300):
  opt.zero_grad();loss,_=loss_fn(model,b,w);initial=initial or float(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  with torch.no_grad():
   h,z,a,_,_=model(b['eeg'].cuda(),b['blur'].cuda(),b['local'].cuda());lab=torch.arange(8,device='cuda')
   ab=float(((F.normalize(h,dim=-1)@a.T).argmax(1)==lab).float().mean());ac=float(((F.normalize(z,dim=-1)@F.normalize(b['clean'].cuda(),dim=-1).T).argmax(1)==lab).float().mean())
  if ab==ac==1 and float(loss)<initial*.3:break
 assert ab==ac==1 and float(loss)<initial*.3,(ab,ac,float(loss),initial)
 write(root/'preflight.json',dict(passed=True,batch64_loss=float(parts['clean']+parts['blur']),gradients=grads,tiny_steps=step+1,tiny_initial=initial,tiny_final=float(loss),updates_discarded=True,peak_gpu_mem_gb=torch.cuda.max_memory_allocated()/1e9))
 print('PREFLIGHT PASSED',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 cfg=json.loads(a.config.read_text());root=Path(cfg['run_root']);torch.set_num_threads(4)
 random.seed(33);np.random.seed(33);torch.manual_seed(33);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 meta,tr,te=load(cfg)
 if not (root/'predictability/complete.json').exists():predictability(cfg,tr,root/'predictability')
 weights=json.loads((root/'predictability/subjects.json').read_text());assert set(weights)==set(np.unique(tr.subjects))
 manifest={'channels':meta['channels']}
 if a.preflight:
  torch.manual_seed(33);model=LocalOTModel(manifest).cuda();model.perturbation_kind=cfg['augmentation'];preflight(cfg,model,tr,weights,root);return
 assert (root/'preflight.json').exists() and not (root/'stage1').exists()
 torch.manual_seed(33);model=LocalOTModel(manifest).cuda();model.perturbation_kind=cfg['augmentation']
 ds=MixedTrials(ConcatDataset([tr]));sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=33,drop_last=True)
 loader=DataLoader(ds,batch_size=256,sampler=sampler,num_workers=6,pin_memory=True,persistent_workers=True,drop_last=True)
 opt=torch.optim.AdamW(optimizer_groups(model,cfg),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[cfg['encoder_lr'],cfg['lr']],total_steps=cfg['epochs']*len(loader),pct_start=.1)
 emas={f"ema{d}":EMA(model,d) for d in cfg['ema_decays']}
 evaluator=LocalOTModel(manifest).cuda().eval();evaluator.perturbation_kind=cfg['augmentation']
 dest=root/'stage1';dest.mkdir();best={v:-1. for v in variants(cfg)};best_epoch={v:0 for v in variants(cfg)};stale=0
 for epoch in range(1,cfg['epochs']+1):
  sampler.set_epoch(epoch-1);model.train();total=np.zeros(4);norms=[];start=time.time()
  write(dest/'status.json',dict(state='training',epoch=epoch,steps=len(loader),best=best,best_epoch=best_epoch,stale=stale))
  for step,b in enumerate(loader):
   w=torch.tensor([weights[s]['weights'] for s in b['subject']],device='cuda');opt.zero_grad();loss,parts=loss_fn(model,b,w);assert torch.isfinite(loss);loss.backward()
   norms.append(float(nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)));opt.step();sched.step()
   for e in emas.values():e.update(model)
   total+=np.array([float(loss.detach())]+[float(parts[k].detach()) for k in ['clean','blur','local']])
   if step%200==0:print('TRAIN',epoch,step,len(loader),'loss',float(loss.detach()),'gradnorm',norms[-1],'lr',[g['lr'] for g in opt.param_groups],flush=True)
  train_seconds=time.time()-start;results={};eval_seconds={}
  for v in variants(cfg):
   t=time.time()
   if v=='raw':results[v]=evaluate(model,te)
   else:evaluator.load_state_dict(emas[v].state_dict(model),strict=True);results[v]=evaluate(evaluator,te)
   eval_seconds[v]=time.time()-t
  torch.save(dict(model=model.state_dict(),manifest=manifest,config=cfg,epoch=epoch,metrics=results['raw']),dest/f'epoch_{epoch:03d}.pt')
  for v,e in emas.items():torch.save(dict(model=e.state_dict(model),manifest=manifest,config=cfg,epoch=epoch,metrics=results[v],ema_decay=e.decay),dest/f'{v}_epoch_{epoch:03d}.pt')
  torch.save(dict(model=model.state_dict(),ema={v:e.state_dict(model) for v,e in emas.items()},manifest=manifest,config=cfg,epoch=epoch,optimizer=opt.state_dict(),scheduler=sched.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),best=best,best_epoch=best_epoch),dest/'last.pt')
  improved=False
  for v in variants(cfg):
   score=results[v]['blur'][cfg['checkpoint_group']]['top5']
   if score>best[v]:best[v]=score;best_epoch[v]=epoch;improved=True
  stale=0 if improved else stale+1
  n=np.array(norms);row=dict(epoch=epoch,loss=(total/len(loader)).tolist(),loss_parts=['total','clean','blur','local'],lr_end=[g['lr'] for g in opt.param_groups],grad_norm=dict(mean=float(n.mean()),max=float(n.max()),clipped_fraction=float((n>1).mean())),metrics=results,best=best,best_epoch=best_epoch,stale=stale,train_seconds=train_seconds,eval_seconds=eval_seconds)
  with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print('EPOCH',epoch,json.dumps({v:{f'{m}_{g}':{k:results[v][m][g][k] for k in ['top1','top5']} for m in ['blur','clean'] for g in ['single_trial','subject_aggregated']} for v in results}),'stale',stale,flush=True)
  if stale>=cfg['patience']:break
 write(dest/'result.json',dict(best=best,best_epoch=best_epoch,selection=f"blur {cfg['checkpoint_group']} Top5 per variant; Alljoined2025 public test development, not independent",stopped_epoch=epoch))
 write(dest/'status.json',dict(state='complete',epoch=epoch,best=best,best_epoch=best_epoch))
if __name__=='__main__':main()
