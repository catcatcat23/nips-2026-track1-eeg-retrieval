"""Joint THINGS/Alljoined stage1, native channel banks, equal-domain gradient accumulation."""
import argparse,json,time,random,math,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,ConcatDataset,DistributedSampler
from local_ot_train import LocalData,LocalOTModel,MixedTrials,loss_fn,optimizer_groups,write,evaluate as things_evaluate
from alljoined_v67 import load as load_alljoined,evaluate as alljoined_evaluate,Alljoined
from local_ot_train_ema import EMA

class JointModel(LocalOTModel):
 def __init__(self,manifest,all_channels):
  super().__init__(manifest)
  names=[str(n).removeprefix('EEG ').split('-',1)[0] for n in all_channels]
  self.register_buffer('alljoined_positions',self.encoder.get_positions(names))
  self.domain='things'
 def select(self,domain):
  assert domain in ['things','alljoined'];self.domain=domain;return self
 def encode_eeg(self,x):
  positions=self.positions if self.domain=='things' else self.alljoined_positions
  assert x.shape[1]==len(positions),(self.domain,x.shape,positions.shape)
  x=self.preprocess(x);pos=positions.to(x).unsqueeze(0).expand(len(x),-1,-1)
  tokens=self.encoder(x,pos=pos,return_output=True)[-1];h=self.blur_head(tokens.mean(1));z=self.recovery(h)
  keys=self.local_projection(tokens);q,_=self.attention(self.queries.expand(len(x),-1,-1),keys,keys,need_weights=False)
  return h,z,F.normalize(self.local_norm(q),dim=-1)

def seed():
 random.seed(33);np.random.seed(33);torch.manual_seed(33);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 cfg=json.loads(Path(a.config).read_text());root=Path(cfg['run_root']);seed()
 tc=json.loads(Path(cfg['things_config']).read_text());ac=json.loads(Path(cfg['alljoined_config']).read_text())
 manifest=json.loads((Path(tc['source_root'])/'data/manifest.json').read_text())
 meta,atr,ate=load_alljoined(ac);tds=[LocalData(tc,s) for s in ['train','val']];tte=LocalData(tc,'test')
 train_t=np.unique(np.concatenate([d.ids for d in tds]));train_a=np.unique(atr.ids);test_t=np.unique(tte.ids);test_a=np.unique(ate.ids)
 assert np.array_equal(train_t,train_a) and np.array_equal(test_t,test_a)
 assert not np.intersect1d(np.union1d(train_t,train_a),np.union1d(test_t,test_a)).size
 assert Path(ac['clean_features']).resolve()==(Path(tc['source_root'])/'data/clean.npy').resolve()
 assert Path(ac['blur_features']).resolve()==(Path(tc['source_root'])/'features/blur.npy').resolve()
 assert len(set(manifest['channels']))==63 and len(set(meta['channels']))==32
 assert set(atr.subjects).isdisjoint(set(tte.subjects))
 # Resolve canonical image paths as well as IDs: no path alias across train/test.
 paths=manifest['paths'];train_paths={str(Path(paths[int(i)]).resolve()) for i in train_t};test_paths={str(Path(paths[int(i)]).resolve()) for i in test_t};assert not train_paths & test_paths
 # Alljoined original test joins training only after original split audit.
 atr=Alljoined(meta['train']+meta['test'],atr.clean,atr.blur,atr.local)
 assert len(atr)==meta['train_trials']+meta['test_trials']
 assert len(np.unique(atr.ids))==16740
 for field in [atr.clean,atr.blur,atr.local]:
  selected=np.asarray(field[np.unique(atr.ids)])
  assert np.isfinite(selected).all() and np.all(np.linalg.norm(selected.reshape(len(selected),-1),axis=1)>0)
 weights={'things':json.loads((Path(tc['run_root'])/'predictability/subjects.json').read_text()),'alljoined':json.loads((Path(ac['run_root'])/'predictability/subjects.json').read_text())}
 for domain,subs in [('things',np.concatenate([d.subjects for d in tds])),('alljoined',atr.subjects)]:assert set(np.unique(subs))<=set(weights[domain])
 def make():
  torch.manual_seed(33);m=JointModel(manifest,meta['channels']).cuda();m.perturbation_kind='channel';return m
 model=make()
 assert torch.isfinite(model.positions).all() and torch.isfinite(model.alljoined_positions).all()
 assert len(torch.unique(model.positions,dim=0))==63 and len(torch.unique(model.alljoined_positions,dim=0))==32
 def canonical(names):return [str(n).removeprefix('EEG ').split('-',1)[0] for n in names]
 tn=canonical(manifest['channels']);an=canonical(meta['channels'])
 for n in set(tn)&set(an):torch.testing.assert_close(model.positions[tn.index(n)],model.alljoined_positions[an.index(n)],rtol=0,atol=0)
 audit=dict(passed=True,train_images=len(train_t),test_images=len(test_t),original_split_overlap=0,alljoined_original_test_is_training=True,independent_evaluation=False,shared_image_index_verified=True,canonical_paths_disjoint=True,things_train_trials=sum(map(len,tds)),alljoined_train_trials=len(atr),things_test_trials=len(tte),alljoined_test_trials=len(ate),channels={'things':tn,'alljoined':an},positions={'things':model.positions.cpu().tolist(),'alljoined':model.alljoined_positions.cpu().tolist()},sampling='v69 unchanged: equal-domain batches; one THINGS pass and a shuffled Alljoined subset per epoch',steps_per_epoch=sum(map(len,tds))//256,domain_batch=256,effective_update_batch=512)
 write(root/'data_channel_audit.json',audit)
 def w(domain,b):return torch.tensor([weights[domain][s]['weights'] for s in b['subject']],device='cuda')
 if a.preflight:
  # Confirm each native layout produces the same outputs as its single-domain model.
  parity={};tiny={}
  for domain,ds,ch in [('things',tds[0],manifest['channels']),('alljoined',atr,meta['channels'])]:
   _,ix=np.unique(ds.ids,return_index=True);b=next(iter(DataLoader(torch.utils.data.Subset(ds,ix[:8]),batch_size=8)))
   model.select(domain).eval();reference=LocalOTModel({'channels':ch}).cuda().eval();sd=model.state_dict();rs={k:sd[k] for k in reference.state_dict()};rs['positions']=sd['positions' if domain=='things' else 'alljoined_positions'];reference.load_state_dict(rs,strict=True)
   with torch.no_grad():
    actual=model(b['eeg'].cuda());expected=reference(b['eeg'].cuda());torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5);parity[domain]=float((actual-expected).abs().max())
   del reference
   probe=make().select(domain).eval();opt=torch.optim.AdamW(probe.parameters(),lr=1e-4,weight_decay=0);initial=float(loss_fn(probe,b,w(domain,b))[0].detach())
   for step in range(150):
    opt.zero_grad();loss,_=loss_fn(probe,b,w(domain,b));loss.backward();nn.utils.clip_grad_norm_(probe.parameters(),1,error_if_nonfinite=True);opt.step()
    with torch.no_grad():
     h,z,img,_,_=probe(b['eeg'].cuda(),b['blur'].cuda(),b['local'].cuda());labels=torch.arange(8,device='cuda');ab=float(((F.normalize(h,dim=-1)@img.T).argmax(1)==labels).float().mean());az=float(((z@F.normalize(b['clean'].cuda(),dim=-1).T).argmax(1)==labels).float().mean())
    if ab==az==1 and float(loss.detach())<initial*.3:break
   assert ab==az==1 and float(loss.detach())<initial*.3
   tiny[domain]=dict(steps=step+1,initial=initial,final=float(loss.detach()),blur=ab,clean=az);del probe,opt
  model=make().train();model.zero_grad(set_to_none=True);branch={}
  for domain,ds in [('things',tds[0]),('alljoined',atr)]:
   b=next(iter(DataLoader(ds,batch_size=256,num_workers=2)));model.select(domain);loss,_=loss_fn(model,b,w(domain,b));(.5*loss).backward()
   for name,module in [('encoder',model.encoder),('projection',model.blur_head),('adapter',model.image_adapter),('recovery',model.recovery),('attention',model.attention),('local_image',model.image_projection)]:
    gs=[p.grad for p in module.parameters() if p.requires_grad];assert all(g is not None and torch.isfinite(g).all() for g in gs);branch[domain+'/'+name]=sum(float(g.abs().sum()) for g in gs);assert branch[domain+'/'+name]>0
  nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
  write(root/'preflight.json',dict(passed=True,layout_parity=parity,tiny_overfit=tiny,accumulated_branch_gradients=branch,updates_discarded=True,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9));print('PREFLIGHT PASSED',flush=True);return
 assert json.loads((root/'preflight.json').read_text())['passed']
 ds={'things':MixedTrials(ConcatDataset(tds)),'alljoined':MixedTrials(ConcatDataset([atr]))};loaders={};samplers={}
 for d in ds:
  samplers[d]=DistributedSampler(ds[d],num_replicas=1,rank=0,seed=33,drop_last=True)
  loaders[d]=DataLoader(ds[d],batch_size=256,sampler=samplers[d],num_workers=4,persistent_workers=True,pin_memory=True,drop_last=True)
 steps=len(loaders['things']);assert steps<=len(loaders['alljoined'])
 opt=torch.optim.AdamW(optimizer_groups(model,cfg),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[cfg['encoder_lr'],cfg['lr']],total_steps=cfg['epochs']*steps,pct_start=.1)
 emas={f'ema{d}':EMA(model,d) for d in cfg['ema_decays']};evaluator=make().eval();dest=root/'stage1';dest.mkdir();best=-1;stale=0;ranking=[]
 for epoch in range(1,cfg['fixed_training_epochs']+1):
  for s in samplers.values():s.set_epoch(epoch-1)
  model.train();its={d:iter(loaders[d]) for d in loaders};total={d:np.zeros(4) for d in loaders};start=time.time();norms=[]
  write(dest/'status.json',dict(state='training',epoch=epoch,steps=steps,stale=stale,best=best))
  for step in range(steps):
   opt.zero_grad(set_to_none=True)
   for d in ['things','alljoined']:
    b=next(its[d]);model.select(d);loss,parts=loss_fn(model,b,w(d,b));assert torch.isfinite(loss);(.5*loss).backward();total[d]+=np.array([float(loss.detach())]+[float(parts[k].detach()) for k in ['clean','blur','local']])
   norms.append(float(nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)));opt.step();sched.step()
   for ema in emas.values():ema.update(model)
   if step%100==0:print('TRAIN',epoch,step,steps,{d:(total[d]/(step+1)).tolist() for d in total},flush=True)
  results={}
  for variant in ['raw']+list(emas):
   m=model if variant=='raw' else evaluator
   if variant!='raw':m.load_state_dict(emas[variant].state_dict(model),strict=True)
   m.select('things');tm=things_evaluate(m,tc,weights['things']);m.select('alljoined');am=alljoined_evaluate(m,ate)
   results[variant]={'things':tm,'alljoined':am};score=None
   path=dest/f'{variant}_epoch_{epoch:03d}.pt';state=model.state_dict() if variant=='raw' else emas[variant].state_dict(model)
   torch.save(dict(model=state,manifest=manifest,alljoined_channels=meta['channels'],epoch=epoch,variant=variant,config=cfg,metrics=results[variant],selection_score=score),path)
   ranking.append(dict(epoch=epoch,variant=variant,score=score,path=path.name))
  # Fixed-budget refit: never rank checkpoints using the now-trained public test.
  stale=0;best=None
  write(dest/'checkpoint_inventory.json',ranking)
  torch.save(dict(model=model.state_dict(),ema={v:e.state_dict(model) for v,e in emas.items()},optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,config=cfg,best=best,stale=stale,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),dest/'last.pt')
  row=dict(epoch=epoch,loss={d:(total[d]/steps).tolist() for d in total},metrics=results,selection='none; fixed budget',stale=stale,seconds=time.time()-start,lr=[g['lr'] for g in opt.param_groups],gradient_norm_mean=float(np.mean(norms)))
  with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print('EPOCH',epoch,'fixed budget',cfg['fixed_training_epochs'],flush=True)
  # No early stopping on in-sample diagnostic metrics.
 write(dest/'result.json',dict(stopped_epoch=epoch,selection='fixed final epoch EMA .9998; no metric selection',role='full-public-data refit; old tests are in-sample/seen-image diagnostics'));write(dest/'status.json',dict(state='complete',epoch=epoch))
if __name__=='__main__':main()
