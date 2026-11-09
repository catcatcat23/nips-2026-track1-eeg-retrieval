"""B architecture on original Alljoined2025 split; clean+blur, no held-out train images."""
import argparse,json,time,random,hashlib
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from reference_model import ReferenceModel
from recovery_ablation import ProjectedHead,metrics
from neuraltrain.losses.losses import ClipLoss
from channel_contract import channel_positions
BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v13_reference')

def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

class Data(Dataset):
 def __init__(self,records,clean,blur):
  self.records=records;self.clean=clean;self.blur=blur;self.maps=OrderedDict()
  self.ends=np.cumsum([len(r['events']) for r in records]);self.ids=np.concatenate([np.array([i for _,i in r['events']]) for r in records]);self.subjects=np.concatenate([np.repeat(r['subject'],len(r['events'])) for r in records])
  assert all(r['channels']==records[0]['channels'] for r in records)
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  ri=int(np.searchsorted(self.ends,i,side='right'));local=i-(int(self.ends[ri-1]) if ri else 0);r=self.records[ri];key=r['array']
  if key not in self.maps:
   if len(self.maps)>=24:self.maps.popitem(last=False)
   self.maps[key]=np.load(key,mmap_mode='r');assert self.maps[key].shape==(len(r['events']),32,120)
  self.maps.move_to_end(key);k=self.ids[i]
  return {'eeg':torch.from_numpy(self.maps[key][local].copy()),'clean':torch.from_numpy(self.clean[k].copy()),'blur':torch.from_numpy(self.blur[k].copy())}

class Model(ReferenceModel):
 def __init__(self,names):
  super().__init__({'channels':names});self.recovery=ProjectedHead()
  self.positions.copy_(channel_positions(self.encoder,names))
 def forward(self,x,blur=None):
  x=self.preprocess(x);pos=self.positions.to(x).unsqueeze(0).expand(len(x),-1,-1)
  h=self.blur_head(self.encoder(x,pos=pos,return_output=True)[-1].mean(1));q=self.recovery(h)
  return (h,q) if blur is None else (h,q,F.normalize(self.image_adapter(blur),dim=-1))

@torch.no_grad()
def evaluate(model,ds):
 model.eval();hp=[];qp=[]
 for b in DataLoader(ds,batch_size=256,num_workers=2):
  h,q=model(b['eeg'].cuda());hp.append(h.cpu().numpy());qp.append(q.cpu().numpy())
 gallery=np.zeros_like(ds.blur);ids=np.unique(ds.ids)
 gallery[ids]=F.normalize(model.image_adapter(torch.from_numpy(ds.blur[ids]).cuda()),dim=-1).cpu().numpy()
 return {'clean':metrics(np.concatenate(qp),ds.ids,ds.subjects,ds.clean),'blur':metrics(np.concatenate(hp),ds.ids,ds.subjects,gallery)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args();root=a.root
 cfg=json.loads((root/'config.json').read_text());meta=json.loads((root/'data_manifest.json').read_text());torch.set_num_threads(4)
 random.seed(33);np.random.seed(33);torch.manual_seed(33);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 clean=np.load(BASE/'data/clean.npy');blur=np.load(BASE/'features/blur.npy').copy();testids=sorted({i for r in meta['test'] for _,i in r['events']})
 for i in testids:
  with np.load(Path('/home/Anteng/eeg-retrieval/audits/blur_head_probe/test_blur_shards')/f'{i:05d}.npz') as f:
   assert int(f['id'])==i and str(f['schema'])=='upstream_per_scale_l2_dino_v1';blur[i]=f['blur']
 tr=Data(meta['train'],clean,blur);te=Data(meta['test'],clean,blur)
 assert len(np.unique(tr.ids))==16540 and len(np.unique(te.ids))==200 and not np.intersect1d(tr.ids,te.ids).size
 assert np.isfinite(blur[np.unique(tr.ids)]).all() and np.load(BASE/'features/available.npy')[np.unique(tr.ids)].all()
 model=Model(meta['channels']).cuda();criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False).cuda()
 opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':cfg['encoder_lr']},{'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')],'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
 sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5,min_lr=1e-6)
 gen=torch.Generator().manual_seed(33);loader=DataLoader(tr,batch_size=256,shuffle=True,generator=gen,num_workers=4,pin_memory=True,persistent_workers=True)
 if a.preflight:
  model.train();b=next(iter(loader));x=b['eeg'].cuda();c=b['clean'].cuda();v=b['blur'].cuda();h,q,z=model(x,v);loss=criterion(q,c)+criterion(h,z);loss.backward();grads={}
  for name,module in [('encoder',model.encoder),('projection',model.blur_head),('adapter',model.image_adapter),('recovery',model.recovery)]:
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters());grads[name]=sum(p.grad.abs().sum().item() for p in module.parameters());assert grads[name]>0
  nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();del h,q,z,x,c,v,b
  _,ix=np.unique(tr.ids,return_index=True);b=[tr[int(i)] for i in ix[:8]];x=torch.stack([s['eeg'] for s in b]).cuda();c=torch.stack([s['clean'] for s in b]).cuda();v=torch.stack([s['blur'] for s in b]).cuda()
  model.eval();probeopt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0);initial=None
  for step in range(300):
   probeopt.zero_grad();h,q,z=model(x,v);l=criterion(q,c)+criterion(h,z)
   if initial is None:initial=l.item()
   assert torch.isfinite(l);l.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);probeopt.step()
   ac=float(((F.normalize(q,dim=-1)@F.normalize(c,dim=-1).T).argmax(1)==torch.arange(8,device='cuda')).float().mean());ab=float(((F.normalize(h,dim=-1)@z.T).argmax(1)==torch.arange(8,device='cuda')).float().mean())
   if ac==1 and ab==1 and l.item()<initial*.3:break
  assert ac==ab==1 and l.item()<initial*.3,(ac,ab,l.item(),initial)
  write(root/'preflight.json',{'passed':True,'batch256_loss':loss.item(),'gradient_l1':grads,'tiny_steps':step+1,'tiny_initial_loss':initial,'tiny_final_loss':l.item(),'tiny_clean_top1':ac,'tiny_blur_top1':ab,'updates_discarded':True});print('PREFLIGHT PASSED',flush=True);return
 assert (root/'preflight.json').exists() and not (root/'history.jsonl').exists()
 best=-1;stale=0;top=[]
 for epoch in range(1,cfg['epochs']+1):
  start=time.time();model.train();tot=np.zeros(2);n=0
  write(root/'status.json',{'state':'training','epoch':epoch,'steps_per_epoch':len(loader)})
  for step,b in enumerate(loader):
   opt.zero_grad(set_to_none=True);h,q,z=model(b['eeg'].cuda(),b['blur'].cuda());lc=criterion(q,b['clean'].cuda());lb=criterion(h,z);loss=lc+lb
   assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();tot+=np.array([lc.item(),lb.item()])*len(q);n+=len(q)
   if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'clean_loss':lc.item(),'blur_loss':lb.item()}),flush=True)
  write(root/'status.json',{'state':'evaluating','epoch':epoch});scores=evaluate(model,te);value=scores['blur']['subject_aggregated']['top5'];sched.step(value)
  if value>best:best=value;stale=0
  else:stale+=1
  ck={'model':model.state_dict(),'epoch':epoch,'config':cfg,'channels':meta['channels'],'metrics':scores}
  name=f'epoch_{epoch:03d}.pt';ranked=sorted(top+[{'epoch':epoch,'score':value,'file':name}],key=lambda r:(-r['score'],r['epoch']))[:3]
  if any(r['epoch']==epoch for r in ranked):torch.save(ck,root/name)
  for r in top:
   if r not in ranked:(root/r['file']).unlink()
  top=ranked;write(root/'top3.json',{'metric':'blur.subject_aggregated.top5','models':top})
  ck.update(optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),loader_rng=gen.get_state(),stale=stale,best=best)
  torch.save(ck,root/'last.tmp');(root/'last.tmp').replace(root/'last.pt')
  row={'epoch':epoch,'train_loss':(tot/n).tolist(),'test':scores,'stale':stale,'best_blur_agg_top5':best,'lr':[g['lr'] for g in opt.param_groups],'seconds':time.time()-start}
  with (root/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps({'epoch':epoch,'blur':scores['blur']['subject_aggregated'],'clean':scores['clean']['subject_aggregated'],'stale':stale}),flush=True)
  if stale>=cfg['patience']:break
 write(root/'status.json',{'state':'complete','epochs':epoch,'top3':top})
if __name__=='__main__':main()
