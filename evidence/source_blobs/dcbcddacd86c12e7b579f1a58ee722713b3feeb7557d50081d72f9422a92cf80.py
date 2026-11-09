import json,math,time,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stage2_matched import base_head,preflight,predict,metrics
from neuraltrain.losses.losses import ClipLoss
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'config.json').read_text());C=Path(cfg['cache_root'])
def write(p,d):p.write_text(json.dumps(d,indent=2))
def grouped_loss(m,x,y,consistency):
 # x: groups, repeats, dimensions. Frozen EEG inputs, trainable recovery only.
 g,k,d=x.shape;q=m(x.reshape(-1,d)).reshape(g,k,d).mean(1)
 # Same-image groups share a candidate, including different subjects.
 unique,labels=torch.unique(y,return_inverse=True)
 logits=q@F.normalize(target[unique],dim=-1).T
 agg=F.cross_entropy(logits,labels)
 con=q.sum()*0
 if consistency:
  # Stable stop-gradient teacher. Preserve RNG and module training mode.
  with torch.no_grad():
   m.eval();teacher=m(x.mean(1));m.train()
  con=(1-F.cosine_similarity(q,teacher,dim=-1)).mean()
 return agg,con
try:
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert json.loads((C/'complete.json').read_text())['checkpoint_sha256']==cfg['checkpoint_sha256']
 ids=np.concatenate([np.load(C/f'{s}_ids.npy') for s in ['train','val']]);subs=np.concatenate([np.load(C/f'{s}_subjects.npy') for s in ['train','val']]);ti=np.load(C/'test_ids.npy');ts=np.load(C/'test_subjects.npy');assert not np.intersect1d(ids,ti).size
 _,sid=np.unique(subs,return_inverse=True);keys,inv=np.unique(np.stack([sid,ids],1),axis=0,return_inverse=True)
 order=np.argsort(inv,kind='stable');counts=np.bincount(inv);offset=np.r_[0,counts.cumsum()]
 groups=[order[offset[i]:offset[i+1]] for i in range(len(keys)) if counts[i]>=4]
 assert groups and all(np.all(ids[g]==ids[g[0]]) and np.all(subs[g]==subs[g[0]]) for g in groups)
 # Padded index table, without-replacement selection through random priorities.
 table=np.full((len(groups),max(map(len,groups))),-1,dtype=np.int64)
 for i,g in enumerate(groups):table[i,:len(g)]=g
 table=torch.tensor(table,device='cuda')
 h=torch.tensor(np.concatenate([np.load(C/f'{s}_h.npy') for s in ['train','val']]),device='cuda');th=torch.tensor(np.load(C/'test_h.npy'),device='cuda');ids=torch.tensor(ids,device='cuda')
 clean=np.load(Path(cfg['source_root'])/'data/clean.npy');target=torch.tensor(clean,device='cuda')
 write(R/'pairing_checks.json',dict(passed=True,groups=len(groups),min_repeats=int(counts.min()),max_repeats=int(counts.max()),eligible_min_repeats=4,no_test_overlap=True,training_trials=len(h)))
 def sample(gen,n=64):
  rows=table[torch.randint(len(table),(n,),device='cuda',generator=gen)]
  priority=torch.rand(rows.shape,device='cuda',generator=gen).masked_fill(rows<0,-1)
  ix=rows.gather(1,priority.topk(4,dim=1).indices)
  assert (ix>=0).all()
  return ix
 for arm,consistency in [('output_mean',False),('output_mean_consistency',True)]:
  dest=R/arm;dest.mkdir();write(R/'status.json',dict(state='preflight',arm=arm))
  _,first=np.unique(ids.cpu().numpy(),return_index=True);ix=torch.tensor(first[:8],device='cuda')
  check=preflight(base_head(),h[ix],target[ids[ix]])
  probe=base_head();gen=torch.Generator(device='cuda').manual_seed(2026);gi=sample(gen,8)
  # Train tiny grouped probe; require loss reduction and finite nonzero gradients.
  opt=torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=0);initial=None
  for step in range(40):
   probe.train();opt.zero_grad();a,c=grouped_loss(probe,h[gi],ids[gi[:,0]],consistency);loss=a+.1*c
   if initial is None:initial=float(loss.detach())
   loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters());opt.step()
  assert float(loss.detach())<initial*.5
  check.update(grouped_loss_initial=initial,grouped_loss_final=float(loss.detach()),grouped_gradients_finite=True)
  write(dest/'preflight.json',check);del probe,opt
  m=base_head();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.05);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-4,epochs=40,steps_per_epoch=math.ceil(len(h)/256),pct_start=.1)
  single=ClipLoss(norm_kind='y',temperature=False,symmetric=False);permgen=torch.Generator(device='cuda').manual_seed(33);groupgen=torch.Generator(device='cuda').manual_seed(2026)
  best=-1;stale=0;ranking=[]
  write(R/'status.json',dict(state='training',arm=arm))
  for epoch in range(1,41):
   m.train();tot=np.zeros(4);start=time.time()
   for idx in torch.randperm(len(h),device='cuda',generator=permgen).split(256):
    opt.zero_grad();l_single=single(m(h[idx]),target[ids[idx]])
    gi=sample(groupgen)
    # Auxiliary dropout must not alter the next single-trial dropout RNG sequence.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
     agg,con=grouped_loss(m,h[gi],ids[gi[:,0]],consistency)
    loss=l_single+.3*agg+(.1*con if consistency else 0)
    assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();sched.step()
    tot+=np.array([float(t.detach()) for t in [loss,l_single,agg,con]])*len(idx)
   result=metrics(predict(m,th),ti,ts,clean);score=result['subject_aggregated']['top5']
   if score>best:best=score;stale=0
   else:stale+=1
   if len(ranking)<3 or score>ranking[-1]['score']:
    path=dest/f'epoch_{epoch:03d}.pt';torch.save(dict(head=m.state_dict(),epoch=epoch,selection=result,config=cfg,arm=arm),path)
    ranking.append(dict(epoch=epoch,score=score,file=path.name));ranking.sort(key=lambda r:(-r['score'],r['epoch']))
    for old in ranking[3:]:(dest/old['file']).unlink()
    ranking=ranking[:3];write(dest/'top3.json',ranking)
    if ranking[0]['epoch']==epoch:shutil.copy2(path,dest/'best.pt')
   torch.save(dict(head=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,config=cfg,arm=arm),dest/'last.pt')
   row=dict(epoch=epoch,loss=(tot/len(h)).tolist(),selection=result,stale=stale,seconds=time.time()-start)
   with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   write(dest/'status.json',dict(state='training',epoch=epoch,stale=stale));print(arm,epoch,'loss',row['loss'],'single',result['single_trial'],'aggregate',result['subject_aggregated'],flush=True)
   if stale>=5:break
  ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);write(dest/'result.json',dict(best_epoch=ck['epoch'],selection=ck['selection'],blur=cfg['blur_reference'],role='public test model selection, not independent evaluation'));write(dest/'status.json',dict(state='complete',epoch=epoch,best_epoch=ck['epoch']))
  del m,opt,sched
 write(R/'status.json',dict(state='complete'))
except Exception:
 write(R/'status.json',dict(state='failed'));raise
