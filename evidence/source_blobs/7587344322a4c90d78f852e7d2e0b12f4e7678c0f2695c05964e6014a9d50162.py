import json,math,time,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stage2_matched import base_head,predict
from v33_recovery_stage2 import metrics
from neuraltrain.losses.losses import ClipLoss
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'config.json').read_text());C=Path(cfg['cache_root'])
def write(p,x):p.write_text(json.dumps(x,indent=2))
try:
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 write(R/'status.json',{'state':'preparing'})
 assert json.loads((C/'complete.json').read_text())['checkpoint_sha256']==cfg['checkpoint_sha256']
 p=Path(cfg['teacher_checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['teacher_sha256']
 state=torch.load(p,map_location='cpu',weights_only=False)['head']
 def make():
  m=base_head();m.load_state_dict(state);return m
 teacher=make().eval().requires_grad_(False)
 ids=np.concatenate([np.load(C/f'{s}_ids.npy') for s in ['train','val']]);subs=np.concatenate([np.load(C/f'{s}_subjects.npy') for s in ['train','val']]);ti=np.load(C/'test_ids.npy');ts=np.load(C/'test_subjects.npy');assert not np.intersect1d(ids,ti).size
 h_np=np.concatenate([np.load(C/f'{s}_h.npy') for s in ['train','val']]);h=torch.tensor(h_np,device='cuda');th=torch.tensor(np.load(C/'test_h.npy'),device='cuda');clean=np.load(Path(cfg['source_root'])/'data/clean.npy');target=torch.tensor(clean,device='cuda')
 _,sid=np.unique(subs,return_inverse=True);keys,inv=np.unique(np.stack([sid,ids],1),axis=0,return_inverse=True);counts=np.bincount(inv)
 means=np.zeros((len(keys),1536),dtype=np.float64);np.add.at(means,inv,h_np);means=(means/counts[:,None]).astype('float32')
 # Fixed 200 training images, same subject-image groups and same gallery for every repeat count.
 rng=np.random.default_rng(33);chosen=rng.choice(np.unique(ids),200,replace=False);chosen.sort();order=np.argsort(inv,kind='stable');offset=np.r_[0,counts.cumsum()];groups=[]
 for j in np.where(np.isin(keys[:,1],chosen))[0]:
  g=order[offset[j]:offset[j+1]];assert len(g)>=4;groups.append(rng.permutation(g))
 groupids=np.array([ids[g[0]] for g in groups]);group_subs=np.array([subs[g[0]] for g in groups])
 curve={}
 for k in [1,2,4,'all']:
  x=np.stack([h_np[g if k=='all' else g[:k]].mean(0) for g in groups]);q=predict(teacher,torch.tensor(x,device='cuda'))
  curve[str(k)]=metrics(q,groupids,group_subs,clean)['single_trial']
 gate=curve['4']['top1']>curve['1']['top1'] and curve['4']['top5']>curve['1']['top5']
 write(R/'teacher_probe.json',dict(curve=curve,passed=gate,images=chosen.tolist(),groups=len(groups),note='Training images already seen by teacher; diagnostic of denoising, NOT held-out generalization; same 200 candidates and group samples across k'))
 print('TEACHER_PROBE',curve,'passed',gate,flush=True)
 if not gate:
  write(R/'status.json',dict(state='blocked_by_teacher_probe'));raise SystemExit(0)
 # One fixed teacher prediction per subject-image, using training trials only.
 tq=torch.tensor(predict(teacher,torch.tensor(means,device='cuda')),device='cuda');tq=F.normalize(tq,dim=-1).detach();del teacher,means,h_np
 ids=torch.tensor(ids,device='cuda');inv=torch.tensor(inv,device='cuda');crit=ClipLoss(norm_kind='y',temperature=False,symmetric=False)
 def kd(q,idx):
  unique=torch.unique(ids[idx]);gallery=F.normalize(target[unique],dim=-1)
  student=F.normalize(q,dim=-1)@gallery.T/.07;teacher_logits=tq[inv[idx]]@gallery.T/.07
  return F.kl_div(F.log_softmax(student,dim=-1),F.softmax(teacher_logits,dim=-1),reduction='batchmean')
 # Calibrate weighted gradient norm to 20% of clean gradient on 8 fixed training batches.
 m=make();m.train();gen=torch.Generator(device='cuda').manual_seed(101);norms=[]
 for idx in torch.randperm(len(h),device='cuda',generator=gen)[:2048].split(256):
  q=m(h[idx]);a=crit(q,target[ids[idx]]);b=kd(q,idx)
  ga=torch.autograd.grad(a,tuple(m.parameters()),retain_graph=True);gb=torch.autograd.grad(b,tuple(m.parameters()))
  na=float(torch.sqrt(sum((g.float()**2).sum() for g in ga)));nb=float(torch.sqrt(sum((g.float()**2).sum() for g in gb)));assert np.isfinite([na,nb]).all() and nb>0;norms.append([na,nb])
 weight=.2*np.mean([v[0] for v in norms])/np.mean([v[1] for v in norms]);assert np.isfinite(weight) and weight>0
 write(R/'calibration.json',dict(weight=weight,target_gradient_ratio=.2,batches=norms,temperature=.07));del m
 # Throwaway actual combined-loss backward/update test, never used as initialization.
 probe=make();opt=torch.optim.AdamW(probe.parameters(),lr=1e-4);idx=torch.arange(256,device='cuda');probe.train();q=probe(h[idx]);loss=crit(q,target[ids[idx]])+weight*kd(q,idx);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters());opt.step();assert all(torch.isfinite(p).all() for p in probe.parameters());write(R/'preflight.json',dict(passed=True,teacher_frozen=True,finite_combined_gradients=True,source_cache_verified=True));del probe,opt
 for arm in ['continue_clean','frozen_teacher_kd']:
  dest=R/arm;dest.mkdir();m=make();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.05);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-4,epochs=20,steps_per_epoch=math.ceil(len(h)/256),pct_start=.1);gen=torch.Generator(device='cuda').manual_seed(33)
  initial=metrics(predict(m,th),ti,ts,clean);best=initial['subject_aggregated']['top5'];stale=0;ranking=[]
  def save(epoch,score,selection):
   path=dest/f'epoch_{epoch:03d}.pt';torch.save(dict(head=m.state_dict(),epoch=epoch,selection=selection,config=cfg,kd_weight=weight,arm=arm),path);ranking.append(dict(epoch=epoch,score=score,file=path.name));ranking.sort(key=lambda x:(-x['score'],x['epoch']))
   for old in ranking[3:]:(dest/old['file']).unlink()
   del ranking[3:];write(dest/'top3.json',ranking)
   if ranking[0]['epoch']==epoch:shutil.copy2(path,dest/'best.pt')
  save(0,best,initial);write(R/'status.json',dict(state='training',arm=arm))
  for ep in range(1,21):
   m.train();tot=np.zeros(3);start=time.time()
   for idx in torch.randperm(len(h),device='cuda',generator=gen).split(256):
    opt.zero_grad();q=m(h[idx]);a=crit(q,target[ids[idx]]);b=kd(q,idx) if arm=='frozen_teacher_kd' else a*0;loss=a+weight*b;assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();sched.step();tot+=np.array([float(t.detach()) for t in [loss,a,b]])*len(idx)
   result=metrics(predict(m,th),ti,ts,clean);score=result['subject_aggregated']['top5']
   if score>best:best=score;stale=0
   else:stale+=1
   if len(ranking)<3 or score>ranking[-1]['score']:save(ep,score,result)
   row=dict(epoch=ep,loss=(tot/len(h)).tolist(),selection=result,stale=stale,seconds=time.time()-start)
   with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   torch.save(dict(head=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=ep),dest/'last.pt');write(dest/'status.json',dict(state='training',epoch=ep,stale=stale));print(arm,ep,row['loss'],result['single_trial'],result['subject_aggregated'],flush=True)
   if stale>=5:break
  ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);write(dest/'result.json',dict(best_epoch=ck['epoch'],selection=ck['selection'],blur=cfg['blur_reference'],role='public test selection, not independent test'));write(dest/'status.json',dict(state='complete',epoch=ep,best_epoch=ck['epoch']));del m,opt,sched
 write(R/'status.json',dict(state='complete'))
except Exception:
 write(R/'status.json',dict(state='failed'));raise
