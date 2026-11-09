import sys,json,math,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
S=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v57_stage2_epoch18')
sys.path.insert(0,str(S/'source'))
from stage2_matched import preflight,base_head
from neuraltrain.losses.losses import ClipLoss
from v33_recovery_stage2 import metrics
R=Path(sys.argv[1]);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=json.loads((S/'config.json').read_text());src=Path(cfg['source_root']);cache=S/'cache'
def write(n,x):(R/n).write_text(json.dumps(x,indent=2))
def hits(q,y):
 score=F.normalize(q,dim=-1)@F.normalize(y,dim=-1).T
 top=score.topk(5,dim=1).indices;labels=torch.arange(len(q),device='cuda')[:,None]
 return dict(top1=float((top[:,:1]==labels).float().mean()),top5=float((top==labels).any(1).float().mean()),count=len(q))
@torch.no_grad()
def pred(m,x):
 m.eval();return torch.cat([m(b) for b in x.split(512)])
try:
 write('status.json',dict(state='preparing'))
 ids=np.unique(np.concatenate([np.load(cache/f'{s}_ids.npy') for s in ['train','val']]));testids=np.unique(np.load(cache/'test_ids.npy'));assert len(testids)==200 and not np.intersect1d(ids,testids).size
 rng=np.random.default_rng(33);perm=rng.permutation(ids);valid=perm[:1000];train=perm[1000:]
 allids=np.concatenate([train,valid,testids]);n=len(train);nv=len(valid)
 paths=json.loads((src/'data/manifest.json').read_text())['paths'];blur=np.load(src/'features/blur.npy',mmap_mode='r');avail=np.load(src/'features/available.npy')
 vals=[]
 for i in allids:
  if i in testids:
   p=Path('/home/Anteng/eeg-retrieval/audits/blur_head_probe/test_blur_shards')/f'{i:05d}.npz'
   with np.load(p) as f:
    assert int(f['id'])==i and str(f['filepath'])==paths[i];vals.append(f['blur'].copy())
  else:
   assert avail[i];vals.append(blur[i].copy())
 x=torch.tensor(np.stack(vals),device='cuda');assert x.shape==(len(allids),1536) and torch.isfinite(x).all()
 clean=np.load(src/'data/clean.npy');y=torch.tensor(clean[allids],device='cuda')
 ck=torch.load(S/'source_global.pt',map_location='cpu',weights_only=False);assert ck['epoch']==18
 adapter=nn.Sequential(nn.Linear(1536,768),nn.ELU(),nn.Dropout(.2),nn.Linear(768,1536)).cuda().eval()
 adapter.load_state_dict({k[len('image_adapter.'):]:v for k,v in ck['model'].items() if k.startswith('image_adapter.')})
 with torch.no_grad():a=F.normalize(pred(adapter,x),dim=-1)
 del ck,adapter
 write('protocol.json',dict(train_images=n,validation_images=nv,test_images=200,train_ids=train.tolist(),validation_ids=valid.tolist(),test_ids=testids.tolist(),selection='held-out training-image 1000-way top5; official test only after selection',epochs=100,batch=256,lr=1e-4,patience=15,source_epoch=18,head='v45 NoRidge',adapter='frozen eval; no dropout',note='Unique-image training; diagnostic, not matched EEG trial-weighted training'))
 result={'raw_blur_direct_clean':hits(x[-200:],y[-200:])}
 # Existing EEG-trained head: compare averaging before vs after nonlinear recovery.
 head=base_head();head.load_state_dict(torch.load(S/'v45/best.pt',map_location='cpu',weights_only=False)['head']);head.eval()
 h=np.load(cache/'test_h.npy',mmap_mode='r');ti=np.load(cache/'test_ids.npy');sub=np.load(cache/'test_subjects.npy')
 _,si=np.unique(sub,return_inverse=True);keys,inv=np.unique(np.stack([si,ti],1),axis=0,return_inverse=True)
 means=np.zeros((len(keys),1536),dtype=np.float64);np.add.at(means,inv,h);means/=np.bincount(inv)[:,None]
 q=pred(head,torch.tensor(means,dtype=torch.float32,device='cuda')).cpu().numpy()
 result['average_h_then_restore']=metrics(q,keys[:,1],keys[:,0],clean)
 result['restore_then_average']=json.loads((S/'v45/result.json').read_text())['selection']
 result['eeg_head_on_true_adapter_diagnostic_only']=hits(pred(head,a[-200:]),y[-200:])
 del head
 write('partial.json',result)
 for name,z in [('raw_blur',x),('adapted_blur',a)]:
  write('status.json',dict(state='image_recovery_training',arm=name))
  check=preflight(base_head(),z[:8],y[:8]);write(name+'_preflight.json',check)
  m=base_head();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.05)
  sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-4,epochs=100,steps_per_epoch=math.ceil(n/256),pct_start=.1)
  crit=ClipLoss(norm_kind='y',temperature=False,symmetric=False);gen=torch.Generator(device='cuda').manual_seed(33);best=-1;stale=0
  for ep in range(1,101):
   m.train();total=0
   for ix in torch.randperm(n,generator=gen,device='cuda').split(256):
    opt.zero_grad();loss=crit(m(z[ix]),y[ix]);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();sched.step();total+=float(loss.detach())*len(ix)
   v=hits(pred(m,z[n:n+nv]),y[n:n+nv]);score=v['top5']
   if score>best:best=score;stale=0;torch.save(dict(head=m.state_dict(),epoch=ep,validation=v),R/(name+'_best.pt'))
   else:stale+=1
   row=dict(arm=name,epoch=ep,loss=total/n,validation=v,stale=stale)
   with (R/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
   if stale>=15:break
  ck=torch.load(R/(name+'_best.pt'),map_location='cpu',weights_only=False);m.load_state_dict(ck['head']);result[name]=dict(best_epoch=ck['epoch'],validation=ck['validation'],test_200=hits(pred(m,z[-200:]),y[-200:]),train_fixed_200=hits(pred(m,z[:200]),y[:200]))
  write('partial.json',result);del m,opt,sched
 write('result.json',result);write('status.json',dict(state='complete'))
except Exception:
 write('status.json',dict(state='failed'));raise
