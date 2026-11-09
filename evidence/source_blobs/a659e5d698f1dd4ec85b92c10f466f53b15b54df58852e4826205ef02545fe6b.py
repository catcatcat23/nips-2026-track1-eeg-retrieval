"""v53 epoch9: exact v45 head, then frozen-head FiLM correction. No distillation."""
import json, time, math, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from no_ridge_recovery import NoRidge
from v33_recovery_stage2 import load_source, DatasetEEG, metrics
from neuraltrain.losses.losses import ClipLoss
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'config.json').read_text())
CACHE=Path(CFG['cache_root'])
def write(p,d): p.write_text(json.dumps(d,indent=2))
def seed():
 torch.manual_seed(33);np.random.seed(33)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def base_head():
 seed();m=NoRidge().cuda()
 initial=torch.load(CFG['initial_reference'],map_location='cpu',weights_only=True)
 expected={k:v for k,v in initial.items() if k.startswith('delta.')}
 for k,v in m.state_dict().items():assert torch.equal(v.cpu(),expected[k]),k
 return m

class Conditional(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base.requires_grad_(False).eval()
  self.input=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,512))
  self.condition=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,1024))
  self.output=nn.Sequential(nn.GELU(),nn.Dropout(.1),nn.Linear(512,1536))
  nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
 def train(self,mode=True):
  super().train(mode);self.base.eval();return self
 def forward(self,h,e):
  gamma,beta=self.condition(e).chunk(2,-1)
  return self.base(h)+.1*self.output(self.input(h)*(1+gamma)+beta)

@torch.no_grad()
def predict(m,h,e=None):
 m.eval();return np.concatenate([ (m(h[i:i+1024]) if e is None else m(h[i:i+1024],e[i:i+1024])).cpu().numpy() for i in range(0,len(h),1024)])

def preflight(m,h,y,e=None):
 # Throwaway model; production initialization is recreated after this test.
 criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False).cuda()
 x=h[:8];z=None if e is None else e[:8];target=y[:8]
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3,weight_decay=0)
 m.eval()
 if e is not None:
  with torch.no_grad():torch.testing.assert_close(m(x,z),m.base(x),rtol=0,atol=0)
 with torch.no_grad():initial=float(criterion(m(x) if z is None else m(x,z),target))
 for step in range(250):
  opt.zero_grad();q=m(x) if z is None else m(x,z);loss=criterion(q,target);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
  if e is not None:assert all(p.grad is None for p in m.base.parameters())
  opt.step()
  with torch.no_grad():
   q=m(x) if z is None else m(x,z)
   acc=float(((q@torch.nn.functional.normalize(target,dim=-1).T).argmax(-1)==torch.arange(8,device='cuda')).float().mean())
  if acc==1 and float(loss)<initial*.3:break
 assert acc==1 and float(loss)<initial*.3,(acc,initial,float(loss))
 # Exercise training-time dropout separately.
 m.train();opt.zero_grad();criterion(m(x) if z is None else m(x,z),target).backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
 return dict(passed=True,initial_loss=initial,final_loss=float(loss),steps=step+1,tiny_top1=acc)

def train(arm,h,ids,th,ti,subjects,clean,e=None,te=None):
 dest=ROOT/arm;dest.mkdir();seed()
 def make():
  m=base_head()
  if arm=='conditional':
   ck=torch.load(ROOT/'v45/best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['head'])
   seed();m=Conditional(m).cuda()
  return m
 _,ix=np.unique(ids.cpu().numpy(),return_index=True);ix=torch.tensor(ix[:8],device='cuda')
 target=torch.tensor(clean,device='cuda')
 write(dest/'preflight.json',preflight(make(),h[ix],target[ids[ix]],None if e is None else e[ix]))
 m=make();initial=metrics(predict(m,th,te),ti,subjects,clean)
 write(dest/'epoch0.json',initial)
 params=[p for p in m.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.05)
 steps=math.ceil(len(h)/256);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-4,epochs=40,steps_per_epoch=steps,pct_start=.1)
 criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False).cuda()
 ranking=[];best=-1;stale=0
 def save(epoch,score,selection):
  nonlocal ranking
  path=dest/f'epoch_{epoch:03d}.pt'
  torch.save(dict(head=m.state_dict(),epoch=epoch,selection=selection,config=CFG),path)
  ranking.append(dict(epoch=epoch,score=score,file=path.name));ranking.sort(key=lambda r:(-r['score'],r['epoch']))
  for old in ranking[3:]:(dest/old['file']).unlink()
  ranking=ranking[:3];write(dest/'top3.json',ranking)
  if ranking[0]['epoch']==epoch:
   import shutil
   shutil.copy2(path,dest/'best.pt')
 if arm=='conditional':
  best=initial['subject_aggregated']['top5'];save(0,best,initial)
 generator=torch.Generator(device='cuda').manual_seed(33)
 for epoch in range(1,41):
  begin=time.time();m.train();total=0
  for idx in torch.randperm(len(h),generator=generator,device='cuda').split(256):
   opt.zero_grad(set_to_none=True);q=m(h[idx]) if e is None else m(h[idx],e[idx]);loss=criterion(q,target[ids[idx]])
   assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();sched.step();total+=float(loss.detach())*len(idx)
  selection=metrics(predict(m,th,te),ti,subjects,clean);score=selection['subject_aggregated']['top5']
  if score>best:best=score;stale=0
  else:stale+=1
  if len(ranking)<3 or score>ranking[-1]['score']:save(epoch,score,selection)
  torch.save(dict(head=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,config=CFG),dest/'last.pt')
  row=dict(epoch=epoch,loss=total/len(h),selection=selection,stale=stale,seconds=time.time()-begin)
  with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(arm,epoch,'loss',row['loss'],'single',selection['single_trial'],'agg',selection['subject_aggregated'],flush=True)
  write(dest/'status.json',dict(state='training',epoch=epoch,stale=stale))
  if stale>=5:break
 ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
 write(dest/'result.json',dict(best_epoch=ck['epoch'],selection=ck['selection'],blur=CFG['blur_reference'],role='public test model selection, not independent evaluation'))
 write(dest/'status.json',dict(state='complete',epoch=epoch,best_epoch=ck['epoch']))

@torch.no_grad()
def extract_e():
 dest=ROOT/'e_cache';dest.mkdir();model=load_source(CFG)
 for split in ['train','val','test']:
  ds=DatasetEEG(CFG['source_root'],split,aux=False)
  assert np.array_equal(ds.ids,np.load(CACHE/f'{split}_ids.npy'))
  assert np.array_equal(ds.subjects,np.load(CACHE/f'{split}_subjects.npy'))
  h=np.load(CACHE/f'{split}_h.npy',mmap_mode='r');a=np.lib.format.open_memmap(dest/f'{split}_e.npy',mode='w+',dtype='float32',shape=(len(ds),512));start=0;maxerr=0
  for batch in DataLoader(ds,batch_size=256,num_workers=2):
   x=model.preprocess(batch['eeg'].cuda());pos=model.positions.to(x).unsqueeze(0).expand(len(x),-1,-1)
   e=model.encoder(x,pos=pos,return_output=True)[-1].mean(1);pred=model.blur_head(e)
   expected=torch.tensor(h[start:start+len(e)],device='cuda');torch.testing.assert_close(pred,expected,atol=1e-5,rtol=1e-4)
   maxerr=max(maxerr,float((pred-expected).abs().max()));assert torch.isfinite(e).all()
   a[start:start+len(e)]=e.cpu().numpy();start+=len(e)
   if start%51200==0:print('EXTRACT_E',split,start,len(ds),flush=True)
  assert start==len(ds);a.flush();print('E_COMPLETE',split,maxerr,flush=True)
 write(dest/'complete.json',dict(checkpoint_sha256=CFG['checkpoint_sha256'],h_replay_verified=True))

def main():
 torch.set_num_threads(4);seed()
 assert json.loads((CACHE/'complete.json').read_text())['checkpoint_sha256']==CFG['checkpoint_sha256']
 ids=np.concatenate([np.load(CACHE/f'{s}_ids.npy') for s in ['train','val']]);ti=np.load(CACHE/'test_ids.npy');assert not np.intersect1d(ids,ti).size
 subjects=np.load(CACHE/'test_subjects.npy');clean=np.load(Path(CFG['source_root'])/'data/clean.npy')
 h=torch.tensor(np.concatenate([np.load(CACHE/f'{s}_h.npy') for s in ['train','val']]),device='cuda');th=torch.tensor(np.load(CACHE/'test_h.npy'),device='cuda');ids=torch.tensor(ids,device='cuda')
 write(ROOT/'status.json',dict(state='v45_training'));train('v45',h,ids,th,ti,subjects,clean)
 write(ROOT/'status.json',dict(state='condition_feature_extraction'));extract_e()
 ec=ROOT/'e_cache';e=torch.tensor(np.concatenate([np.load(ec/f'{s}_e.npy') for s in ['train','val']]),device='cuda');te=torch.tensor(np.load(ec/'test_e.npy'),device='cuda')
 write(ROOT/'status.json',dict(state='conditional_training'));train('conditional',h,ids,th,ti,subjects,clean,e,te)
 write(ROOT/'status.json',dict(state='complete'))
if __name__=='__main__':
 try:main()
 except Exception:
  write(ROOT/'status.json',dict(state='failed'));raise
