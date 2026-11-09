"""Matched v39-B ablation: output only 0.1*delta(h), no Ridge branch."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ridge_recovery import RidgeResidual,loss_value
from recovery_ablation import ResidualBlock,metrics,predictions
from neuraltrain.losses.losses import ClipLoss

def write(p,x):p.write_text(json.dumps(x,indent=2))
class NoRidge(nn.Module):
 def __init__(self):
  super().__init__();self.delta=nn.Sequential(nn.LayerNorm(1536),ResidualBlock(),ResidualBlock(),nn.LayerNorm(1536),nn.Linear(1536,1536))
  nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
 def forward(self,h):return .1*self.delta(h)

def make(cfg):
 torch.manual_seed(33);model=NoRidge().cuda()
 initial=torch.load(cfg['v39_initial_reference'],map_location='cpu',weights_only=True)
 expected={k:v for k,v in initial.items() if k.startswith('delta.')}
 assert set(expected)==set(model.state_dict())
 for k,v in model.state_dict().items():assert torch.equal(v.cpu(),expected[k]),k
 return model

def preflight(cfg,out,h,ids,target):
 _,ix=np.unique(ids.cpu().numpy(),return_index=True);ix=torch.tensor(ix[:8],device='cuda');x=h[ix];y=target[ids[ix]];model=make(cfg).eval()
 original_state=torch.load(cfg['v39_initial_reference'],map_location='cpu',weights_only=True);ridge={k:original_state[k] for k in ['mean','scale','weight','bias']};original=RidgeResidual(ridge).cuda().eval();original.load_state_dict(original_state)
 with torch.no_grad():
  assert torch.count_nonzero(model(x))==0
  torch.testing.assert_close(model(x),original(x)-(((x-original.mean)/original.scale)@original.weight+original.bias),rtol=0,atol=0)
 del original
 criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0);initial=float(criterion(model(x),y).detach())
 for step in range(200):
  opt.zero_grad();loss,_,_=loss_value(model(x),y,criterion,0.);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  with torch.no_grad():acc=float(((F.normalize(model(x),dim=-1)@F.normalize(y,dim=-1).T).argmax(1)==torch.arange(8,device='cuda')).float().mean())
  if step>=2 and acc==1 and float(loss.detach())<initial*.3:break
 assert acc==1 and float(loss.detach())<initial*.3
 model.train();opt.zero_grad();loss_value(model(x),y,criterion,0.)[0].backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters())
 write(out/'preflight.json',dict(passed=True,only_change='remove fixed ridge addition',delta_initialization_exactly_matches_v39=True,initial_prediction='all zeros, expected after removing ridge with zero-init final delta layer',initial_loss=initial,final_loss=float(loss.detach()),steps=step+1,tiny_top1=acc,updates_discarded=True))
 print('PREFLIGHT PASSED',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['run_root']);cache=Path(cfg['cache_root']);src=Path(cfg['source_root'])
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;np.random.seed(33)
 assert json.loads((cache/'complete.json').read_text())['checkpoint_sha256']==cfg['checkpoint_sha256']
 for s in ['train','val','test']:
  for k in ['ids','subjects']:assert np.array_equal(np.load(cache/f'{s}_{k}.npy'),np.load(src/'data'/f'{s}_{k}.npy'))
 ids=np.concatenate([np.load(cache/f'{s}_ids.npy') for s in ['train','val']]);testids=np.load(cache/'test_ids.npy');subjects=np.load(cache/'test_subjects.npy');assert not np.intersect1d(ids,testids).size
 clean=np.load(src/'data/clean.npy');target=torch.tensor(clean,device='cuda');trainids=torch.tensor(ids,device='cuda');h=torch.from_numpy(np.concatenate([np.load(cache/f'{s}_h.npy') for s in ['train','val']])).cuda();testh=torch.from_numpy(np.load(cache/'test_h.npy')).cuda()
 preflight(cfg,out,h,trainids,target)
 dest=out/'B';dest.mkdir();model=make(cfg);torch.save(model.state_dict(),dest/'initial_head.pt');opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);steps=math.ceil(len(h)/cfg['batch_size']);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg['lr'],epochs=cfg['epochs'],steps_per_epoch=steps,pct_start=.1)
 criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False);score=metrics(predictions(model,testh),testids,subjects,clean);write(dest/'epoch0.json',dict(selection=score,note='all-zero ties: diagnostic only, not learned retrieval'))
 best=score['single_trial']['top5'];stale=0;bestepoch=0
 def save(epoch,selection):torch.save(dict(head=model.state_dict(),epoch=epoch,selection=selection,config=cfg,optimizer=opt.state_dict(),scheduler=sched.state_dict()),dest/'best.pt')
 save(0,score);gen=torch.Generator(device='cuda').manual_seed(33);(out/'status.txt').write_text('TRAINING\n')
 for epoch in range(1,cfg['epochs']+1):
  start=time.time();model.train();tot=np.zeros(3);count=0
  for step,idx in enumerate(torch.randperm(len(h),generator=gen,device='cuda').split(cfg['batch_size'])):
   opt.zero_grad(set_to_none=True);loss,ce,co=loss_value(model(h[idx]),target[trainids[idx]],criterion,0.);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sched.step();tot+=np.array([float(loss.detach()),float(ce.detach()),float(co.detach())])*len(idx);count+=len(idx)
   if step%500==0:print('TRAIN',epoch,step,steps,'loss',float(loss.detach()),flush=True)
  selection=metrics(predictions(model,testh),testids,subjects,clean);diag=metrics(predictions(model,h[:2048]),ids[:2048],np.zeros(2048,dtype=int),clean);value=selection['single_trial']['top5']
  if value>best:best=value;stale=0;bestepoch=epoch;save(epoch,selection)
  else:stale+=1
  row=dict(epoch=epoch,loss=(tot/count).tolist(),selection=selection,train_diagnostic=diag,best_epoch=bestepoch,stale=stale,seconds=time.time()-start)
  with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  write(dest/'status.json',dict(state='training',epoch=epoch,best_epoch=bestepoch,stale=stale));print(json.dumps(row),flush=True)
  if stale>=cfg['patience']:break
 ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);blur=json.loads(Path(cfg['blur_reference']).read_text());assert blur['checkpoint_sha256']==cfg['checkpoint_sha256']
 write(dest/'result.json',dict(best_epoch=ck['epoch'],selection=ck['selection'],blur=blur['metrics']['eeg_to_adapted_blur'],blur_provenance='reused same frozen v33 EEG, projection, adapter and cache; head-only ablation',role='public test selection, not independent test'))
 write(dest/'status.json',dict(state='complete',epochs=epoch,best_epoch=bestepoch));(out/'status.txt').write_text('COMPLETED\n')
if __name__=='__main__':main()
