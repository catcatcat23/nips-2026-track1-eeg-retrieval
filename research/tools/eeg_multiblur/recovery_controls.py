"""Independent v45 controls: learned linear skip, dropout, narrower residual MLP."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import no_ridge_recovery as engine
from neuraltrain.losses.losses import ClipLoss

class NarrowBlock(nn.Module):
 def __init__(self):
  super().__init__();self.alpha=nn.Parameter(torch.tensor(.1));self.net=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,1536),nn.GELU(),nn.Dropout(.1),nn.Linear(1536,1536))
 def forward(self,x):return x+self.alpha*self.net(x)

class ControlHead(engine.NoRidge):
 def __init__(self,variant):
  super().__init__();self.variant=variant
  if variant=='linear_skip':
   self.skip=nn.Linear(1536,1536);nn.init.zeros_(self.skip.weight);nn.init.zeros_(self.skip.bias)
  elif variant=='dropout02':
   self.delta[1].net[3].p=.2;self.delta[2].net[3].p=.2
  elif variant=='width1536':
   self.delta[1]=NarrowBlock();self.delta[2]=NarrowBlock()
  else:raise ValueError(variant)
 def forward(self,h):
  residual=.1*self.delta(h)
  return self.skip(h)+residual if self.variant=='linear_skip' else residual

def make(cfg):
 torch.manual_seed(33);model=ControlHead(cfg['variant']).cuda()
 initial=torch.load(cfg['v39_initial_reference'],map_location='cpu',weights_only=True)
 for k,v in model.state_dict().items():
  if k.startswith('skip.'):assert torch.count_nonzero(v)==0;continue
  if cfg['variant']=='width1536' and k.startswith(('delta.1.','delta.2.')):continue
  assert torch.equal(v.cpu(),initial[k]),k
 return model

def preflight(cfg,out,h,ids,target):
 _,ix=np.unique(ids.cpu().numpy(),return_index=True);ix=torch.tensor(ix[:8],device='cuda');x=h[ix];y=target[ids[ix]]
 model=make(cfg).eval();criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False);assert torch.count_nonzero(model(x))==0
 count=sum(p.numel() for p in model.parameters());opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0);initial=float(criterion(model(x),y).detach())
 for step in range(200):
  opt.zero_grad();loss=criterion(model(x),y);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  with torch.no_grad():acc=float(((F.normalize(model(x),dim=-1)@F.normalize(y,dim=-1).T).argmax(1)==torch.arange(8,device='cuda')).float().mean())
  if step>=2 and acc==1 and float(loss.detach())<initial*.3:break
 assert acc==1 and float(loss.detach())<initial*.3
 model.train();opt.zero_grad();criterion(model(x),y).backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters())
 report=dict(passed=True,variant=cfg['variant'],parameters=count,initial_prediction_matches_v45_zero=True,unchanged_parameters_match_v45=True,initial_loss=initial,final_loss=float(loss.detach()),steps=step+1,tiny_top1=acc,updates_discarded=True)
 if cfg['variant']=='linear_skip':
  report['skip_gradient_l1']=sum(float(p.grad.abs().sum()) for p in model.skip.parameters());assert report['skip_gradient_l1']>0
 engine.write(out/'preflight.json',report);print('PREFLIGHT PASSED',json.dumps(report),flush=True)

@torch.no_grad()
def diagnostics(cfg):
 out=Path(cfg['run_root']);model=make(cfg).eval();ck=torch.load(out/'B/best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['head']);cache=Path(cfg['cache_root']);clean=np.load(Path(cfg['source_root'])/'data/clean.npy');report={'epoch':ck['epoch'],'variant':cfg['variant'],'splits':{}}
 for split in ['train','test']:
  h=np.load(cache/f'{split}_h.npy',mmap_mode='r');x=torch.tensor(h[:2048].copy(),device='cuda');r=.1*model.delta(x);q=model(x)
  stats={'norm_sample_count':len(x),'residual_norm_median':float(r.norm(dim=-1).median()),'output_norm_median':float(q.norm(dim=-1).median())}
  if cfg['variant']=='linear_skip':
   s=model.skip(x);stats.update(skip_norm_median=float(s.norm(dim=-1).median()),skip_residual_cosine_mean=float(F.cosine_similarity(s,r).mean()))
  report['splits'][split]=stats
 if cfg['variant']=='linear_skip':
  h=torch.from_numpy(np.load(cache/'test_h.npy')).cuda();ids=np.load(cache/'test_ids.npy');subjects=np.load(cache/'test_subjects.npy')
  for name,fn in [('skip_only',lambda x:model.skip(x)),('residual_only',lambda x:.1*model.delta(x))]:
   pred=np.concatenate([fn(b).cpu().numpy() for b in h.split(1024)]);report[name]=engine.metrics(pred,ids,subjects,clean)
 engine.write(out/'B/branch_diagnostics.json',report)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text())
 engine.make=make;engine.preflight=preflight
 engine.main()
 out=Path(cfg['run_root']);(out/'status.txt').write_text('BRANCH_DIAGNOSTICS\n');diagnostics(cfg);(out/'status.txt').write_text('COMPLETED\n')
if __name__=='__main__':main()
