"""Higher-capacity EEG-only recovery: learned linear skip + gated deep residual."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import no_ridge_recovery as engine
from recovery_controls import ControlHead
from recovery_ablation import ResidualBlock
from neuraltrain.losses.losses import ClipLoss

class GatedDeepHead(ControlHead):
 def __init__(self):
  super().__init__('linear_skip')
  extra=[ResidualBlock(),ResidualBlock()]
  for block in extra:
   nn.init.zeros_(block.net[-1].weight);nn.init.zeros_(block.net[-1].bias)
  blocks=list(self.delta.children());self.delta=nn.Sequential(*blocks[:3],*extra,*blocks[3:])
  self.gate=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,128),nn.GELU(),nn.Linear(128,1),nn.Sigmoid())
  nn.init.zeros_(self.gate[3].weight);nn.init.constant_(self.gate[3].bias,math.log(.1/.9))
 def forward(self,h):return self.skip(h)+self.gate(h)*self.delta(h)

def make(cfg):
 torch.manual_seed(33);model=GatedDeepHead().cuda();initial=torch.load(cfg['v39_initial_reference'],map_location='cpu',weights_only=True)
 for k,v in initial.items():
  if not k.startswith('delta.'):continue
  parts=k.split('.');index=int(parts[1]);parts[1]=str(index+2 if index>=3 else index)
  assert torch.equal(model.state_dict()['.'.join(parts)].cpu(),v),k
 return model

def preflight(cfg,out,h,ids,target):
 _,ix=np.unique(ids.cpu().numpy(),return_index=True);ix=torch.tensor(ix[:8],device='cuda');x=h[ix];y=target[ids[ix]];model=make(cfg).eval()
 with torch.no_grad():
  torch.testing.assert_close(model.gate(x),torch.full((8,1),.1,device='cuda'))
  assert torch.count_nonzero(model(x))==0
 criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0);first=float(criterion(model(x),y).detach());params=sum(p.numel() for p in model.parameters())
 for step in range(100):
  opt.zero_grad();loss=criterion(model(x),y);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  with torch.no_grad():acc=float(((F.normalize(model(x),dim=-1)@F.normalize(y,dim=-1).T).argmax(1)==torch.arange(8,device='cuda')).float().mean())
  if step>=4 and acc==1 and float(loss.detach())<first*.3:break
 assert acc==1 and float(loss.detach())<first*.3
 model.train();opt.zero_grad();criterion(model(x),y).backward();grads={}
 for name,module in [('linear',model.skip),('gate',model.gate),('residual',model.delta),('extra_block1',model.delta[3]),('extra_block2',model.delta[4])]:
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters());grads[name]=sum(float(p.grad.abs().sum()) for p in module.parameters());assert grads[name]>0
 with torch.no_grad():
  model.eval();q=model(x);perm=torch.tensor([3,0,7,2,4,1,6,5],device='cuda');torch.testing.assert_close(model(x[perm]),q[perm],rtol=1e-4,atol=1e-4)
 report=dict(passed=True,parameters=params,initial_gate=.1,initial_output_matches_v45=True,initial_loss=first,final_loss=float(loss.detach()),steps=step+1,tiny_top1=acc,gradient_l1=grads,batch_invariance=True,updates_discarded=True)
 engine.write(out/'preflight.json',report);print('PREFLIGHT PASSED',json.dumps(report),flush=True)

@torch.no_grad()
def diagnose(cfg):
 out=Path(cfg['run_root']);cache=Path(cfg['cache_root']);model=make(cfg).eval();ck=torch.load(out/'B/best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['head']);report={'epoch':ck['epoch'],'splits':{}}
 for split in ['train','test']:
  h=torch.tensor(np.load(cache/f'{split}_h.npy',mmap_mode='r')[:2048].copy(),device='cuda');g=model.gate(h);s=model.skip(h);r=g*model.delta(h)
  report['splits'][split]=dict(samples=len(h),gate_quantiles=torch.quantile(g.flatten(),torch.tensor([0.,.25,.5,.75,1.],device='cuda')).tolist(),skip_norm_median=float(s.norm(dim=-1).median()),residual_norm_median=float(r.norm(dim=-1).median()),output_norm_median=float((s+r).norm(dim=-1).median()))
 h=torch.from_numpy(np.load(cache/'test_h.npy')).cuda();ids=np.load(cache/'test_ids.npy');subjects=np.load(cache/'test_subjects.npy');clean=np.load(Path(cfg['source_root'])/'data/clean.npy')
 for name,fn in [('skip_only',lambda x:model.skip(x)),('residual_only',lambda x:model.gate(x)*model.delta(x)),('gate_fixed_01',lambda x:model.skip(x)+.1*model.delta(x))]:
  pred=np.concatenate([fn(b).cpu().numpy() for b in h.split(1024)]);report[name]=engine.metrics(pred,ids,subjects,clean)
 engine.write(out/'B/branch_diagnostics.json',report)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['run_root'])
 engine.make=make;engine.preflight=preflight;engine.main()
 (out/'status.txt').write_text('BRANCH_DIAGNOSTICS\n');diagnose(cfg);(out/'status.txt').write_text('COMPLETED\n')
if __name__=='__main__':main()
