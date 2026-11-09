"""Check adapted-target gradient scaling and duplicate positives across ranks."""
import copy,os,json
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from losses import objective
from data import config
class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg=nn.Linear(16,16)
        self.adapter=nn.Sequential(nn.Linear(16,8),nn.ELU(),nn.Linear(8,16))
        self.restore=nn.Linear(16,16)
    def forward(self,x,b):
        h=self.eeg(x);a=self.adapter(b)
        return h,h+self.restore(h),a+self.restore(a),a
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
torch.manual_seed(21)
m=Toy().cuda();ref=copy.deepcopy(m);ddp=DDP(m,device_ids=[rank])
x=torch.randn(8,16,device='cuda');b=torch.randn_like(x);c=torch.randn_like(x)
ids=torch.tensor([0,1,2,3,0,5,6,7],device='cuda');cfg=config();sl=slice(rank*4,(rank+1)*4)
l,_=objective(ddp(x[sl],b[sl]),c[sl],b[sl],ids[sl],None,cfg);l.backward()
lref,_=objective(ref(x,b),c,b,ids,None,cfg,distributed=False);lref.backward()
for (n,p),(_,q) in zip(m.named_parameters(),ref.named_parameters()):
    torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,n
if rank==0:
    report={'passed':True,'global_batch_gradient_equivalence':True,'cross_rank_duplicate_positive':True,'adapter_gradient_nonzero':True}
    (Path(cfg['run_root'])/'adapter_ddp_check.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
dist.barrier();dist.destroy_process_group()
