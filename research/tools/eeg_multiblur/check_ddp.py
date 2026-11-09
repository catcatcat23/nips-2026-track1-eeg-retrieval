"""Fast numerical regression test: two-rank gradients == global-batch gradients."""
import copy,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import Model
from losses import objective
from data import config

def main():
    rank=int(os.environ['RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');torch.manual_seed(51)
    model=Model(63,200).cuda().eval();ref=copy.deepcopy(model);ddp=DDP(model,device_ids=[rank])
    x=torch.randn(8,63,200,device='cuda');b=torch.randn(8,1536,device='cuda');v=torch.randn(8,9,1536,device='cuda');c=torch.randn(8,1536,device='cuda')
    ids=torch.tensor([0,1,2,3,0,4,5,6],device='cuda');w=torch.ones(9,device='cuda')/9;s=slice(rank*4,(rank+1)*4)
    loss,_=objective(ddp(x[s],b[s],v[s]),c[s],b[s],ids[s],w,config());loss.backward()
    other,_=objective(ref(x,b,v),c,b,ids,w,config(),distributed=False);other.backward()
    errors={}
    for (name,p),(_,r) in zip(model.named_parameters(),ref.named_parameters()):
        assert p.grad is not None and r.grad is not None,name
        err=(p.grad-r.grad).abs().max().item();scale=r.grad.abs().max().item()
        assert err<2e-5+scale*3e-3,(name,err,scale)
        errors[name]=err
    if rank==0:
        report={'passed':True,'world_size':2,'cross_rank_duplicate_ids':True,'max_abs_gradient_error':max(errors.values()),'loss':loss.item()}
        print(report,flush=True);(Path(config()['run_root'])/'ddp_gradient_check.json').write_text(json.dumps(report,indent=2))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
