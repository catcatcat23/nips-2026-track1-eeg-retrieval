"""Real-image/EEG two-GPU gradient equivalence and 8-trial overfit gates."""
import json,os,copy
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import Model
from losses import objective
from data import config

def main():
    cfg=config();root=Path(cfg['run_root']);rank=int(os.environ['RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
    torch.manual_seed(21)
    manifest=json.loads((root/'data/manifest.json').read_text())
    shards=[np.load(p) for p in sorted((root/'smoke_features').glob('*.npz'))]
    ids=np.concatenate([s['ids'] for s in shards]);blur=np.concatenate([s['blur'] for s in shards]);local=np.concatenate([s['local'] for s in shards])
    train_ids=np.load(root/'data/train_ids.npy');rows=[int(np.flatnonzero(train_ids==i)[0]) for i in ids]
    eeg=np.load(root/'data/train_eeg.npy',mmap_mode='r')[rows];clean=np.load(root/'data/clean.npy')[ids]
    # Duplicate IDs in an additional regression case are checked below.
    arrays=[torch.as_tensor(a,device='cuda') for a in [eeg,blur,local,clean,ids]]
    weights=torch.full((9,),1/9,device='cuda')
    base=Model(*manifest['input_shape']).cuda().eval();reference=copy.deepcopy(base)
    ddp=DDP(base,device_ids=[rank]);sl=slice(rank*4,(rank+1)*4)
    x,b,v,c,i=arrays
    loss,_=objective(ddp(x[sl],b[sl],v[sl]),c[sl],b[sl],i[sl],weights,cfg);loss.backward()
    ref,_=objective(reference(x,b,v),c,b,i,weights,cfg,distributed=False);ref.backward()
    max_relative=0.
    for (name,p),(_,r) in zip(ddp.module.named_parameters(),reference.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        error=(p.grad-r.grad).abs().max().item();scale=r.grad.abs().max().item()
        assert error<2e-5+scale*3e-3,(name,error,scale)
        max_relative=max(max_relative,error/max(scale,1e-5))
    duplicate_ids=i.clone();duplicate_ids[1]=duplicate_ids[0]
    duplicate_loss,_=objective(reference(x,b,v),c,b,duplicate_ids,weights,cfg,distributed=False)
    assert torch.isfinite(duplicate_loss)
    # Real data overfit with dropout disabled; SyncBN exercised in training.
    del ddp,reference
    model=Model(*manifest['input_shape'])
    for mod in model.modules():
        if isinstance(mod,torch.nn.Dropout):mod.p=0.
        if isinstance(mod,torch.nn.MultiheadAttention):mod.dropout=0.
    model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    ddp=DDP(model,device_ids=[rank]);opt=torch.optim.AdamW(ddp.parameters(),lr=1e-3)
    losses=[]
    for step in range(150):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=ddp(x[sl],b[sl],v[sl])
        loss,parts=objective(out,c[sl],b[sl],i[sl],weights,cfg)
        loss.backward()
        if step==3:
            for prefix in ['spatial','temporal','global_head','restore','queries','image_projection']:
                assert sum(p.grad.abs().sum().item() for n,p in ddp.module.named_parameters() if n.startswith(prefix))>0,prefix
        torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(loss.item())
        if rank==0 and step%25==0:print('overfit',step,loss.item(),parts,flush=True)
    ddp.module.eval()
    with torch.no_grad():
        z=ddp.module(x);scores=torch.nn.functional.normalize(z,dim=-1)@torch.nn.functional.normalize(c,dim=-1).T
        accuracy=(scores.argmax(1)==torch.arange(8,device='cuda')).float().mean().item()
    assert accuracy>=.875 and np.mean(losses[-10:])<losses[0]*.6,(accuracy,losses[0],losses[-1])
    if rank==0:
        report={'state':'gpu_preflight_passed','gradient_max_relative_error':max_relative,'overfit_top1':accuracy,'initial_loss':losses[0],'final_loss':losses[-1],'input_shape':manifest['input_shape']}
        (root/'preflight.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
