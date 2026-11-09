"""v44: four residual blocks, preserving the original B initial function."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import v33_recovery_stage2 as engine

class DeeperHead(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(33)
        old=engine.ProjectedHead()
        extra=[engine.ResidualBlock(),engine.ResidualBlock()]
        for block in extra:
            nn.init.zeros_(block.net[-1].weight)
            nn.init.zeros_(block.net[-1].bias)
        self.net=nn.Sequential(*list(old.net.children())[:4],*extra,*list(old.net.children())[4:])
    def forward(self,h):return self.net(h)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());out=Path(cfg['run_root'])
    torch.set_num_threads(4);torch.manual_seed(33);np.random.seed(33);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=out/'cache';assert json.loads((cache/'complete.json').read_text())['checkpoint_sha256']==cfg['checkpoint_sha256']
    for split in ['train','val','test']:
        for key in ['ids','subjects']:
            assert np.array_equal(np.load(cache/f'{split}_{key}.npy'),np.load(Path(cfg['source_root'])/'data'/f'{split}_{key}.npy'))
    ids=np.load(cache/'train_ids.npy');_,ix=np.unique(ids,return_index=True);ix=ix[:8]
    h=torch.tensor(np.load(cache/'train_h.npy',mmap_mode='r')[ix].copy(),device='cuda')
    clean=np.load(Path(cfg['source_root'])/'data/clean.npy');y=torch.tensor(clean[ids[ix]],device='cuda');image_ids=torch.tensor(ids[ix],device='cuda')
    head=DeeperHead().cuda().eval();base=engine.ProjectedHead().cuda().eval();initial=torch.load(cfg['baseline_initial_reference'],map_location='cpu',weights_only=True);base.load_state_dict(initial)
    # Compare every inherited parameter, not only a few predictions.
    for k,v in initial.items():
        parts=k.split('.');index=int(parts[1]);parts[1]=str(index+2 if index>=4 else index)
        assert torch.equal(head.state_dict()['.'.join(parts)].cpu(),v),k
    with torch.no_grad():
        torch.testing.assert_close(head(h),base(h),rtol=0,atol=0)
        r=torch.randn(32,1536,device='cuda');torch.testing.assert_close(head(r),base(r),rtol=0,atol=0)
    counts={'baseline':sum(p.numel() for p in base.parameters()),'deeper':sum(p.numel() for p in head.parameters())}
    first=float(engine.objective(head(h),y,image_ids,'B')[0].detach());opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0)
    for step in range(100):
        opt.zero_grad();loss,_=engine.objective(head(h),y,image_ids,'B');assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        nn.utils.clip_grad_norm_(head.parameters(),1);opt.step()
        with torch.no_grad():acc=float(((F.normalize(head(h),dim=-1)@F.normalize(y,dim=-1).T).argmax(1)==torch.arange(8,device='cuda')).float().mean())
        if step>=2 and acc==1 and float(loss.detach())<first*.3:break
    assert acc==1 and float(loss.detach())<first*.3
    head.train();opt.zero_grad();engine.objective(head(h),y,image_ids,'B')[0].backward()
    extra_grad={str(i):sum(float(p.grad.abs().sum()) for p in head.net[i].parameters()) for i in [4,5]}
    assert all(x>0 for x in extra_grad.values())
    engine.write(out/'preflight.json',dict(passed=True,initial_function_exactly_matches_B=True,parameters=counts,initial_loss=first,final_loss=float(loss.detach()),steps=step+1,tiny_top1=acc,extra_block_gradients=extra_grad,updates_discarded=True))
    del head,opt,base
    fresh=DeeperHead().cuda();torch.save(fresh.state_dict(),out/'initial_head.pt');del fresh
    engine.make_head=lambda arm:DeeperHead().cuda()
    print('PREFLIGHT PASSED',counts,flush=True)
    (out/'status.txt').write_text('TRAINING\n')
    engine.train(cfg,out,'B')
    report=json.loads((out/'B/result.json').read_text());blur=json.loads(Path(cfg['blur_reference']).read_text());assert blur['checkpoint_sha256']==cfg['checkpoint_sha256']
    report.update(blur=blur['metrics']['eeg_to_adapted_blur'],blur_provenance='reuse measured frozen v33 encoder/projection/adapter with identical cached EEG features',parameters=counts)
    engine.write(out/'B/result.json',report);(out/'status.txt').write_text('COMPLETED\n')
if __name__=='__main__':main()
