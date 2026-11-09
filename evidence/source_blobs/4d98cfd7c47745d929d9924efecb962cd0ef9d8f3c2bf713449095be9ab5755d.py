"""Eight real EEG/image pairs: FP32 capacity, gradients and inference contract."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import config
from reve_model import build_model,optimizer_groups
from losses import objective

cfg=config();root=Path(cfg['run_root']);torch.manual_seed(cfg['seed'])
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
manifest=json.loads((root/'data/manifest.json').read_text())
ids=np.load(root/'data/train_ids.npy');eeg=np.load(root/'data/train_eeg.npy',mmap_mode='r');clean=np.load(root/'data/clean.npy',mmap_mode='r')
files=sorted((root/'reference_smoke').glob('*.npz'));assert len(files)==8
image_ids=[];indices=[];blur=[]
for file in files:
    with np.load(file) as z:
        i=int(z['id']);assert str(z['filepath'])==manifest['paths'][i]
        matches=np.flatnonzero(ids==i);assert len(matches),i
        image_ids.append(i);indices.append(int(matches[0]));blur.append(z['blur'].copy())
x=torch.from_numpy(np.asarray(eeg[indices]).copy()).cuda();c=torch.from_numpy(np.asarray(clean[image_ids]).copy()).cuda();b=torch.from_numpy(np.stack(blur)).cuda();labels=torch.tensor(image_ids,device='cuda')
m=build_model(manifest,cfg).cuda();opt=torch.optim.AdamW(optimizer_groups(m,cfg),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
# Eval disables dropout but preserves autograd for deliberate tiny-set fitting.
m.eval();probe=m.encoder.to_patch_embedding[0].weight;before=probe.detach().clone();initial=None
for step in range(200):
    opt.zero_grad(set_to_none=True);outputs=m(x,b);loss,_=objective(outputs,c,b,labels,None,cfg,distributed=False)
    assert torch.isfinite(loss)
    if initial is None:initial=loss.item()
    loss.backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert probe.grad.abs().sum()>0 and m.image_adapter[0].weight.grad.abs().sum()>0
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    with torch.no_grad():
        pred=m(x);scores=F.normalize(pred,dim=-1)@F.normalize(c,dim=-1).T
        accuracy=(scores.argmax(-1)==torch.arange(len(x),device='cuda')).float().mean().item()
    if step%10==0:print(step,loss.item(),accuracy,flush=True)
    if accuracy==1 and loss.item()<initial*.2:break
assert accuracy==1 and loss.item()<initial*.2,(accuracy,initial,loss.item())
with torch.no_grad():
    assert pred.shape==(8,1536) and torch.isfinite(pred).all()
    torch.testing.assert_close(pred,torch.cat([m(t[None]) for t in x]),rtol=1e-4,atol=1e-4)
report={'passed':True,'samples':8,'steps':step+1,'top1':accuracy,'initial_loss':initial,'final_loss':loss.item(),'encoder_weight_max_change':(probe.detach()-before).abs().max().item(),'all_gradients_finite':True,'image_adapter_gradients_nonzero':True,'batch_independent_inference':True}
(root/'tiny_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
