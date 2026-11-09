"""Real eight-image end-to-end REVE overfit and encoder update gate."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate
from data import config, DatasetEEG
from reve_model import build_model, optimizer_groups
from losses import objective

cfg=config(); root=Path(cfg['run_root']); torch.manual_seed(cfg['seed'])
manifest=json.loads((root/'data/manifest.json').read_text())
ds=DatasetEEG(root,'train',aux=not cfg.get('clean_only',False),local_aux=False)
_,indices=np.unique(ds.ids,return_index=True)
b=default_collate([ds[int(i)] for i in indices[:8]])
b={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in b.items()}
m=build_model(manifest,cfg).cuda().train()
assert m.encoder.training and m.finetune
if cfg.get('clean_only',False):
    assert not hasattr(m,'restore') and not hasattr(m,'image_adapter')
    assert 'blur' not in b
for layer in m.modules():
    if isinstance(layer,torch.nn.Dropout):layer.p=0
probe=m.encoder.to_patch_embedding[0].weight
before=probe.detach().clone()
opt=torch.optim.AdamW(optimizer_groups(m,cfg),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
initial=None
for step in range(400):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):outputs=m(b['eeg'],b.get('blur'))
    loss,parts=objective(outputs,b['clean'],b.get('blur'),b['id'],None,cfg,distributed=False)
    if initial is None:initial=loss.item()
    loss.backward()
    for name,p in m.named_parameters():
        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert probe.grad.abs().sum()>0
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    if step%50==0:print(step,loss.item(),flush=True)
    if loss.item()<initial*.1:break
m.eval()
with torch.no_grad():
    z=m(b['eeg']);scores=F.normalize(z,dim=-1)@F.normalize(b['clean'],dim=-1).T
    accuracy=(scores.argmax(-1)==torch.arange(8,device='cuda')).float().mean().item()
delta=(probe.detach()-before).abs().max().item()
assert accuracy==1.0 and loss.item()<initial*.2 and delta>0,(accuracy,initial,loss.item(),delta)
report={'passed':True,'samples':8,'top1':accuracy,'initial_loss':initial,'final_loss':loss.item(),'encoder_weight_max_change':delta,'steps':step+1,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad)}
(root/'tiny_preflight.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
