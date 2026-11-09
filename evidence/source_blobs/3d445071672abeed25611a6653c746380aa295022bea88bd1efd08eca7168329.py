"""Observe actual two-branch encoder gradients before shared clipping; no extra backward."""
import json
from pathlib import Path
import torch


def capture(model):
    return {n:p.grad.detach().clone() for n,p in model.encoder.named_parameters() if p.grad is not None}


def measure(model, before):
    ref=next(model.encoder.parameters()); a=ref.new_zeros((),dtype=torch.float32);b=a.clone();dot=a.clone()
    for n,p in model.encoder.named_parameters():
        if p.grad is None: continue
        retrieval=before.get(n)
        if retrieval is None: retrieval=torch.zeros_like(p.grad)
        image_contrastive=p.grad.detach()-retrieval
        a+=retrieval.float().square().sum();b+=image_contrastive.float().square().sum()
        dot+=(retrieval.float()*image_contrastive.float()).sum()
    nr=float(a.sqrt());ni=float(b.sqrt());cos=float(dot/(a*b).sqrt().clamp_min(1e-20))
    ratio=ni/max(nr,1e-20)
    return dict(retrieval_encoder_l2=nr,image_contrastive_weighted_encoder_l2=ni,
                image_contrastive_to_retrieval_ratio=ratio,cosine=cos,
                encoder_combined_l2=float((a+b+2*dot).clamp_min(0).sqrt()),
                scale_alert=ratio<.1 or ratio>10)


def save(out,row,total_norm,weight):
    row['image_contrastive_unweighted_encoder_l2']=row['image_contrastive_weighted_encoder_l2']/weight
    row['all_parameter_preclip_l2']=total_norm
    row['shared_clip_factor']=min(1.,1./(total_norm+1e-6))
    row['measurement']='actual branch gradients before common clipping; image_contrastive already weighted'
    path=Path(out)/'stage1/gradient_balance.jsonl'
    with path.open('a') as f:f.write(json.dumps(row)+'\n')
    print('GRADIENT_BALANCE',json.dumps(row),flush=True)
