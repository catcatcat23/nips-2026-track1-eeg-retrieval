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
        identity=p.grad.detach()-retrieval
        a+=retrieval.float().square().sum();b+=identity.float().square().sum()
        dot+=(retrieval.float()*identity.float()).sum()
    nr=float(a.sqrt());ni=float(b.sqrt());cos=float(dot/(a*b).sqrt().clamp_min(1e-20))
    ratio=ni/max(nr,1e-20)
    return dict(retrieval_encoder_l2=nr,auxiliary_weighted_encoder_l2=ni,
                auxiliary_to_retrieval_ratio=ratio,cosine=cos,
                encoder_combined_l2=float((a+b+2*dot).clamp_min(0).sqrt()),
                scale_alert=ratio<.1 or ratio>10)


def save(out,row,total_norm,weight):
    row['all_parameter_preclip_l2']=total_norm
    row['shared_clip_factor']=min(1.,1./(total_norm+1e-6))
    row['measurement']='actual branch gradients before common clipping; combined identity + picture auxiliary gradients, both weighted'
    path=Path(out)/'stage1/gradient_balance.jsonl'
    with path.open('a') as f:f.write(json.dumps(row)+'\n')
    print('GRADIENT_BALANCE',json.dumps(row),flush=True)
