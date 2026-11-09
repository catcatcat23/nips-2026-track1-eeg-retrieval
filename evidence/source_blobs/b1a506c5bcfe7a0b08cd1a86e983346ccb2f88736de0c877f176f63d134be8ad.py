"""Image discrimination within subjects, reusing the identity auxiliary batch."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import grouped_identity as gi

SAMPLER = None
CALLS = 0

def picture_loss(h, images, subjects):
    f=F.normalize(h.float(),dim=-1)
    scores=f@f.T/.1
    eye=torch.eye(len(h),device=h.device,dtype=torch.bool)
    allowed=(subjects[:,None]==subjects[None,:]) & ~eye
    positive=allowed & (images[:,None]==images[None,:])
    negative=allowed & ~positive
    valid=positive.any(1)&negative.any(1)
    assert positive.sum(1).eq(1).all()
    if not valid.any(): return h.sum()*0, 0.
    logp=scores-scores.masked_fill(~allowed,-torch.inf).logsumexp(1,keepdim=True)
    loss=-(logp.masked_fill(~positive,0).sum(1)/positive.sum(1).clamp_min(1))[valid].mean()
    return loss,float(valid.float().mean())

def auxiliary_backward(model,cfg):
    global SAMPLER,CALLS
    if SAMPLER is None:SAMPLER=gi.GroupedIdentity(cfg)
    x,images,subjects=SAMPLER.sample();images=images.cuda();subjects=subjects.cuda()
    e=gi.pooled(model,x.cuda())
    identity=gi.conditional_loss(e,images,subjects,cfg['identity_temperature'])
    image,coverage=picture_loss(model.blur_head(e),images,subjects)
    if CALLS<3 or CALLS%100==0:
        params=list(model.encoder.parameters())
        a=torch.autograd.grad(cfg['identity_weight']*identity,params,retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(cfg['picture_weight']*image,params,retain_graph=True,allow_unused=True)
        aa=image.new_zeros(());bb=aa.clone();ab=aa.clone()
        for u,v in zip(a,b):
            if u is not None:aa+=u.detach().float().square().sum()
            if v is not None:bb+=v.detach().float().square().sum()
            if u is not None and v is not None:ab+=(u.detach().float()*v.detach().float()).sum()
        row=dict(step=CALLS,valid_fraction=coverage,identity_l2=float(aa.sqrt()),picture_l2=float(bb.sqrt()),cosine=float(ab/(aa*bb).sqrt().clamp_min(1e-20)))
        with (Path(cfg['run_root'])/'aux_gradient_balance.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    (cfg['identity_weight']*identity+cfg['picture_weight']*image).backward()
    model.pooled_for_identity=None
    CALLS+=1
    return identity.detach(),image.detach()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--gpu',action='store_true');a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(cfg['run_root']);sampler=gi.GroupedIdentity(cfg)
    fractions=[]
    for _ in range(20):
        x,im,sub=sampler.sample();h=torch.randn(256,32,requires_grad=True)
        loss,frac=picture_loss(h,im,sub);loss.backward();assert torch.isfinite(h.grad).all();fractions.append(frac)
        perm=torch.randperm(256);torch.testing.assert_close(loss.detach(),picture_loss(h.detach()[perm],im[perm],sub[perm])[0])
        # Excluding all other subjects must leave each subject's objective unchanged.
        ss=[sub==s for s in sub.unique()];weighted=[]
        for mask in ss:
            if len(im[mask].unique())>1:weighted.append((picture_loss(h[mask],im[mask],sub[mask])[0],int(mask.sum())))
        torch.testing.assert_close(loss.detach(),sum(v*n for v,n in weighted)/sum(n for _,n in weighted))
    report=dict(passed=True,batches=20,min_valid_fraction=min(fractions),mean_valid_fraction=float(np.mean(fractions)),same_sampler_as_v82=True,no_extra_encoder_forward=True)
    if a.gpu:
        import extensions as ex
        ex.CFG.update(cfg);torch.set_num_threads(4);torch.manual_seed(33)
        m=ex.IdentityModel(json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text())).cuda().eval()
        idloss,imgloss=auxiliary_backward(m,cfg)
        for module in [m.encoder,m.blur_head]:
            g=[p.grad for p in module.parameters() if p.grad is not None]
            assert g and all(torch.isfinite(v).all() for v in g) and sum(float(v.abs().sum()) for v in g)>0
        assert all(p.grad is None for p in m.recovery.parameters())
        # Two images, two real trials each, from one person: picture-only tiny fit.
        selected=sub==sub[0];x=x[selected][:8].cuda();im=im[selected][:8].cuda();sub=sub[selected][:8].cuda()
        opt=torch.optim.AdamW(list(m.encoder.parameters())+list(m.blur_head.parameters()),lr=1e-4)
        initial=float(picture_loss(m.blur_head(gi.pooled(m,x)),im,sub)[0].detach())
        for step in range(50):
            opt.zero_grad();loss,_=picture_loss(m.blur_head(gi.pooled(m,x)),im,sub);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
            if step>2 and float(loss.detach())<initial*.5:break
        assert float(loss.detach())<initial*.5
        report.update(initial=initial,final=float(loss.detach()),steps=step+1,updates_discarded=True)
    (out/('picture_gpu_check.json' if a.gpu else 'picture_cpu_check.json')).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
