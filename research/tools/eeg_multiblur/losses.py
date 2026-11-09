import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F

def gather(x):
    return torch.cat(all_gather(x.contiguous()),0) if dist.is_initialized() else x

def gather_ids(ids):
    if not dist.is_initialized():return ids
    result=[torch.empty_like(ids) for _ in range(dist.get_world_size())]
    dist.all_gather(result,ids);return torch.cat(result)

def multipos(logits,positive,allowed=None):
    if allowed is not None:logits=logits.masked_fill(~allowed,-torch.inf)
    logp=F.log_softmax(logits,dim=-1)
    return -(logp.masked_fill(~positive,0).sum(-1)/positive.sum(-1).clamp_min(1)).mean()

def contrast(a,b,positive,temp):
    scores=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).T/temp
    return (multipos(scores,positive)+multipos(scores.T,positive.T))*.5

def ot_scores(q,v,weights):
    patch=weights.topk(5).indices
    sim=torch.einsum('aqd,bpd->abqp',q,v[:,patch])
    qi=sim.max(-1).values.topk(3,dim=-1).indices
    sim=sim.gather(-2,qi.unsqueeze(-1).expand(-1,-1,-1,5))
    w=weights[patch];w=w/w.sum()
    kernel=sim/.1
    a=torch.full_like(sim[...,0],-torch.log(sim.new_tensor(3.)))
    b=w.log().expand(sim.shape[:-2]+(5,))
    u=torch.zeros_like(a);vv=torch.zeros_like(b)
    for _ in range(6):
        u=a-torch.logsumexp(kernel+vv.unsqueeze(-2),dim=-1)
        vv=b-torch.logsumexp(kernel+u.unsqueeze(-1),dim=-2)
    plan=torch.exp(kernel+u.unsqueeze(-1)+vv.unsqueeze(-2))
    return (plan*sim).sum((-2,-1))

def hard_mask(scores,positive):
    # All repeated-image positives are retained. Mask topk invalid selections.
    n=min(31,max(0,scores.shape[1]-1));mask=positive.clone()
    if n:
        idx=scores.detach().masked_fill(positive,-torch.inf).topk(n,dim=1).indices
        mask.scatter_(1,idx,True)
    return mask

def objective(outputs,clean,blur,ids,weights,cfg,distributed=True):
    if cfg.get('recipe') == 'reference_94b3a31':
        from image_id_loss import ImageIdClipLoss
        pred,z,adapted=[gather(x.float()) if distributed else x.float() for x in outputs]
        target=gather(clean.float()) if distributed else clean.float()
        criterion=ImageIdClipLoss().to(z.device)
        image_ids=gather_ids(ids) if distributed else ids
        lc=criterion(z,target,image_ids);lb=criterion(pred,adapted,image_ids)
        return lc+lb,{'clean':lc.detach().item(),'blur':lb.detach().item(),'restore':0.,'local':0.}
    # Full global loss on every rank. Autograd gather SUMs feature gradients;
    # DDP averages parameter gradients: the two world-size factors cancel.
    if cfg.get('clean_only', False):
        assert not cfg.get('image_adapter', False) and not cfg['local_weight']
        assert cfg['blur_weight']==0 and cfg['restore_weight']==0
        z=gather(outputs.float()) if distributed else outputs.float()
        target=gather(clean.float()) if distributed else clean.float()
        image_ids=gather_ids(ids) if distributed else ids
        lc=contrast(z,target,image_ids[:,None]==image_ids[None,:],cfg['temperature'])
        return lc,{'clean':lc.detach().item(),'blur':0.,'restore':0.,'local':0.}
    gathered=[gather(x.float()) if distributed else x.float() for x in outputs]
    h,z,r=gathered[:3]
    clean=gather(clean.float()) if distributed else clean.float()
    if cfg.get('image_adapter', False):
        assert not cfg['local_weight'], 'Image adapter branch does not implement local OT'
        blur=gathered[3]  # Autograd gather preserves image adapter gradients.
    else:
        blur=gather(blur.float()) if distributed else blur.float()
    ids=gather_ids(ids) if distributed else ids
    positive=ids[:,None]==ids[None,:]
    lc=contrast(z,clean,positive,cfg['temperature'])
    lb=contrast(h,blur,positive,cfg['temperature'])
    lr=(1-F.cosine_similarity(r,clean)).mean()
    ll=lc.new_zeros(())
    if cfg['local_weight']:
        q,v=gathered[3:]
        hard=F.normalize(z.detach(),dim=-1)@F.normalize(clean,dim=-1).T
        scores=ot_scores(q,v,weights)/cfg['local_temperature']
        ll=.5*(multipos(scores,positive,hard_mask(hard,positive))+multipos(scores.T,positive.T,hard_mask(hard.T,positive.T)))
    loss=lc+cfg['blur_weight']*lb+cfg['restore_weight']*lr+cfg['local_weight']*ll
    return loss,{'clean':lc.detach().item(),'blur':lb.detach().item(),'restore':lr.detach().item(),'local':ll.detach().item()}
