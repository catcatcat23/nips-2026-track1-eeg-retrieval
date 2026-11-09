"""Fix distillation weight from eight training-only batches, before optimization."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from blur_distill_recovery import prepare_teacher,make_head,write
from neuraltrain.losses.losses import ClipLoss
ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
cfg=json.loads(args.config.read_text());cache=Path(cfg['cache_root']);out=Path(cfg['run_root'])
assert not (out/'gradient_calibration.json').exists()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ids=np.load(cache/'train_ids.npy');testids=np.load(cache/'test_ids.npy');h=np.load(cache/'train_h.npy',mmap_mode='r')
clean=np.load(Path(cfg['source_root'])/'data/clean.npy');teacher=prepare_teacher(cfg,ids,testids)
model=make_head('B').train();initial=torch.load(cfg['initial_reference'],map_location='cpu',weights_only=True)
assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
params=list(model.parameters());rng=np.random.default_rng(33041);order=rng.permutation(len(ids))[:8*256];rows=[]
for b in range(8):
 ix=order[b*256:(b+1)*256];bi=torch.tensor(ids[ix],device='cuda');bh=torch.tensor(h[ix].copy(),device='cuda');y=torch.tensor(clean[ids[ix]],device='cuda')
 q=model(bh);ce=ClipLoss(norm_kind='y',temperature=False,symmetric=False)(q,y)
 unique,inverse=torch.unique(bi,sorted=True,return_inverse=True)
 first=torch.full((len(unique),),len(bi),device='cuda',dtype=torch.long);first.scatter_reduce_(0,inverse,torch.arange(len(bi),device='cuda'),reduce='amin')
 tau=cfg['distill_temperature']
 with torch.no_grad():p=(F.normalize(bh,dim=-1)@teacher[unique].T/tau).softmax(-1)
 kl=F.kl_div((F.normalize(q,dim=-1)@F.normalize(y[first],dim=-1).T/tau).log_softmax(-1),p,reduction='batchmean')*tau*tau
 gc=torch.autograd.grad(ce,params,retain_graph=True);gk=torch.autograd.grad(kl,params)
 cn=torch.sqrt(sum(g.square().sum() for g in gc));kn=torch.sqrt(sum(g.square().sum() for g in gk));dot=sum((a*b).sum() for a,b in zip(gc,gk))
 assert torch.isfinite(cn) and torch.isfinite(kn) and kn>0
 rows.append(dict(batch=b,clean_loss=float(ce.detach()),kl_scaled=float(kl.detach()),clean_grad=float(cn),distill_grad=float(kn),gradient_cosine=float(dot/(cn*kn)),teacher_entropy_fraction=float((-(p*p.clamp_min(1e-12).log()).sum(1).mean())/math.log(len(unique))),unique_images=len(unique)))
ratio=float(np.median([r['distill_grad']/r['clean_grad'] for r in rows]));weight=.2/ratio
assert math.isfinite(weight) and weight>0
cfg['distill_weight']=weight;cfg['calibration']='8 train-only batches at original initialization; weight fixed to median distill/clean gradient norm ratio 0.2; no optimization during calibration'
write(out/'gradient_calibration.json',dict(rows=rows,unweighted_gradient_ratio=ratio,target_ratio=.2,selected_weight=weight,test_used=False))
write(args.config,cfg);print('CALIBRATION',json.dumps(dict(ratio=ratio,weight=weight)),flush=True)
