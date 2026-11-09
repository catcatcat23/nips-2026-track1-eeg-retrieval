"""v64 stage1: exact v57 recipe (imports v57 data/loss/eval/model unchanged) plus passive weight EMA.
Changes vs local_ot_train.train: keep every epoch; EMA decays .999/.9998 updated after each optimizer step
(parameters lerped, buffers copied; never feeds back into training); raw + each EMA evaluated per epoch;
early stop when none of raw/EMA improved blur single-trial Top5 for `patience` epochs. No stage2 here."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,ConcatDataset,DistributedSampler
from local_ot_train import LocalData,LocalOTModel,MixedTrials,loss_fn,evaluate,optimizer_groups,write
import local_ot_train
import gradient_balance as gb
from image_contrastive import auxiliary_backward

class EMA:
 def __init__(self,model,decay):
  self.decay=decay;self.names=[n for n,_ in model.named_parameters()]
  self.params=[p.detach().clone() for _,p in model.named_parameters()]
 @torch.no_grad()
 def update(self,model):
  torch._foreach_lerp_(self.params,[p.detach() for p in model.parameters()],1-self.decay)
 def state_dict(self,model):
  # Buffers (e.g. electrode positions) come from the live model, parameters from the average.
  state={k:v.detach().clone() for k,v in model.state_dict().items()}
  for n,p in zip(self.names,self.params):state[n]=p.clone()
  return state

def variants(cfg):return ['raw']+[f"ema{d}" for d in cfg['ema_decays']]

def train(cfg,out):
 assert (out/'local_complete.json').exists() and (out/'predictability/complete.json').exists()
 weights=json.loads((out/'predictability/subjects.json').read_text());manifest=json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text())
 torch.manual_seed(33);model=LocalOTModel(manifest).cuda();model.perturbation_kind=cfg.get('augmentation','none')
 from image_contrastive import state_sha256
 initial_hash=state_sha256(model)
 assert initial_hash==json.loads((out/'image_contrastive_gpu_check.json').read_text())['initial_state_sha256'], 'Formal model initialization differs from verified v82 initialization'
 write(out/'initialization.json',dict(state_sha256=initial_hash,matches_v82=True,trained_retrieval_checkpoint_loaded=False))
 ds=MixedTrials(ConcatDataset([LocalData(cfg,s) for s in ['train','val']]));sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=33,drop_last=True);loader=DataLoader(ds,batch_size=256,sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True)
 opt=torch.optim.AdamW(optimizer_groups(model,cfg),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[cfg['encoder_lr'],cfg['lr']],total_steps=cfg['epochs']*len(loader),pct_start=.1)
 emas={f"ema{d}":EMA(model,d) for d in cfg['ema_decays']}
 evaluator=LocalOTModel(manifest).cuda().eval();evaluator.perturbation_kind=cfg.get('augmentation','none')
 dest=out/'stage1';dest.mkdir();best={v:-1. for v in variants(cfg)};best_epoch={v:0 for v in variants(cfg)};stale=0
 for epoch in range(1,cfg['epochs']+1):
  sampler.set_epoch(epoch-1);model.train();total=np.zeros(5);norms=[];start=time.time()
  for step,b in enumerate(loader):
   w=torch.tensor([weights[s]['weights'] for s in b['subject']],device='cuda');opt.zero_grad();loss,parts=loss_fn(model,b,w);assert torch.isfinite(loss);loss.backward()
   monitor=(step<3 or step%100==0)
   prior=gb.capture(model) if monitor else None
   image_contrastive=auxiliary_backward(model,cfg);parts['image_contrastive']=image_contrastive;loss=loss.detach()+cfg['image_contrastive_weight']*image_contrastive
   balance=gb.measure(model,prior) if monitor else None
   del prior
   norms.append(float(nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)))
   if monitor:
    balance.update(epoch=epoch,step=step,image_contrastive_loss=float(image_contrastive),retrieval_loss=float(loss)-cfg['image_contrastive_weight']*float(image_contrastive));gb.save(out,balance,norms[-1],cfg['image_contrastive_weight'])
   opt.step();sched.step()
   for e in emas.values():e.update(model)
   total+=np.array([float(loss.detach())]+[float(parts[k].detach()) for k in ['clean','blur','local','image_contrastive']])
   if step%100==0:print('TRAIN',epoch,step,len(loader),'loss',float(loss.detach()),'gradnorm',norms[-1],'lr',[g['lr'] for g in opt.param_groups],flush=True)
  train_seconds=time.time()-start;results={};eval_seconds={}
  for v in variants(cfg):
   t=time.time()
   if v=='raw':results[v]=evaluate(model,cfg,weights)
   else:evaluator.load_state_dict(emas[v].state_dict(model),strict=True);results[v]=evaluate(evaluator,cfg,weights)
   eval_seconds[v]=time.time()-t
  # Every epoch kept: raw with optimizer/scheduler (possible SWA fork later), EMA weights only.
  torch.save(dict(model=model.state_dict(),manifest=manifest,config=cfg,epoch=epoch,metrics=results['raw'],optimizer=opt.state_dict(),scheduler=sched.state_dict()),dest/f'epoch_{epoch:03d}.pt')
  for v,e in emas.items():torch.save(dict(model=e.state_dict(model),manifest=manifest,config=cfg,epoch=epoch,metrics=results[v],ema_decay=e.decay),dest/f'{v}_epoch_{epoch:03d}.pt')
  torch.save(dict(model=model.state_dict(),ema={v:e.state_dict(model) for v,e in emas.items()},manifest=manifest,config=cfg,epoch=epoch,optimizer=opt.state_dict(),scheduler=sched.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),best=best,best_epoch=best_epoch),dest/'last.pt')
  improved=False
  for v in variants(cfg):
   score=results[v]['blur'][cfg['checkpoint_group']]['top5']
   if score>best[v]:best[v]=score;best_epoch[v]=epoch;improved=True
  stale=0 if improved else stale+1
  n=np.array(norms);row=dict(epoch=epoch,loss=(total/len(loader)).tolist(),loss_parts=['total','clean','blur','local','image_contrastive'],lr_end=[g['lr'] for g in opt.param_groups],grad_norm=dict(mean=float(n.mean()),median=float(np.median(n)),max=float(n.max()),clipped_fraction=float((n>1).mean())),metrics=results,best=best,best_epoch=best_epoch,stale=stale,train_seconds=train_seconds,eval_seconds=eval_seconds)
  with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  write(dest/'status.json',dict(state='training',epoch=epoch,best=best,best_epoch=best_epoch,stale=stale))
  print('EPOCH',epoch,json.dumps({v:{'single':results[v]['blur']['single_trial'],'agg':results[v]['blur']['subject_aggregated'],'clean_single':results[v]['clean']['single_trial']} for v in results}),'stale',stale,flush=True)
  if stale>=cfg['patience']:break
 write(dest/'result.json',dict(best=best,best_epoch=best_epoch,selection=f"blur {cfg['checkpoint_group']} Top5 per variant; public test development selection; not independent heldout testing",stopped_epoch=epoch))
 write(dest/'status.json',dict(state='complete',epoch=epoch,best=best,best_epoch=best_epoch))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train']);ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['run_root'])
 torch.set_num_threads(4);torch.manual_seed(33);np.random.seed(33);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 train(cfg,out)
if __name__=='__main__':main()
