"""v64 preflight. All updates on throwaway models; production model is re-created in train()."""
import json,sys,time,shutil
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,ConcatDataset,Subset
import local_ot_train as base
from local_ot_train_ema import EMA
from stability_ablation import AblationModel
from data import DatasetEEG
cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(cfg['run_root'])
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.manual_seed(33);np.random.seed(33)
res={}
base.preflight(cfg,out);res['base_preflight']=json.loads((out/'preflight.json').read_text())['passed']
shutil.move(out/'preflight.json',out/'preflight_base.json')
manifest=json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text())
weights=json.loads((out/'predictability/subjects.json').read_text())
ds=base.LocalData(cfg,'train',True);_,first=np.unique(ds.ids,return_index=True)
b=next(iter(DataLoader(Subset(ds,first[:16]),batch_size=16)));w=torch.tensor([weights[s]['weights'] for s in b['subject']],device='cuda')
def make():
 torch.manual_seed(33);m=base.LocalOTModel(manifest).cuda();m.perturbation_kind=cfg['augmentation'];return m
a,c=make(),make();d=0.9998
oa=torch.optim.AdamW(base.optimizer_groups(a,cfg),lr=1e-4,weight_decay=.05);oc=torch.optim.AdamW(base.optimizer_groups(c,cfg),lr=1e-4,weight_decay=.05)
ema=EMA(c,d)
assert all(torch.equal(v,c.state_dict()[k]) for k,v in ema.state_dict(c).items());res['ema_init_equals_model']=True
snaps=[[p.detach().clone() for p in c.parameters()]]
for step in range(2):
 for m,o,use in [(a,oa,False),(c,oc,True)]:
  m.train();torch.manual_seed(100+step);o.zero_grad();loss,_=base.loss_fn(m,b,w);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);o.step()
  if use:ema.update(m)
 snaps.append([p.detach().clone() for p in c.parameters()])
assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),c.state_dict().values()));res['ema_does_not_change_training_bitwise']=True
moved=sum(int(not torch.equal(x,y)) for x,y in zip(snaps[0],snaps[2]));assert moved>0;res['params_changed_by_steps']=moved
err=max(float((e-(d*d*p0+d*(1-d)*p1+(1-d)*p2)).abs().max()) for e,p0,p1,p2 in zip(ema.params,*snaps));assert err<1e-6,err;res['ema_formula_max_abs_err']=err
st=ema.state_dict(c);sd=c.state_dict();bufs={k:v for k,v in c.named_buffers() if k in sd};assert all(torch.equal(st[k],v) for k,v in bufs.items());res['persistent_buffers_copied']=list(bufs);res['non_persistent_buffers_ignored']=[k for k,_ in c.named_buffers() if k not in sd]
ev=base.LocalOTModel(manifest).cuda().eval();ev.load_state_dict(st,strict=True)
g=AblationModel(manifest,'dropout');keys=set(g.state_dict());g.load_state_dict({k:v.cpu() for k,v in st.items() if k in keys},strict=True)
if cfg.get('v57_global_reference'):
 ref=torch.load(Path(cfg['v57_global_reference']),map_location='cpu',weights_only=False)
 assert keys==set(ref['model']);res['stage2_loader_strict_load_and_keys_match_v57_global']=True
with torch.no_grad():
 g=g.cuda().eval();x=g.preprocess(b['eeg'].cuda());pos=g.positions.to(x).unsqueeze(0).expand(len(x),-1,-1);h=g.blur_head(g.encoder(x,pos=pos,return_output=True)[-1].mean(1));assert torch.isfinite(h).all()
del a,c,oa,oc,ev,g;torch.cuda.empty_cache()
full=base.MixedTrials(ConcatDataset([base.LocalData(cfg,s) for s in ['train','val']]));assert len(full)==668400
trainids=np.concatenate([d.ids for d in full.ds.datasets]);testids=DatasetEEG(cfg['source_root'],'test',aux=False).ids;assert not np.intersect1d(trainids,testids).size
res['train_trials']=len(full);res['train_test_image_overlap']=0
loader=DataLoader(full,batch_size=256,shuffle=True,num_workers=4,drop_last=True);bb=next(iter(loader))
m=make();m.train();o=torch.optim.AdamW(base.optimizer_groups(m,cfg),lr=1e-4)
t=time.time();ww=torch.tensor([weights[s]['weights'] for s in bb['subject']],device='cuda');loss,parts=base.loss_fn(m,bb,ww);loss.backward();gn=float(nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));torch.cuda.synchronize()
res['real_batch']=dict(loss=float(loss),parts={k:float(v) for k,v in parts.items()},grad_norm=gn,repeat_counts=np.bincount(bb['repeat_count'].numpy()).tolist(),seconds=time.time()-t)
m.eval();t=time.time();r=base.evaluate(m,cfg,weights);res['eval_seconds_per_variant']=time.time()-t
res['epoch0_untrained']={k:{g2:r[k][g2] for g2 in ('single_trial','subject_aggregated')} for k in ('blur','clean')}
res['peak_gpu_mem_gb']=torch.cuda.max_memory_allocated()/1e9;res['passed']=True
(out/'preflight_ema.json').write_text(json.dumps(res,indent=1));print(json.dumps(res,indent=1))
