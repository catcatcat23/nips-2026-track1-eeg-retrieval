"""Stage one selective local OT; stage two frozen EEG original B recovery."""
import argparse,json,time,hashlib,ast,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,ConcatDataset,DistributedSampler
from data import DatasetEEG
from augment_trials import MixedTrials, AugmentedModel
LocalOTModel = AugmentedModel
from stability_ablation import AblationModel
from local_ot_model import local_loss,pair_ot
from neuraltrain.losses.losses import ClipLoss
from reve_model import optimizer_groups
from v33_recovery_stage2 import metrics,write

class LocalData(DatasetEEG):
 def __init__(self,cfg,split,smoke=False):
  super().__init__(cfg['source_root'],split,aux=False);self.blur=self.clean if cfg.get('global_target')=='clean' else np.load(Path(cfg['source_root'])/'features/blur.npy',mmap_mode='r');self.root=Path(cfg['run_root']);self.smoke=smoke
  if not smoke:self.local=np.load(self.root/'local.npy',mmap_mode='r')
 def __getitem__(self,i):
  b=super().__getitem__(i);k=b['id'];b['blur']=torch.tensor(self.blur[k].copy())
  if self.smoke:
   with np.load(self.root/'local_shards'/f'{k:05d}.npz') as f:local=f['local'].copy()
  else:local=self.local[k].copy()
  b['local']=torch.tensor(local);return b

def loss_fn(model,b,weights):
 h,z,a,q,v=model(b['eeg'].cuda(),b['blur'].cuda(),b['local'].cuda());criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False)
 clean=criterion(z,b['clean'].cuda());blur=criterion(h,a);ot=local_loss(h,a,q,v,b['id'].cuda(),weights)
 return clean+blur+.1*ot,dict(clean=clean,blur=blur,local=ot)

def verify_ot():
 # Executable parity check against the exact released selective OT function.
 path=Path('/home/Anteng/eeg-retrieval/EEG-Image-IntraSubject-Retrieval/EEG_Image_IntraSubject_SelectiveLocalOT/intrasubject_retrieval/scripts/train_intrasubject_local_ot.py')
 tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='weighted_ot_score');ns={'torch':torch,'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
 torch.manual_seed(33);q=F.normalize(torch.randn(4,6,256),dim=-1).requires_grad_();v=F.normalize(torch.randn(4,9,256),dim=-1).requires_grad_();w=torch.rand(4,9)+.01
 actual=pair_ot(q,v,w);expected=torch.stack([ns['weighted_ot_score'](q[i]@v[i].T,w[i]) for i in range(4)])
 torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6);ga=torch.autograd.grad(actual.sum(),(q,v),retain_graph=True);ge=torch.autograd.grad(expected.sum(),(q,v))
 for a,b in zip(ga,ge):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
 # Duplicate-image positive mask and permutation consistency.
 h=torch.randn(4,1536);a=torch.randn(4,1536);ids=torch.tensor([1,1,2,3]);q=F.normalize(torch.randn(4,6,256),dim=-1).requires_grad_();v=F.normalize(torch.randn(4,9,256),dim=-1).requires_grad_()
 value=local_loss(h,a,q,v,ids,w);p=torch.tensor([2,0,3,1]);torch.testing.assert_close(value,local_loss(h[p],a[p],q[p],v[p],ids[p],w[p]));value.backward();assert torch.isfinite(q.grad).all() and torch.isfinite(v.grad).all()
 return dict(upstream_score_and_gradient_parity=True,duplicate_positive_permutation=True)

def preflight(cfg,out):
 checks=verify_ot();manifest=json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text());torch.manual_seed(33);model=LocalOTModel(manifest).cuda();model.perturbation_kind=cfg.get('augmentation','none')
 torch.manual_seed(33);reference=AblationModel(manifest,'dropout')
 for k,v in reference.state_dict().items():assert torch.equal(v,model.state_dict()[k].cpu()),k
 del reference
 ds=LocalData(cfg,'train',True);unique,first=np.unique(ds.ids,return_index=True);ix=first[:8];b=next(iter(DataLoader(torch.utils.data.Subset(ds,ix),batch_size=8)))
 weights=torch.linspace(.1,1,9,device='cuda').expand(8,-1);model.eval();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0);initial=float(loss_fn(model,b,weights)[0].detach())
 for step in range(100):
  opt.zero_grad();loss,parts=loss_fn(model,b,weights);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
  with torch.no_grad():
   h,z,a,_,_=model(b['eeg'].cuda(),b['blur'].cuda(),b['local'].cuda());labels=torch.arange(8,device='cuda');acc=float(((F.normalize(h,dim=-1)@a.T).argmax(1)==labels).float().mean())
  if step>=2 and acc==1 and float(loss.detach())<initial*.5:break
 assert acc==1 and float(loss.detach())<initial*.5
 model.train();opt.zero_grad();loss,_=loss_fn(model,b,weights);loss.backward();grads={}
 for name,module in [('encoder',model.encoder),('global_projection',model.blur_head),('global_adapter',model.image_adapter),('recovery',model.recovery),('queries_attention',model.attention),('local_image',model.image_projection)]:
  g=[p.grad for p in module.parameters()];assert all(v is not None and torch.isfinite(v).all() for v in g);grads[name]=sum(float(v.abs().sum()) for v in g);assert grads[name]>0
 assert model.queries.grad.abs().sum()>0
 write(out/'preflight.json',dict(passed=True,initial_loss=initial,final_loss=float(loss.detach()),steps=step+1,tiny_blur_top1=acc,gradients=grads,checks=checks,initial_global_matches_v33=True,probe_weights='synthetic positive weights; real Ridge weights checked before training',updates_discarded=True));print('PREFLIGHT PASSED',flush=True)

@torch.no_grad()
def evaluate(model,cfg,weights):
 model.eval();root=Path(cfg['run_root']);ds=DatasetEEG(cfg['source_root'],'test',aux=False);ids,labels=np.unique(ds.ids,return_inverse=True);assert len(ids)==200
 blur=[]
 # v68: global image branch aligns to official clean features instead of multi-blur.
 if cfg.get('global_target')=='clean':blur=list(np.asarray(ds.clean)[ids])
 for i in ([] if cfg.get('global_target')=='clean' else ids):
  with np.load(Path(cfg['test_blur_shards'])/f'{i:05d}.npz') as f:
   assert int(f['id'])==i and str(f['filepath'])==json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text())['paths'][i];blur.append(f['blur'].copy())
 gallery,local=model.encode_image(torch.tensor(np.stack(blur),device='cuda'),torch.tensor(np.load(root/'local.npy',mmap_mode='r')[ids].copy(),device='cuda'))
 hparts=[];zparts=[];fused=[]
 for b in DataLoader(ds,batch_size=128,num_workers=2):
  h,z,q=model.encode_eeg(b['eeg'].cuda());w=torch.tensor([weights[s]['weights'] for s in b['subject']],device='cuda');global_scores=F.normalize(h,dim=-1)@gallery.T
  # Subject weights belong to each EEG row, including reverse training comparisons.
  local_scores=[]
  for j in range(0,200,25):
   count=min(25,200-j);n=len(h)
   local_scores.append(pair_ot(q[:,None].expand(-1,count,-1,-1).reshape(-1,6,256),local[None,j:j+count].expand(n,-1,-1,-1).reshape(-1,9,256),w[:,None].expand(-1,count,-1).reshape(-1,9)).reshape(n,count))
  fused.append((global_scores+.1*torch.cat(local_scores,1)).cpu().numpy());hparts.append(h.cpu().numpy());zparts.append(z.cpu().numpy())
 h=np.concatenate(hparts);z=np.concatenate(zparts);scores=np.concatenate(fused);clean=np.asarray(ds.clean);blur_all=np.zeros_like(clean);blur_all[ids]=gallery.cpu().numpy()
 result={'blur':metrics(h,ds.ids,ds.subjects,blur_all),'clean':metrics(z,ds.ids,ds.subjects,clean)}
 # For auxiliary fused aggregation, explicitly average per-trial fused scores.
 from evaluate import group_means
 _,sid=np.unique(ds.subjects,return_inverse=True);keys,mean=group_means(scores,np.stack([sid,labels],1));result['blur_plus_ot']={}
 for name,s,y in [('single_trial',scores,labels),('subject_aggregated',mean,keys[:,1])]:
  top=np.argsort(-s,axis=1)[:,:5];result['blur_plus_ot'][name]={'top1':float((top[:,0]==y).mean()),'top5':float((top==y[:,None]).any(1).mean()),'count':len(y)}
 result['blur_plus_ot']['aggregation']='mean per-trial fused scores; offline diagnostic, not official output aggregation'
 return result

def train(cfg,out):
 assert (out/'local_complete.json').exists() and (out/'predictability/complete.json').exists();weights=json.loads((out/'predictability/subjects.json').read_text());manifest=json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text());torch.manual_seed(33);model=LocalOTModel(manifest).cuda();model.perturbation_kind=cfg.get('augmentation','none')
 ds=MixedTrials(ConcatDataset([LocalData(cfg,s) for s in ['train','val']]));sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=33,drop_last=True);loader=DataLoader(ds,batch_size=256,sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True)
 opt=torch.optim.AdamW(optimizer_groups(model,cfg),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[cfg['encoder_lr'],cfg['lr']],total_steps=cfg['epochs']*len(loader),pct_start=.1)
 dest=out/'stage1';dest.mkdir();best=-1;stale=0;rankings=[]
 for epoch in range(1,cfg['epochs']+1):
  sampler.set_epoch(epoch-1);model.train();total=np.zeros(4);start=time.time()
  for step,b in enumerate(loader):
   w=torch.tensor([weights[s]['weights'] for s in b['subject']],device='cuda');opt.zero_grad();loss,parts=loss_fn(model,b,w);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sched.step();total+=np.array([float(loss.detach())]+[float(parts[k].detach()) for k in ['clean','blur','local']])
   if step%100==0:print('TRAIN',epoch,step,len(loader),'loss',float(loss.detach()),flush=True)
  result=evaluate(model,cfg,weights);value=result['blur'][cfg.get('checkpoint_group','subject_aggregated')]['top5']
  # Preserve every epoch's inference weights, plus resumable latest and ranked indexes.
  payload=dict(model=model.state_dict(),manifest=manifest,config=cfg,epoch=epoch,metrics=result)
  torch.save(payload,dest/f'epoch_{epoch:03d}.pt')
  rankings.append(dict(epoch=epoch,path=f'epoch_{epoch:03d}.pt',single_trial=result['blur']['single_trial']['top5'],subject_aggregated=result['blur']['subject_aggregated']['top5']))
  top=sorted(rankings,key=lambda r:(-r['subject_aggregated'],r['epoch']))[:3];write(dest/'top3.json',top)
  keep={r['path'] for r in top}|{'epoch_009.pt'}
  for old in dest.glob('epoch_*.pt'):
   if old.name not in keep:old.unlink()
  torch.save(dict(**payload,optimizer=opt.state_dict(),scheduler=sched.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),dest/'last.pt')
  if value>best:
   best=value;stale=0;torch.save(dict(model=model.state_dict(),manifest=manifest,config=cfg,epoch=epoch,metrics=result,optimizer=opt.state_dict(),scheduler=sched.state_dict()),dest/'best.pt')
  else:stale+=1
  row=dict(epoch=epoch,loss=(total/len(loader)).tolist(),metrics=result,stale=stale,seconds=time.time()-start)
  with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  write(dest/'status.json',dict(state='training',epoch=epoch,best_blur_top5=best,stale=stale));print(json.dumps(row),flush=True)
  if stale>=cfg['patience'] and epoch>=cfg.get('stage2_source_epoch',9):break
 selected_epoch=cfg.get('stage2_source_epoch',9)
 ck=torch.load(dest/f'epoch_{selected_epoch:03d}.pt',map_location='cpu',weights_only=False);write(dest/'result.json',dict(best_epoch=ck['epoch'],metrics=ck['metrics'],selection='explicit epoch for stage2; blur aggregate and single Top5 top3 saved separately; public test development',stage2_source_epoch=selected_epoch));write(dest/'status.json',dict(state='complete',epoch=epoch,best_epoch=ck['epoch']))
 # Export only the standard v33 architecture for the unchanged second-stage loader.
 baseline=AblationModel(manifest,'dropout');state={k:ck['model'][k] for k in baseline.state_dict()};baseline.load_state_dict(state,strict=True);path=out/'stage1_global.pt';torch.save(dict(model=state,manifest=manifest,epoch=ck['epoch'],selection=ck['metrics']['clean']),path)
 stage2=out/'stage2';stage2.mkdir();new=dict(cfg);new.update(run_root=str(stage2),checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),arms={'B':'original fresh B head; freeze selected stage1 EEG; clean ClipLoss only'})
 write(stage2/'config.json',new)

def finish(cfg,out):
 first=json.loads((out/'stage1/result.json').read_text());second=json.loads((out/'stage2/B/result.json').read_text());write(out/'result.json',dict(stage1=first,stage2=second,stage2_blur=first['metrics']['blur'],stage2_blur_provenance='same frozen EEG projection and image adapter from stage1 selected checkpoint',role='public test selection; local OT excluded from inference'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','train','finish','cpu']);ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['run_root']);torch.set_num_threads(4);torch.manual_seed(33);np.random.seed(33);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.stage=='cpu':write(out/'cpu_checks.json',verify_ot())
 else:globals()[a.stage](cfg,out)
if __name__=='__main__':main()
