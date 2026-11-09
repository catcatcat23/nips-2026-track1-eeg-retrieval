"""Train-only subject-reference translation; matched blur-only calibration controls."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import DatasetEEG
import extensions as ex
from local_ot_train import metrics

def main(a):
 torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
 out=a.out; out.mkdir(parents=True,exist_ok=True)
 def save(name,obj): (out/name).write_text(json.dumps(obj,indent=2))
 cfg=json.loads(a.config.read_text()); ex.CFG.update(cfg)
 fold=Path(cfg['run_root']);res=json.loads((fold/'stage1/result.json').read_text());v=max(res['best'],key=res['best'].get);ep=res['best_epoch'][v]
 cp=fold/'stage1'/(f'epoch_{ep:03d}.pt' if v=='raw' else f'{v}_epoch_{ep:03d}.pt')
 ck=torch.load(cp,map_location='cpu',weights_only=False)
 model=(ex.IdentityModel if cfg.get('identity_weight') else ex.AugmentedModel)(ck['manifest']).cuda().eval().requires_grad_(False); model.load_state_dict(ck['model'])
 captured=[]
 handle=model.blur_head.register_forward_pre_hook(lambda m,x: captured.append(x[0].detach()))
 train=[DatasetEEG(cfg['source_root'],s,aux=False) for s in ['train','val']];test=DatasetEEG(cfg['source_root'],'test',aux=False)
 groups={}
 for d,ds in enumerate(train):
  for i,(s,k) in enumerate(zip(ds.subjects.astype(str),ds.ids)):groups.setdefault((s,int(k)),[]).append((d,i))
 subjects=sorted(set(s for s,k in groups)); held=cfg['heldout_subject'];old=[s for s in subjects if s!=held]
 val=set(json.loads(Path(cfg['validation_ids_path']).read_text()))
 eligible=sorted(set(k for s,k in groups if s==held)-val-set(map(int,test.ids)))
 eligible=[k for k in eligible if all(len(groups.get((s,k),[]))>=4 for s in old)]
 rng=np.random.default_rng(910033);images=rng.choice(eligible,1000,replace=False);basis_images=np.random.default_rng(33).choice(eligible,1024,replace=False);union=np.unique(np.concatenate([images,basis_images]));lookup={int(k):i for i,k in enumerate(union)}
 records=[groups[(held,int(k))][int(rng.integers(len(groups[(held,int(k))])))] for k in images]
 @torch.inference_mode()
 def encode_records(records):
  values=[]
  for start in range(0,len(records),64):
   xs=torch.stack([torch.from_numpy(train[d].x[i].copy()) for d,i in records[start:start+64]]).cuda(); captured.clear();model.encode_eeg(xs);values.append(captured[-1].float().cpu())
  return torch.cat(values)
 source=[]
 for s in old:
  rec=[r for k in union for r in groups[(s,int(k))][:4]]
  source.append(encode_records(rec).reshape(len(union),4,512).mean(1));print('SOURCE',s,len(rec),flush=True)
 source=torch.stack(source);mean=source.mean(0);diff=source-mean
 basis_idx=[lookup[int(k)] for k in basis_images];P=diff[:,basis_idx].mean(1);_,sv,V=torch.linalg.svd(P-P.mean(0),full_matrices=False);rank=min(len(old)-1,int((sv>sv.max()*1e-6).sum()));U=V[:rank].T.contiguous();assert rank>0
 assert torch.allclose(U.T@U,torch.eye(rank),atol=1e-5)
 etrain=encode_records(records); ti=np.flatnonzero(test.subjects.astype(str)==held); tids=test.ids[ti];ts=test.subjects[ti]
 @torch.inference_mode()
 def encode_test():
  values=[]
  for start in range(0,len(ti),64):
   captured.clear();model.encode_eeg(torch.tensor(test.x[ti[start:start+64]].copy(),device='cuda'));values.append(captured[-1].float().cpu())
  return torch.cat(values)
 etest=encode_test();handle.remove();assert not np.intersect1d(images,tids).size
 cache=np.load(Path(cfg['source_root'])/'features/blur.npy',mmap_mode='r')
 with torch.no_grad():targets=F.normalize(model.image_adapter(torch.tensor(cache[images].copy(),device='cuda')),dim=-1)
 galleryids=np.unique(tids); gb=[]
 for k in galleryids:
  with np.load(Path(cfg['test_blur_shards'])/f'{k:05d}.npz') as f:gb.append(f['blur'].copy())
 with torch.no_grad():g=F.normalize(model.image_adapter(torch.tensor(np.stack(gb),device='cuda')),dim=-1).cpu().numpy()
 gallery=np.zeros_like(test.clean);gallery[galleryids]=g
 torch.save(dict(source_means=source,union=union,images=images,basis_images=basis_images,P=P,U=U,etrain=etrain,etest=etest,records=records),out/'features.pt')
 scale=float((P@U).square().mean().sqrt().clamp_min(1e-6));U=U.cuda();etrain=etrain.cuda();etest=etest.cuda()
 protocol=dict(checkpoint=str(cp),subjects=old,heldout=held,rank=rank,budgets=[50,200,1000],steps=[0,10,25,50,100,200],lr=.01,reference_weight=1.,scale=scale,images=images.tolist(),records=records,basis_images=basis_images.tolist(),source_repeats=4,selection='fixed training-only common images; exclude source validation IDs and test IDs',reference='top3 distance softmax; temperature median positive source pairwise squared distance',note='exploratory fixed settings, no test tuning; e_prime=e+scale*U*q; q zero-init; blur-only CE; same batch draws across conditions; random reference same weights with random subjects')
 save('protocol.json',protocol)
 @torch.inference_mode()
 def evaluate(q):
  pred=torch.cat([model.blur_head(x+scale*(U@q)).float().cpu() for x in etest.split(128)]).numpy()
  return metrics(pred,tids,ts,gallery)
 zero=torch.zeros(rank,device='cuda');base=evaluate(zero);results={'zero':base};save('result.json',results)
 coords=P@U.cpu();pair=torch.pdist(coords).square();tau=float(pair[pair>0].median().clamp_min(1e-8))
 for budget in [50,200,1000]:
  pnew=(etrain[:budget].cpu()-mean[[lookup[int(k)] for k in images[:budget]]]).mean(0)@U.cpu()
  distances=(coords-pnew).square().sum(1);near=distances.argsort()[:3];w=torch.softmax(-distances[near]/tau,0);random_idx=torch.tensor(np.random.default_rng(333).choice(len(old),3,replace=False))
  refs={'nearest':coords[near[0]],'weighted':(coords[near]*w[:,None]).sum(0),'random':(coords[random_idx]*w[:,None]).sum(0),'none':pnew}
  for name,ref in refs.items():
   torch.manual_seed(33);q=torch.nn.Parameter(torch.zeros(rank,device='cuda'));opt=torch.optim.Adam([q],lr=.01);desired=((ref-pnew)/scale).cuda();scores={'0':base};rows=[]
   for step in range(1,201):
    ix=torch.randperm(budget,device='cuda')[:50];h=model.blur_head(etrain[ix]+scale*(U@q));ce=F.cross_entropy(h@targets[ix].T,torch.arange(len(ix),device='cuda'));reg=(q-desired).square().mean();loss=ce+(reg if name!='none' else reg*0)
    opt.zero_grad();loss.backward();assert q.grad is not None and torch.isfinite(q.grad).all();torch.nn.utils.clip_grad_norm_([q],1);opt.step();rows.append([step,float(ce.detach()),float(reg.detach()),float(q.detach().norm())])
    if step in [10,25,50,100,200]:scores[str(step)]=evaluate(q);print('EVAL',budget,name,step,scores[str(step)]['subject_aggregated'],flush=True)
   results[f'{budget}_{name}']=dict(scores=scores,losses=rows,nearest_subjects=[old[i] for i in near],weights=w.tolist(),random_subjects=[old[i] for i in random_idx],delta=q.detach().cpu().tolist());save('result.json',results)
   torch.save(dict(q=q.detach().cpu(),U=U.cpu(),scale=scale,checkpoint=str(cp)),out/f'{budget}_{name}.pt')
 save('status.json',dict(state='complete'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
