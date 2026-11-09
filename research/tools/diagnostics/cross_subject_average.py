"""Fixed v69 checkpoint: matched-count within/cross-subject fusion diagnostic."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from joint_train import JointModel,seed,LocalData,load_alljoined
root=Path(sys.argv[1]);base=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v69_joint_stage1')
seed();cfg=json.loads((base/'config.json').read_text());tc=json.loads(Path(cfg['things_config']).read_text());ac=json.loads(Path(cfg['alljoined_config']).read_text())
ckpath=base/'stage1/ema0.9998_epoch_015.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
model=JointModel(ck['manifest'],ck['alljoined_channels']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
meta,tr,ad=load_alljoined(ac);del tr
(root/'protocol.json').write_text(json.dumps(dict(checkpoint=str(ckpath),counts=[1,2,4,8],seeds=[101,102,103],paired=True,rule='same image; same total trials; cross includes anchor subject once and k-1 distinct others; within uses k distinct anchor trials; same selected anchor trial; separate datasets; no retraining',limitation='seen-subject public-test diagnostic, not evidence of unseen-subject generalization'),indent=2))
@torch.inference_mode()
def enc(x):
 hs=[];zs=[]
 for a in x.split(128):
  h,z,_=model.encode_eeg(a.cuda());hs.append(h);zs.append(z)
 return torch.cat(hs),torch.cat(zs)
rows=[]
for domain,ds in [('things',LocalData(tc,'test')),('alljoined',ad)]:
 model.select(domain);ids=np.unique(ds.ids);subjects=np.unique(ds.subjects);assert len(ids)==200
 groups={(s,int(i)):[] for s in subjects for i in ids}
 for j,(s,i) in enumerate(zip(ds.subjects,ds.ids)):groups[(s,int(i))].append(j)
 assert min(map(len,groups.values()))>=8
 blur=[]
 for i in ids:
  with np.load(Path(tc['test_blur_shards'])/f'{i:05d}.npz') as f:
   assert int(f['id'])==i;blur.append(f['blur'].copy())
 with torch.no_grad(): bg=F.normalize(model.image_adapter(torch.tensor(np.stack(blur),device='cuda')),dim=-1);cg=F.normalize(torch.tensor(ds.clean[ids],device='cuda'),dim=-1)
 for sd in [101,102,103]:
  rng=np.random.default_rng(sd)
  for label,i in enumerate(ids):
   chosen=[rng.choice(groups[(s,int(i))],8,replace=False) for s in subjects]
   eeg=torch.stack([ds[int(j)]['eeg'] for js in chosen for j in js]).reshape(len(subjects),8,-1,120)
   with torch.no_grad():
    hh,zz=enc(eeg.flatten(0,1));hh=hh.reshape(len(subjects),8,-1);zz=zz.reshape(len(subjects),8,-1)
    inputs=[];out_h=[];out_z=[];keys=[]
    for a,s in enumerate(subjects):
     others=rng.permutation([j for j in range(len(subjects)) if j!=a]);order=np.r_[a,others]
     for k in [1,2,4,8]:
      for mode in ['within','cross']:
       if mode=='within': x=eeg[a,:k];h=hh[a,:k];z=zz[a,:k]
       else: x=eeg[order[:k],0];h=hh[order[:k],0];z=zz[order[:k],0]
       inputs.append(x.mean(0));out_h.append(h.mean(0));out_z.append(z.mean(0));keys.append((str(s),k,mode))
    ih,iz=enc(torch.stack(inputs))
    for stage,h,z in [('mean_input',ih,iz),('mean_output',torch.stack(out_h),torch.stack(out_z))]:
     for space,p,g in [('blur',h,bg),('clean',z,cg)]:
      score=F.normalize(p,dim=-1)@g.T;top=score.topk(5,dim=-1).indices;wrong=score.clone();wrong[:,label]=-torch.inf;margin=score[:,label]-wrong.max(1).values
      for j,(s,k,mode) in enumerate(keys):rows.append(dict(domain=domain,seed=sd,image=int(i),subject=s,k=k,mode=mode,stage=stage,space=space,top1=int(top[j,0]==label),top5=int((top[j]==label).any()),margin=float(margin[j])))
   if label%50==0: print(domain,sd,label,'/200',flush=True)
  with (root/'queries.jsonl').open('w') as f:
   for row in rows:f.write(json.dumps(row)+'\n')
  summary=[]
  for stage in ['mean_input','mean_output']:
   for space in ['clean','blur']:
    for k in [1,2,4,8]:
     for mode in ['within','cross']:
      rr=[r for r in rows if r['domain']==domain and r['seed']==sd and r['stage']==stage and r['space']==space and r['k']==k and r['mode']==mode]
      summary.append(dict(domain=domain,seed=sd,stage=stage,space=space,k=k,mode=mode,**{t:float(np.mean([r[t] for r in rr])) for t in ['top1','top5','margin']}))
  with (root/'summary.jsonl').open('a') as f:
   for row in summary:f.write(json.dumps(row)+'\n')
  print('FINISHED',domain,sd,flush=True)
(root/'status.txt').write_text('COMPLETE\n')
