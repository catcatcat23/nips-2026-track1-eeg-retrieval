"""Evaluation only: how often a correct blur-teacher retrieval survives recovery."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v33_recovery_stage2 import make_head,predictions,write
from evaluate import normalize,group_means
from evaluate_parallel import gpu_hits
ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
cfg=json.loads(args.config.read_text());out=Path(cfg['run_root']);cache=Path(cfg['cache_root'])
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ck=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
ids=np.load(cache/'test_ids.npy');subjects=np.load(cache/'test_subjects.npy');unique,labels=np.unique(ids,return_inverse=True)
raw=[]
for i in unique:
 with np.load(Path('/home/Anteng/eeg-retrieval/audits/blur_head_probe/test_blur_shards')/f'{i:05d}.npz') as f:
  assert int(f['id'])==i and str(f['filepath'])==ck['manifest']['paths'][i]
  raw.append(f['blur'].copy())
adapter=nn.Sequential(nn.Linear(1536,768),nn.ELU(),nn.Dropout(.2),nn.Linear(768,1536)).cuda().eval()
adapter.load_state_dict({k.removeprefix('image_adapter.'):v for k,v in ck['model'].items() if k.startswith('image_adapter.')})
with torch.inference_mode():gallery=F.normalize(adapter(torch.tensor(np.stack(raw),device='cuda')),dim=-1).cpu().numpy()
h=np.load(cache/'test_h.npy');head=make_head('B');best=torch.load(out/'B/best.pt',map_location='cpu',weights_only=False);head.load_state_dict(best['head'])
z=predictions(head,torch.from_numpy(h).cuda());clean=np.load(Path(cfg['source_root'])/'data/clean.npy')[unique]
_,sid=np.unique(subjects,return_inverse=True);keys=np.stack([sid,labels],1)
groups,hmean=group_means(h,keys);_,zmean=group_means(z,keys)
report={'epoch':best['epoch'],'evaluation_only':True,'metrics':{}}
for name,a,b,y in [('single_trial',h,z,labels),('subject_aggregated',hmean,zmean,groups[:,1])]:
 th=gpu_hits(a,gallery,y,torch.device('cuda'));sh=gpu_hits(b,normalize(clean),y,torch.device('cuda'))
 report['metrics'][name]={}
 for j,k in enumerate(['top1','top5']):
  t=th[:,j].astype(bool);s=sh[:,j].astype(bool)
  report['metrics'][name][k]={'teacher':float(t.mean()),'student':float(s.mean()),'teacher_correct_retained':float(s[t].mean()),'teacher_wrong_student_correct':float(s[~t].mean()),'teacher_correct_count':int(t.sum())}
write(out/'B/retention.json',report);print(json.dumps(report),flush=True)
