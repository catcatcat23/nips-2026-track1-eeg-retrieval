import numpy as np
import torch
from torch.utils.data import Dataset
from local_ot_model import LocalOTModel
class MixedTrials(Dataset):
 def __init__(self,ds):
  self.ds=ds;self.keys=[]
  for d in ds.datasets:self.keys.extend(zip(d.subjects.astype(str).tolist(),d.ids.tolist()))
  self.groups={}
  for i,k in enumerate(self.keys):self.groups.setdefault(k,[]).append(i)
 def __len__(self):return len(self.ds)
 def __getitem__(self,i):
  b=self.ds[i];r=np.random.random();requested=1 if r<.5 else (2 if r<.75 else 4)
  pool=self.groups[self.keys[i]];n=min(requested,len(pool))
  if n>1:
   others=[j for j in pool if j!=i];chosen=[i]+np.random.choice(others,n-1,replace=False).tolist()
   assert all(self.keys[j]==self.keys[i] for j in chosen)
   b['eeg']=torch.stack([self.ds[j]['eeg'] for j in chosen]).mean(0)
  b['repeat_count']=n
  return b
class AugmentedModel(LocalOTModel):
 perturbation_kind='none'
 def preprocess(self,x):
  x=super().preprocess(x)
  if self.training and self.perturbation_kind!='none':
   selected=torch.rand(len(x),device=x.device)<.2
   if self.perturbation_kind=='noise':
    x=x+selected[:,None,None]*torch.randn_like(x)*.02
   elif self.perturbation_kind=='channel':
    x=x.clone();rows=selected.nonzero().flatten();cols=torch.randint(x.shape[1],(len(rows),),device=x.device);x[rows,cols]=0
  return x
