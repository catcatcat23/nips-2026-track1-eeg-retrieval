import json
from pathlib import Path
import numpy as np
import torch
import extensions as ex
from torch.utils.data import Dataset,ConcatDataset

# Exact four-trial pairing with a deterministic, inspectable synthetic dataset.
class Tiny(Dataset):
    subjects=np.repeat(['s1','s2'],8)
    ids=np.tile(np.repeat([1,2],4),2)
    def __len__(self):return 16
    def __getitem__(self,i):return dict(eeg=torch.full((2,3),float(i)),subject=self.subjects[i],id=int(self.ids[i]))
ex.CFG['fixed_repeats']=4
ds=ex.Trials(ConcatDataset([Tiny()]))
for i in range(16):
    b=ds[i];group=ds.groups[ds.keys[i]]
    assert b['repeat_count']==4
    torch.testing.assert_close(b['eeg'],torch.full((2,3),float(np.mean(group))))
# Subject contrastive objective: finite gradients, permutation invariant,
# self excluded, no-positive anchors safe, matching subjects score better.
y=torch.tensor([0,0,1,1]);good=torch.tensor([[1.,0],[1.,.01],[0.,1],[.01,1]],requires_grad=True)
bad=good[[0,2,1,3]].detach().requires_grad_()
a=ex.subject_loss(good,y);b=ex.subject_loss(bad,y);assert a<b
p=torch.tensor([2,0,3,1]);torch.testing.assert_close(a,ex.subject_loss(good[p],y[p]));a.backward();assert torch.isfinite(good.grad).all()
assert ex.subject_loss(good,torch.arange(4))==0
# Real v64 trial-group minimum counts; no four-repeat fallback silently allowed.
root=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v13_reference/data')
ids=np.concatenate([np.load(root/f'{s}_ids.npy') for s in ['train','val']])
subs=np.concatenate([np.load(root/f'{s}_subjects.npy').astype(str) for s in ['train','val']])
_,counts=np.unique(np.rec.fromarrays([subs,ids]),return_counts=True)
assert counts.min()>=4
print(json.dumps(dict(passed=True,rows=len(ids),groups=len(counts),min_repeats=int(counts.min()),max_repeats=int(counts.max()),subjects=np.unique(subs).tolist(),subject_loss_good=float(a.detach()),subject_loss_bad=float(b.detach()))))
