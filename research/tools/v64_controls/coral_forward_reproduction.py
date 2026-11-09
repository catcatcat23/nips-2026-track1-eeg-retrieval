"""CORAL Algorithm1 source->target; source-labelled Ridge retrieval readout.
Sun et al AAAI2016 arXiv1511.05547v2 Section3.2 Algorithm1.
Algorithm reproduction, EEG retrieval downstream adaptation, not original benchmark.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common_target_readout import load,write
from pooled_residual_calibration import native
from closed_form_adaptation import cov,power,evaluate_prediction

# Same raw feature coordinates and fixed regularization for both arms.
# No target labels used; no post-CORAL standardization that could undo alignment.
def fit(x,y):
    x=x.double();y=y.double();mu=x.mean(0);b=y.mean(0);z=x-mu
    w=torch.linalg.solve(z.T@z/len(z)+10*torch.eye(z.shape[1],dtype=z.dtype),z.T@(y-b)/len(z))
    return mu,w,b

def main(a):
    torch.set_num_threads(4);a.out.mkdir(exist_ok=False)
    write(a.out/'protocol.json',dict(paper='https://arxiv.org/html/1511.05547v2',section='3.2 Algorithm1',covariance_regularizer=1,ridge_regularizer=10,ridge_objective='mean squared Frobenius residual +10||W||²',source_trials=9000,target_labels=0,readout='source-labelled Ridge replaces original projection; compare identical no-CORAL source Ridge',budgets=[50,200],seeds=list(range(33,38)),preprocessing='feature centering, no dimensionwise scale; source->target CORAL; target test untouched'))
    for f in [1,2]:
        for arm in ['baseline','identity']:
            c,_,_,labels,meta=load(f,arm);_,_,gallery,_=native(f,arm,c,meta)
            src=torch.load(a.source/f's{f}_{arm}_source.pt',map_location='cpu',weights_only=False)
            assert src['checkpoint_sha256']==meta['checkpoint_sha256']
            x=src['features'].double();ms=x.mean(0);xc=x-ms;cs=cov(x)+torch.eye(512);invs=power(cs,-.5)
            cfg=json.loads((Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/identity_reference_loso_pilot')/f'fold_{f:02d}_{arm}'/'config.json').read_text())
            state=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['model']
            image=torch.nn.Sequential(torch.nn.Linear(1536,768),torch.nn.ELU(),torch.nn.Dropout(0),torch.nn.Linear(768,1536))
            image.load_state_dict({k.removeprefix('image_adapter.'):v for k,v in state.items() if k.startswith('image_adapter.')});image.eval().requires_grad_(False)
            raw=np.load(Path(cfg['source_root'])/'features/blur.npy',mmap_mode='r')[src['protocol']['images']].copy()
            with torch.no_grad():y=F.normalize(image(torch.from_numpy(raw)),dim=-1).repeat(9,1)
            xm,w,b=fit(x,y);base=(c['etest'].double()-xm)@w+b;base_score=evaluate_prediction(base,labels,gallery)
            torch.save(dict(mu=xm,w=w,bias=b),a.out/f's{f}_{arm}_source_ridge.pt')
            for n in [50,200]:
                for seed in range(33,38):
                    indices=np.random.default_rng(seed).permutation(1000)[:n];target=c['etrain'][indices].double();mt=target.mean(0);ct=cov(target)+torch.eye(512)
                    mapping=invs@power(ct,.5);torch.testing.assert_close(mapping.T@cs@mapping,ct,rtol=1e-5,atol=1e-5)
                    adjusted=xc@mapping+mt;xm,w,b=fit(adjusted,y)
                    pred=(c['etest'].double()-xm)@w+b
                    scores=evaluate_prediction(pred,labels,gallery)
                    write(a.out/f's{f}_{arm}_n{n}_seed{seed}.json',dict(zero=base_score,methods={'source_ridge_no_coral':{'metrics':base_score},'forward_coral_source_ridge':{'metrics':scores}},support_indices=indices.tolist(),target_labels_used=False))
                    torch.save(dict(mu=xm,w=w,bias=b,mapping=mapping),a.out/f's{f}_{arm}_n{n}_seed{seed}.pt');print('DONE',f,arm,n,seed,flush=True)
    write(a.out/'status.json',dict(state='complete'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except Exception as exc:
        if a.out.exists():write(a.out/'status.json',dict(state='failed',error=repr(exc)))
        raise
