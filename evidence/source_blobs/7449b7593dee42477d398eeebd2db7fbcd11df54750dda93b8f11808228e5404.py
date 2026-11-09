import sys,json
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
from torch.utils.data import default_collate
from dino_image_ablation import UPSTREAM
code=UPSTREAM;out=Path(sys.argv[1]);sys.path.insert(0,str(code));sys.path.insert(0,str(code/'scripts'))
from train_global_mean_baseline import GlobalMeanEncoder,CHANNELS,LEVEL_KEYS
from main_eeg import get_dataset,set_seed,ClipLoss
set_seed(21)
root=str(out/'data')
train,val,test=get_dataset(root,1,False,CHANNELS,False,.1,50.,[0,250],'MultiBlur_RN50',None)
report={'counts':[len(train),len(val),len(test)],'eeg_shapes':[list(d.eeg_data.shape) for d in [train,val,test]]}
assert report['counts']==[15713,827,200]
keysets=[]
for ds in [train,val,test]:
 keys=[str(k).replace('\\','/') for k in ds.index_dict];assert len(set(keys))==len(keys);keysets.append(set(keys))
 assert set(ds.blur_features)=={k[2:] for k in LEVEL_KEYS}
 for level,features in ds.blur_features.items():
  assert set(keys)<=set(features)
  x=torch.stack([features[k] for k in keys]);assert x.shape==(len(ds),1536) and torch.isfinite(x).all()
 assert torch.isfinite(ds.eeg_data).all()
assert not keysets[0]&keysets[1] and not (keysets[0]|keysets[1])&keysets[2]
b=default_collate([train[i] for i in range(8)])
eeg=b['eeg'].cuda();images=torch.cat([b[k][:,None].cuda() for k in LEVEL_KEYS],1)
m=GlobalMeanEncoder(len(CHANNELS),1536,250).cuda()
for layer in m.modules():
 if isinstance(layer,torch.nn.Dropout):layer.p=0
opt=torch.optim.AdamW(m.parameters(),lr=.001);crit=ClipLoss();initial=None
for step in range(300):
 opt.zero_grad(set_to_none=True);e=m(eeg);v=F.normalize(m.get_image_feature(images),dim=-1);a,b=crit(e,v,1);loss=(a.mean()+b.mean())/2
 if initial is None:initial=loss.item()
 assert torch.isfinite(loss);loss.backward();opt.step()
 if loss.item()<.05:break
m.eval()
with torch.no_grad():
 sim=F.normalize(m(eeg),dim=-1)@F.normalize(m.get_image_feature(images),dim=-1).T
 acc=(sim.argmax(1)==torch.arange(8,device='cuda')).float().mean().item()
assert acc==1 and loss.item()<initial*.2
report.update(passed=True,tiny_top1=acc,initial_loss=initial,final_loss=loss.item(),steps=step+1)
(out/'preflight.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
