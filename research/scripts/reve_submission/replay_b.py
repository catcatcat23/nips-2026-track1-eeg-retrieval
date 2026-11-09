import sys,json,importlib.util,zipfile,tempfile
from pathlib import Path
import torch,numpy as np
from torch.utils.data import DataLoader
p=Path('/home/Anteng/eeg-retrieval');sys.path[:0]=[str(p/'tools/eeg_multiblur'),str(p/'competition-2026')]
from data import DatasetEEG
from recovery_ablation import metrics
folder=Path(tempfile.mkdtemp(prefix='b_zip_real_'))
with zipfile.ZipFile(p/'submissions/reve_b_epoch6_track1.zip') as z:z.extractall(folder)
spec=importlib.util.spec_from_file_location('b_zip',folder/'submission.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
base=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v13_reference');manifest=json.loads((base/'data/manifest.json').read_text())
meta=dict(n_chans=63,n_times=120,sfreq=120.,ch_names=manifest['channels'],n_outputs=1536,submission_dir=folder,device='cuda')
# Reject network access during both weight loading and the entire real-data prediction.
import socket
original_connect=socket.socket.connect
def guarded_connect(sock,address):
 if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('Network disabled')
 return original_connect(sock,address)
socket.socket.connect=guarded_connect
torch.backends.cuda.matmul.allow_tf32=False
model=mod.Solver().load_model(meta);ds=DatasetEEG(base,'test',aux=False);out=[]
for i,b in enumerate(DataLoader(ds,batch_size=256,num_workers=2)):
 out.append(model.predict(b['eeg'].cuda()).cpu().numpy())
 if i%200==0:print('REPLAY',i,flush=True)
r=metrics(np.concatenate(out),ds.ids,ds.subjects,np.asarray(ds.clean))
expected=json.loads(Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/recovery_v15_frozen_v13/B/result.json').read_text())['selection']
for g in ['single_trial','subject_aggregated']:
 for k in ['top1','top5']:assert abs(r[g][k]-expected[g][k])<1e-8,(g,k,r[g][k],expected[g][k])
(p/'submissions/reve_b_epoch6_track1.real_replay.json').write_text(json.dumps({'verified':True,'metrics':r,'network':'disabled','artifact':'actual ZIP extracted','scope':'local full public test replay, not platform container'},indent=2))
print('FULL REAL ZIP REPLAY PASSED',flush=True)
