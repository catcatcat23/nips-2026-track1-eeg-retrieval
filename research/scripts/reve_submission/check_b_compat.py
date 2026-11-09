import sys,json,importlib.util,socket
from pathlib import Path
import torch,numpy as np
p=Path('/home/Anteng/eeg-retrieval');sys.path.insert(0,str(p/'competition-2026'))
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
meta=json.load(open('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v13_reference/data/manifest.json'))
names=meta['channels']
def load(folder,device,names=names):
 spec=importlib.util.spec_from_file_location(folder.name,folder/'submission.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.Solver().load_model(dict(n_chans=len(names),n_times=120,sfreq=120.,ch_names=names,n_outputs=1536,submission_dir=folder,device=device))
source=p/'submissions/reve_b_epoch6';folder=p/'submissions/reve_b_epoch6_compat_v2'
def blocked(*a,**k):raise RuntimeError('network forbidden')
socket.socket.connect=blocked
old=load(source,'cuda');new=load(folder,'cuda')
x=torch.tensor(np.load('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v13_reference/data/test_eeg.npy',mmap_mode='r')[:257].copy(),device='cuda')
with torch.inference_mode():
 expected=old.predict(x);actual=new.predict(x);torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
 perm=torch.randperm(len(x),device='cuda');torch.testing.assert_close(new.predict(x[perm]),actual[perm],rtol=1e-4,atol=1e-4)
 torch.testing.assert_close(new.predict(x[:1]),actual[:1],rtol=1e-4,atol=1e-4)
 assert new.predict(x[:0]).shape==(0,1536)
 cpu=load(folder,'cpu');torch.testing.assert_close(cpu.predict(x[:3].cpu()),actual[:3].cpu(),rtol=1e-4,atol=1e-4)
 synthetic=load(folder,'cpu',names[:32]);assert synthetic.predict(torch.zeros(2,32,120)).shape==(2,1536)
report={'weights':'unchanged','real_samples':257,'max_abs_error':float((actual-expected).abs().max()),'batch_permutation_invariance':'passed','CPU_GPU_equivalence':'passed','32_channel_synthetic':'passed','network':'blocked','empty_batch':'passed','environment':'local, not platform Docker'}
(folder/'compatibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
