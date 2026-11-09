"""Export the frozen v33 encoder plus v39-B ridge/residual recovery."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
root=Path('/home/Anteng/eeg-retrieval')
sys.path[:0]=[str(root/'tools/eeg_multiblur'),str(root/'competition-2026')]
from v33_recovery_stage2 import load_source, encode
from ridge_recovery import RidgeResidual
exp=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
run=exp/'reve_v39_ridge_recovery'
headpath=run/'B/best.pt'
basepath=exp/'reve_v33_b_stability/dropout/best.pt'
cfg=json.loads((run/'config.json').read_text())
assert hashlib.sha256(basepath.read_bytes()).hexdigest()==cfg['checkpoint_sha256']
base=torch.load(basepath,map_location='cpu',weights_only=False)
ck=torch.load(headpath,map_location='cpu',weights_only=False)
assert ck['epoch']==8 and ck['selection']['subject_aggregated']['top5']==.9305
out=root/'submissions/reve_v39b_epoch8';out.mkdir(exist_ok=True)
src=(root/'submissions/reve_b_epoch6_compat_v2/submission.py').read_text()
start=src.index('class ProjectedHead');end=src.index('class ReveRetrieval',start)
block='''class RidgeResidual(nn.Module):
    def __init__(self):
        super().__init__()
        for name, shape in [('mean',(1536,)),('scale',(1536,)),('weight',(1536,1536)),('bias',(1536,))]:
            self.register_buffer(name, torch.zeros(shape))
        self.delta=nn.Sequential(nn.LayerNorm(1536),ResidualBlock(),ResidualBlock(),nn.LayerNorm(1536),nn.Linear(1536,1536))
    def forward(self,h):
        return ((h-self.mean)/self.scale)@self.weight+self.bias+0.1*self.delta(h)


'''
src=src[:start]+block+src[end:]
src=src.replace('self.recovery = ProjectedHead()','self.recovery = RidgeResidual()').replace('REVE-B-Epoch6-CompatV2','REVE-v39B-RidgeResidual-Epoch8')
(out/'submission.py').write_text(src)
weights={k.removeprefix('encoder.'):v for k,v in base['model'].items() if k.startswith('encoder.')}
weights.update({'head.'+k.removeprefix('blur_head.'):v for k,v in base['model'].items() if k.startswith('blur_head.')})
weights.update({'recovery.'+k:v for k,v in ck['head'].items()})
torch.save(weights,out/'weights.pt')
for name in ['positions.json','LICENSE_BRAINDECODE.txt']:
 (out/name).write_bytes((root/'submissions/reve_b_epoch6_compat_v2'/name).read_bytes())
spec=importlib.util.spec_from_file_location('export_v39',out/'submission.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
meta=dict(n_chans=63,n_times=120,sfreq=120.,ch_names=base['manifest']['channels'],n_outputs=1536,submission_dir=out,device='cuda')
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
source=load_source(dict(checkpoint=str(basepath),checkpoint_sha256=cfg['checkpoint_sha256']))
ridge={k:ck['head'][k] for k in ['mean','scale','weight','bias']}
head=RidgeResidual(ridge).cuda().eval();head.load_state_dict(ck['head'])
import socket
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled'))
model=mod.Solver().load_model(meta)
x=torch.tensor(np.load(exp/'reve_v13_reference/data/test_eeg.npy',mmap_mode='r')[:65].copy(),device='cuda')
with torch.inference_mode():
 expected=head(encode(source,x));actual=model.predict(x)
 torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
 perm=torch.randperm(len(x),device='cuda');torch.testing.assert_close(model.predict(x[perm]),actual[perm],rtol=1e-4,atol=1e-4)
 torch.testing.assert_close(model.predict(x[:1]),actual[:1],rtol=1e-4,atol=1e-4)
 assert model.predict(x[:0]).shape==(0,1536)
 cpu=mod.Solver().load_model(dict(meta,device='cpu'))
 torch.testing.assert_close(cpu.predict(x[:2].cpu()),actual[:2].cpu(),rtol=1e-4,atol=1e-4)
 other=mod.Solver().load_model(dict(meta,device='cpu',n_chans=32,ch_names=meta['ch_names'][:32]))
 assert other.predict(torch.zeros(1,32,120)).shape==(1,1536)
report=dict(encoder_sha256=cfg['checkpoint_sha256'],head_sha256=hashlib.sha256(headpath.read_bytes()).hexdigest(),epoch=8,selection=ck['selection'],max_abs_error=float((actual-expected).abs().max()),offline=True,batch_invariance=True,cpu_gpu=True,channels32_synthetic=True,uploaded=False)
(out/'validation.json').write_text(json.dumps(report,indent=2))
print('V39B build verified',report['max_abs_error'],flush=True)
