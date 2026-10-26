"""Export a v64 stage1 checkpoint (encoder + blur_head + jointly trained ProjectedHead recovery) to submission/.

No stage2: the stage1 recovery head is the clean-space output. Local OT / image adapter weights are dropped.
Verifies prediction equivalence between the packaged Solver and the training-code model on real test EEG.
Adapted from the original build script; the packaged submission.py is the template.
"""
import argparse,hashlib,importlib.util,json,shutil,socket,sys
from pathlib import Path
import numpy as np
import torch
HOME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HOME/'src'))
from stability_ablation import AblationModel
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--competition-root',type=Path,required=True,help='checkout of neural-interfaces26/2026-competition (provides benchmark_utils)')
a=p.parse_args();sys.path.insert(0,str(a.competition_root))
out=a.output;out.mkdir(parents=True,exist_ok=False)
for name in ['submission.py','positions.json','LICENSE_BRAINDECODE.txt']:shutil.copyfile(HOME/'submission'/name,out/name)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=ck['model']
weights={k.removeprefix('encoder.'):v for k,v in m.items() if k.startswith('encoder.')}
weights.update({'head.'+k.removeprefix('blur_head.'):v for k,v in m.items() if k.startswith('blur_head.')})
weights.update({k:v for k,v in m.items() if k.startswith('recovery.')})
torch.save(weights,out/'weights.pt')
# Reference: training-code model restricted to the global (non-OT) keys, eval mode.
manifest=ck['manifest'];reference=AblationModel(manifest,'dropout')
reference.load_state_dict({k:m[k] for k in reference.state_dict()},strict=True);reference=reference.cuda().eval()
socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network forbidden'))
spec=importlib.util.spec_from_file_location('submission',out/'submission.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
meta=dict(n_chans=63,n_times=120,sfreq=120.,ch_names=manifest['channels'],n_outputs=1536,submission_dir=out,device='cuda')
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=mod.Solver().load_model(meta)
x=torch.tensor(np.load(a.data_root/'data/test_eeg.npy',mmap_mode='r')[:257].copy(),device='cuda')
with torch.inference_mode():
 expected=reference(x);actual=model.predict(x);torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
 perm=torch.randperm(len(x),device='cuda');torch.testing.assert_close(model.predict(x[perm]),actual[perm],rtol=1e-4,atol=1e-4)
 assert model.predict(x[:0]).shape==(0,1536)
report=dict(source_checkpoint=str(a.checkpoint),source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
 weights_sha256=hashlib.sha256((out/'weights.pt').read_bytes()).hexdigest(),max_abs_error=float((actual-expected).abs().max()),
 dropped=sorted({k.split('.')[0] for k in m if k.split('.')[0] not in ('encoder','blur_head','recovery')}))
(out/'build.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
