"""Export v64 stage1 EMA checkpoint (encoder + blur_head + jointly trained ProjectedHead recovery) in the compat_v2 format.

No stage2: the stage1 recovery head is the clean-space output. Local OT / image adapter weights are dropped.
Verifies exact prediction equivalence with the training model on real test EEG.
"""
import argparse,hashlib,importlib.util,json,shutil,socket,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/home/Anteng/eeg-retrieval');exp=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
sys.path[:0]=[str(exp/'reve_v64_stage1_ema/source'),str(root/'competition-2026')]
from stability_ablation import AblationModel
parser=argparse.ArgumentParser();parser.add_argument('--variant',default='ema0.999');parser.add_argument('--epoch',type=int,default=9)
args=parser.parse_args()
tag=f"reve_v64_{args.variant.replace('.','')}_ep{args.epoch}"
out=Path('/mnt/DATA-4/anteng/eeg-retrieval/submissions')/tag;out.mkdir(parents=True)
template=root/'submissions/reve_b_epoch6_compat_v2'
src=exp/f'reve_v64_stage1_ema/stage1/{args.variant}_epoch_{args.epoch:03d}.pt'
ck=torch.load(src,map_location='cpu',weights_only=False);assert ck['epoch']==args.epoch
s=(template/'submission.py').read_text()
assert s.count("name = 'REVE-B-Epoch6-CompatV2'")==1
s=s.replace("name = 'REVE-B-Epoch6-CompatV2'",f"name = 'REVE-v64-{args.variant}-Epoch{args.epoch}'")
(out/'submission.py').write_text(s)
for name in ['positions.json','LICENSE_BRAINDECODE.txt']:shutil.copyfile(template/name,out/name)
m=ck['model']
weights={k.removeprefix('encoder.'):v for k,v in m.items() if k.startswith('encoder.')}
weights.update({'head.'+k.removeprefix('blur_head.'):v for k,v in m.items() if k.startswith('blur_head.')})
weights.update({k:v for k,v in m.items() if k.startswith('recovery.')})
reference_keys=set(torch.load(template/'weights.pt',map_location='cpu',weights_only=True))
assert set(weights)==reference_keys,(sorted(set(weights)^reference_keys)[:10])
torch.save(weights,out/'weights.pt')
# Reference: training-code model restricted to the global (non-OT) keys, eval mode.
manifest=ck['manifest'];reference=AblationModel(manifest,'dropout')
reference.load_state_dict({k:m[k] for k in reference.state_dict()},strict=True);reference=reference.cuda().eval()
socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network forbidden'))
spec=importlib.util.spec_from_file_location(tag,out/'submission.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
meta=dict(n_chans=63,n_times=120,sfreq=120.,ch_names=manifest['channels'],n_outputs=1536,submission_dir=out,device='cuda')
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=mod.Solver().load_model(meta)
x=torch.tensor(np.load(exp/'reve_v13_reference/data/test_eeg.npy',mmap_mode='r')[:257].copy(),device='cuda')
with torch.inference_mode():
 expected=reference(x);actual=model.predict(x);torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
 perm=torch.randperm(len(x),device='cuda');torch.testing.assert_close(model.predict(x[perm]),actual[perm],rtol=1e-4,atol=1e-4)
 assert model.predict(x[:0]).shape==(0,1536)
 cpu=mod.Solver().load_model(dict(meta,device='cpu'));torch.testing.assert_close(cpu.predict(x[:3].cpu()),actual[:3].cpu(),rtol=1e-4,atol=1e-4)
 synthetic=mod.Solver().load_model(dict(meta,device='cpu',n_chans=32,ch_names=manifest['channels'][:32]));assert synthetic.predict(torch.zeros(2,32,120)).shape==(2,1536)
hist={json.loads(l)['epoch']:json.loads(l) for l in open(exp/'reve_v64_stage1_ema/stage1/history.jsonl')}
report=dict(source_checkpoint=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),weights_sha256=hashlib.sha256((out/'weights.pt').read_bytes()).hexdigest(),
 template='reve_b_epoch6_compat_v2 (submission.py unchanged except Solver.name)',dropped=sorted(k for k in m if k.split('.')[0] not in ('encoder','blur_head','recovery')),
 real_samples=257,max_abs_error=float((actual-expected).abs().max()),batch_permutation_invariance='passed',CPU_GPU_equivalence='passed',empty_batch='passed',
 synthetic_32_channel='passed',network='blocked',expected_clean=hist[args.epoch]['metrics'][args.variant]['clean'],
 role='public test model selection; THINGS-EEG2 development score, not hidden Alljoined estimate')
(out/'build.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['max_abs_error','weights_sha256','dropped']}),flush=True)
link=root/'submissions'/tag
if not link.exists():link.symlink_to(out)
print(out)
