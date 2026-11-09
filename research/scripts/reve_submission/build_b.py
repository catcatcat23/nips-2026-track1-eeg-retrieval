"""Build offline EEG-only B epoch6 package; verify exact prediction equivalence."""
import json,sys,hashlib,importlib.util
from pathlib import Path
import torch,numpy as np
root=Path('/home/Anteng/eeg-retrieval');sys.path[:0]=[str(root/'tools/eeg_multiblur'),str(root/'competition-2026')]
from reference_model import ReferenceModel
from recovery_ablation import ProjectedHead
out=root/'submissions/reve_b_epoch6';out.mkdir(exist_ok=True)
basepath=root/'audits/blur_head_probe/v13_best.pt'
headpath=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/recovery_v15_frozen_v13/B/best.pt')
base=torch.load(basepath,map_location='cpu',weights_only=False);b=torch.load(headpath,map_location='cpu',weights_only=False)
assert b['source_checkpoint_sha256']==hashlib.sha256(basepath.read_bytes()).hexdigest() and b['epoch']==6
src=(root/'submissions/reve_seed33/submission.py').read_text()
blocks=(root/'tools/eeg_multiblur/recovery_ablation.py').read_text();blocks=blocks[blocks.index('class ResidualBlock'):blocks.index('def make_head')]
src=src.replace('class ReveRetrieval(nn.Module):',blocks+'\nclass ReveRetrieval(nn.Module):')
src=src.replace('self.head = nn.Linear(512,1536)','self.head = nn.Linear(512,1536)\n        self.recovery = ProjectedHead()')
src=src.replace('return self.head(self.transformer(tokens).mean(dim=1))','return self.recovery(self.head(self.transformer(tokens).mean(dim=1)))')
src=src.replace("name = 'REVE-NeuralBench-Finetuned-Seed33'","name = 'REVE-B-ProjectedRecovery-Epoch6'")
(out/'submission.py').write_text(src)
weights={k.removeprefix('encoder.'):v for k,v in base['model'].items() if k.startswith('encoder.')}
weights.update({'head.'+k.removeprefix('blur_head.'):v for k,v in base['model'].items() if k.startswith('blur_head.')})
weights.update({'recovery.'+k:v for k,v in b['head'].items()})
torch.save(weights,out/'weights.pt')
for name in ['positions.json','LICENSE_BRAINDECODE.txt']:(out/name).write_bytes((root/'submissions/reve_seed33'/name).read_bytes())
spec=importlib.util.spec_from_file_location('submission_b',out/'submission.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
manifest=base['manifest'];meta=dict(n_chans=63,n_times=120,sfreq=120.,ch_names=manifest['channels'],n_outputs=1536,submission_dir=out,device='cuda')
model=mod.Solver().load_model(meta)
reference=ReferenceModel(manifest).cuda().eval();reference.load_state_dict(base['model']);recovery=ProjectedHead().cuda().eval();recovery.load_state_dict(b['head'])
captured=[]
hook=reference.blur_head.register_forward_hook(lambda m,i,o:captured.append(o))
x=torch.tensor(np.load('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v13_reference/data/test_eeg.npy',mmap_mode='r')[:16].copy(),device='cuda')
torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode():
 reference(x);expected=recovery(captured.pop());actual=model.predict(x)
 torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
 singles=torch.cat([model.predict(i[None]) for i in x]);torch.testing.assert_close(actual,singles,rtol=1e-4,atol=1e-4)
 perm=torch.randperm(len(x),device='cuda');torch.testing.assert_close(model.predict(x[perm]),actual[perm],rtol=1e-4,atol=1e-4)
hook.remove()
report={'encoder_sha256':hashlib.sha256(basepath.read_bytes()).hexdigest(),'head_sha256':hashlib.sha256(headpath.read_bytes()).hexdigest(),'head_epoch':b['epoch'],'max_abs_error':float((actual-expected).abs().max()),'batch_permutation_invariance':'passed','selection':b['selection'],'inputs':'EEG only, no candidates or image adapter','training':'train+val; public test used for checkpoint selection','uploaded':False}
(out/'validation.json').write_text(json.dumps(report,indent=2));print('B BUILD VERIFIED',report['max_abs_error'],flush=True)
