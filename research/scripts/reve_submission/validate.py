from pathlib import Path
import importlib.util,json,sys,torch,numpy as np
project=Path('/home/Anteng/eeg-retrieval');sys.path.insert(0,str(project/'competition-2026'))
folder=project/'submissions/reve_seed33'
spec=importlib.util.spec_from_file_location('reve_submission',folder/'submission.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from braindecode.models import REVE
meta_info=json.loads(Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/multiblur_v1/data/manifest.json').read_text())
names=meta_info['channels'];meta=dict(n_chans=63,n_times=120,sfreq=120.,ch_names=names,n_outputs=1536,submission_dir=folder,device='cuda')
model=module.Solver().load_model(meta)
c=torch.load(json.loads((folder/'metadata.json').read_text())['checkpoint'],map_location='cpu',weights_only=True)['state_dict']
ref=REVE(n_chans=63,n_times=200,n_outputs=2).cuda().eval()
prefix='model.wrapped_model.model.';ref.load_state_dict({k[len(prefix):]:v for k,v in c.items() if k.startswith(prefix)},strict=True)
head=torch.nn.Linear(512,1536).cuda();head.load_state_dict({k.removeprefix('model.probe.'):v for k,v in c.items() if k.startswith('model.probe.')})
x=torch.tensor(np.load('/mnt/DATA-4/anteng/eeg-retrieval/experiments/multiblur_v1/data/train_eeg.npy',mmap_mode='r')[:4].copy(),device='cuda')
with torch.inference_mode():
 expected=head(ref(x,pos=c['model.wrapped_model.bank_positions'].cuda().expand(len(x),-1,-1),return_output=True)[-1].mean(1))
 actual=model(x);error=(actual-expected).abs().max().item();assert torch.allclose(actual,expected,rtol=1e-5,atol=1e-5),error
 del ref,head
 # Network-denial gate verifies load/predict need no download.
 import socket
 original=socket.socket.connect
 def no_network(*args,**kwargs):raise RuntimeError('Network forbidden in submission validation')
 socket.socket.connect=no_network
 try:
  model=module.Solver().load_model(meta)
  inputs=torch.randn(3,63,120,device='cuda')
  y=model.predict(inputs);assert y.shape==(3,1536) and torch.isfinite(y).all()
  single=torch.cat([model.predict(a[None]) for a in inputs])
  assert torch.allclose(y,single,rtol=1e-4,atol=1e-4)
  names32=names[:32];m32=module.Solver().load_model(dict(meta,n_chans=32,ch_names=names32))
  assert m32.predict(torch.randn(1,32,120,device='cuda')).shape==(1,1536)
  z=model.predict(torch.zeros(1,63,120,device='cuda'));assert torch.isfinite(z).all()
 finally:socket.socket.connect=original
report={'architecture_max_abs_error_on_real_preprocessed_EEG':error,'warmup_63ch_120samples':'passed','variable_batch_sizes':[1,3],'32_channel_shape_check':'passed (synthetic; not sealed-data validation)','network_disabled':'passed','constant_input_finite':'passed','submission_weights':'strict state_dict load','platform_submission':'not uploaded','preprocessing_caveat':'window resampling/standardization is a bridge; does not reproduce recording-level training preprocessing'}
(folder/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
