"""CPU equivalence against the checked-out 94b3a31 reference, no pretrained downloads."""
import importlib.util,os,sys,tempfile,types,json
REAL="--real" in sys.argv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from upstream_blur import load_blurred,LEVELS
from losses import objective
from neuraltrain.losses.losses import ClipLoss

REF=Path(__file__).resolve().parents[2]/'reve-multiblur/tools'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);im=root/'test.png';Image.fromarray(np.random.default_rng(4).integers(0,256,(317,529,3),dtype=np.uint8)).save(im)
    upstream=load('upstream',REF/'eeg_multiblur/upstream_blur.py')
    for k in LEVELS:np.testing.assert_array_equal(np.asarray(load_blurred(im,k)),np.asarray(upstream.load_blurred(im,k)))
    # Fake only the pretrained backbone; compare exact model initialization and trainable forward code.
    class Encoder(nn.Module):
        embed_dim=512
        def __init__(self):
            super().__init__();self.proj=nn.Linear(200,512);self.final_layer=nn.Linear(512,1)
        @classmethod
        def from_pretrained(cls,*a,**kw):return cls()
        def get_positions(self,names):return torch.zeros(len(names),3)
        def forward(self,x,**kw):return [self.proj(x)]
    if not REAL:
        fake=types.ModuleType('braindecode.models');fake.REVE=Encoder;sys.modules['braindecode.models']=fake
    base=root/'archive/tools/eeg_reve_ab';base.mkdir(parents=True)
    (base/'model.py').write_text((REF/'eeg_reve_ab/model.py').read_text())
    os.environ['EEG_ROOT']=tmp;sys.path.insert(0,str(REF/'eeg_reve_blur'))
    original=load('original_serial',REF/'eeg_reve_blur/model.py')
    from reference_model import ReferenceModel
    manifest={'channels':['Fp1','Fp2']};meta={'sfreq':120,'n_times':120,'n_chans':2,'ch_names':manifest['channels']}
    if REAL:
        data=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/multiblur_v4_compet120/data')
        manifest=json.loads((data/'manifest.json').read_text());meta.update(n_chans=len(manifest['channels']),ch_names=manifest['channels'])
    torch.manual_seed(33);a=original.Model(meta)
    torch.manual_seed(33);b=ReferenceModel(manifest)
    for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
    x=torch.randn(4,2,120);clean=torch.randn(4,1536);blur=torch.randn(4,1536);blur[1]=blur[0];clean[1]=clean[0]
    if REAL:
        a.cuda();b.cuda();x=torch.from_numpy(np.load(data/'train_eeg.npy',mmap_mode='r')[:4].copy()).cuda();clean=clean.cuda();blur=blur.cuda()
    criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False)
    torch.manual_seed(9);la=original.training_loss(a,x,torch.cat([clean,blur],-1),criterion)
    torch.manual_seed(9);lb,_=objective(b(x,blur),clean,blur,torch.tensor([0,0,2,3]),None,{'recipe':'reference_94b3a31'},distributed=False)
    torch.testing.assert_close(la,lb,rtol=0,atol=0);la.backward();lb.backward()
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb
        torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
    a.eval();b.eval();torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
print('PASS: all 12 blur pixels, initialization, train loss, gradients, eval forward match 94b3a31; real pretrained REVE + EEG:',REAL)
