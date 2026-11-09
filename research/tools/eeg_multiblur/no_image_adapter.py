"""DINOv2 control: only replace the original image adapter with Identity."""
import argparse,json,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate
from dino_image_ablation import UPSTREAM
sys.path.insert(0,str(UPSTREAM));sys.path.insert(0,str(UPSTREAM/'scripts'))
import train_global_mean_baseline as upstream
from main_eeg import get_dataset,set_seed,ClipLoss
Base=upstream.GlobalMeanEncoder
DATA=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_dinov2_mean_sub01_seed21/data')

class NoImageAdapter(Base):
    def __init__(self,channels,proj_dim,temporal_len):
        # Construct the same modules first to preserve EEG initialization/RNG state.
        super().__init__(channels,1536,temporal_len)
        self.img_adapter=nn.Identity()


def preflight(root):
    set_seed(21)
    train,val,test=get_dataset(str(DATA),1,False,upstream.CHANNELS,False,.1,50.,[0,250],'MultiBlur_RN50',None)
    assert [len(train),len(val),len(test)]==[15713,827,200]
    rng=torch.get_rng_state();baseline=Base(63,1536,250)
    torch.set_rng_state(rng);model=NoImageAdapter(63,1536,250)
    for name,param in model.state_dict().items():assert torch.equal(param,baseline.state_dict()[name]),name
    del baseline
    b=default_collate([train[i] for i in range(8)])
    eeg=b['eeg'].cuda();images=torch.cat([b[k][:,None].cuda() for k in upstream.LEVEL_KEYS],1)
    model.cuda().train()
    assert torch.equal(model.get_image_feature(images),images.mean(1))
    assert not list(model.img_adapter.parameters())
    for layer in model.modules():
        if isinstance(layer,nn.Dropout):layer.p=0
    opt=torch.optim.AdamW(model.parameters(),lr=.001);criterion=ClipLoss();initial=None
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        h=model(eeg);target=F.normalize(model.get_image_feature(images),dim=-1)
        a,b=criterion(h,target,1);loss=(a.mean()+b.mean())/2
        if initial is None:initial=loss.item()
        assert torch.isfinite(loss);loss.backward()
        for name,param in model.named_parameters():
            if name.startswith(('eeg_encoder.','spatial_conv.','bn.','eeg_adapter.')):
                assert param.grad is not None and torch.isfinite(param.grad).all(),name
        opt.step()
        if loss.item()<.05:break
    model.eval()
    with torch.no_grad():
        scores=F.normalize(model(eeg),dim=-1)@target.T
        accuracy=(scores.argmax(1)==torch.arange(8,device='cuda')).float().mean().item()
    assert accuracy==1 and loss.item()<initial*.2,(accuracy,loss.item())
    report={'passed':True,'eeg_initialization_matches_adapter_control':True,'image_output_equals_raw_scale_mean':True,'image_adapter_parameters':0,'samples':8,'top1':accuracy,'initial_loss':initial,'final_loss':loss.item(),'steps':step+1,'dropout_disabled_only_for_preflight':True}
    (root/'preflight.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


def train(root):
    upstream.GlobalMeanEncoder=NoImageAdapter
    args=argparse.Namespace(data_path=str(DATA),subject=1,seed=21,epochs=200,batch_size=1024,lr=.001,device='cuda:0',output_dir=str(root/'results'),ablation='image adapter replaced by Identity; EEG initialization preserved')
    upstream.train(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','train']);p.add_argument('--root',required=True);a=p.parse_args()
    (preflight if a.stage=='preflight' else train)(Path(a.root))
