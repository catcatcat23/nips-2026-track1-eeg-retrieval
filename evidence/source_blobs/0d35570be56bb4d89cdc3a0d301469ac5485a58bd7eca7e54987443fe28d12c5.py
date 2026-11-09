"""Matched REVE comparison: competition 120 Hz inputs, identical inference adapter."""
import torch
from torch import nn
from torch.nn import functional as F

class ReferenceBase(nn.Module):
    def __init__(self,meta,variant):
        super().__init__()
        from braindecode.models import REVE
        assert meta['sfreq']==120 and meta['n_times']==120
        assert variant in ('resample200','native120')
        self.variant=variant
        self.encoder=REVE.from_pretrained('brain-bzh/reve-base',n_outputs=1,
            n_chans=meta['n_chans'],n_times=200,sfreq=200,chs_info=None)
        names=[str(n).removeprefix('EEG ').split('-',1)[0] for n in meta['ch_names']]
        self.register_buffer('positions',self.encoder.get_positions(names))
        self.encoder.final_layer=nn.Identity() # unused pretrained classifier
        self.head=nn.Linear(self.encoder.embed_dim,1536)
        if variant=='native120':
            old=self.encoder.to_patch_embedding[0]
            new=nn.Linear(120,old.out_features,bias=old.bias is not None)
            with torch.no_grad():
                new.weight.copy_(F.interpolate(old.weight[:,None],size=120,mode='linear',align_corners=False)[:,0]*(200/120))
                if old.bias is not None:new.bias.copy_(old.bias)
            self.encoder.to_patch_embedding[0]=new
            self.encoder.patch_size=120
            self.encoder.patch_overlap=12
        self.encoder.requires_grad_(True)

    def preprocess(self,x):
        assert x.ndim==3 and x.shape[-1]==120
        if self.variant=='resample200':
            x=F.interpolate(x,size=200,mode='linear',align_corners=False)
        return ((x-x.mean(-1,keepdim=True))/x.std(-1,keepdim=True,unbiased=False).clamp_min(1e-6)).clamp(-15,15)

    def forward(self,x):
        x=self.preprocess(x)
        pos=self.positions.to(x).unsqueeze(0).expand(len(x),-1,-1)
        return self.head(self.encoder(x,pos=pos,return_output=True)[-1].mean(1))


class ReferenceModel(ReferenceBase):
    """94b3a31 serial model; forward includes all trainable branches for DDP."""
    def __init__(self, manifest):
        meta={'sfreq':120,'n_times':120,'n_chans':len(manifest['channels']),'ch_names':manifest['channels']}
        super().__init__(meta, 'resample200')
        self.blur_head=self.head
        del self.head
        self.image_adapter=nn.Sequential(nn.Linear(1536,768),nn.ELU(),nn.Dropout(.85),nn.Linear(768,1536))
        self.recovery=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,768),nn.GELU(),nn.Linear(768,1536))

    def forward(self, x, blur=None, local=None):
        x=self.preprocess(x)
        pos=self.positions.to(x).unsqueeze(0).expand(len(x),-1,-1)
        h=self.encoder(x,pos=pos,return_output=True)[-1].mean(1)
        pred=self.blur_head(h)
        recovered=pred+self.recovery(pred)
        if blur is None:return recovered
        return pred,recovered,F.normalize(self.image_adapter(blur),dim=-1)
