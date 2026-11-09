"""Single-trial EEG encoder. Architecture adapted from upstream MIT code."""
import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self,channels,time,dropout=(.25,.65),use_local=True):
        super().__init__()
        self.spatial=nn.Parameter(torch.empty(25,channels));nn.init.xavier_uniform_(self.spatial)
        self.bn=nn.BatchNorm1d(25)
        self.temporal=nn.Sequential(nn.Linear(time,200),nn.ELU(),nn.Dropout(dropout[0]),nn.Linear(200,200),nn.ELU(),nn.Dropout(dropout[1]))
        self.global_head=nn.Linear(5000,1536)
        self.restore=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,768),nn.GELU(),nn.Linear(768,1536))
        nn.init.zeros_(self.restore[-1].weight);nn.init.zeros_(self.restore[-1].bias)
        self.use_local=use_local
        self.local_projection=nn.Linear(200,256)
        self.eeg_positions=nn.Parameter(torch.randn(1,25,256)*.02)
        self.queries=nn.Parameter(torch.randn(1,6,256)*.02)
        self.attention=nn.MultiheadAttention(256,4,dropout=.1,batch_first=True)
        self.local_norm=nn.LayerNorm(256)
        self.image_projection=nn.Sequential(nn.Linear(1536,256),nn.GELU(),nn.LayerNorm(256))
        self.image_positions=nn.Parameter(torch.randn(1,9,256)*.02)
        if not use_local:
            for name,parameter in self.named_parameters():
                if name.startswith(('local_projection','eeg_positions','queries','attention','local_norm','image_projection','image_positions')):
                    parameter.requires_grad_(False)
    def recover(self,x):return x+self.restore(x)
    def encode(self,eeg):
        shared=self.temporal(self.bn(torch.einsum('kc,bct->bkt',self.spatial.abs(),eeg)))
        h=self.global_head(shared.flatten(1))
        return shared,h,self.recover(h)
    def forward(self,eeg,blur=None,local=None):
        shared,h,z=self.encode(eeg)
        if blur is None:return z
        if not self.use_local:return h,z,self.recover(blur)
        keys=self.local_projection(shared)+self.eeg_positions
        q,_=self.attention(self.queries.expand(len(eeg),-1,-1),keys,keys,need_weights=False)
        q=F.normalize(self.local_norm(q),dim=-1)
        v=F.normalize(self.image_projection(local)+self.image_positions,dim=-1)
        return h,z,self.recover(blur),q,v
