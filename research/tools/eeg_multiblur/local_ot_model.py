"""Add training-only regional matching to the v33 global model."""
import torch
from torch import nn
from torch.nn import functional as F
from stability_ablation import AblationModel
from losses import multipos,hard_mask
class LocalOTModel(AblationModel):
 def __init__(self,manifest):
  super().__init__(manifest,'dropout')
  self.local_projection=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256))
  self.queries=nn.Parameter(torch.randn(1,6,256)*.02)
  self.attention=nn.MultiheadAttention(256,4,batch_first=True,dropout=.1)
  self.local_norm=nn.LayerNorm(256)
  self.image_projection=nn.Sequential(nn.Linear(1536,256),nn.GELU(),nn.LayerNorm(256))
  self.image_positions=nn.Parameter(torch.randn(1,9,256)*.02)
 def encode_eeg(self,x):
  x=self.preprocess(x);pos=self.positions.to(x).unsqueeze(0).expand(len(x),-1,-1);tokens=self.encoder(x,pos=pos,return_output=True)[-1]
  h=self.blur_head(tokens.mean(1));z=self.recovery(h)
  keys=self.local_projection(tokens);q,_=self.attention(self.queries.expand(len(x),-1,-1),keys,keys,need_weights=False)
  return h,z,F.normalize(self.local_norm(q),dim=-1)
 def encode_image(self,blur,local):
  return F.normalize(self.image_adapter(blur),dim=-1),F.normalize(self.image_projection(local)+self.image_positions,dim=-1)
 def forward(self,x,blur=None,local=None):
  h,z,q=self.encode_eeg(x)
  if blur is None:return z
  a,v=self.encode_image(blur,local)
  return h,z,a,q,v

def pair_ot(q,v,weights):
 sim=torch.einsum('nqd,npd->nqp',q.float(),v.float())
 patch=weights.topk(5,dim=-1).indices
 sim=sim.gather(-1,patch[:,None,:].expand(-1,6,-1))
 qi=sim.max(-1).values.topk(3,dim=-1).indices
 sim=sim.gather(-2,qi[:,:,None].expand(-1,-1,5))
 pw=weights.gather(-1,patch);pw=pw/pw.sum(-1,keepdim=True)
 kernel=sim/.1
 u=torch.zeros_like(sim[:,:,0]);w=torch.zeros_like(sim[:,0,:]);la=-sim.new_tensor(sim.shape[1]).log();lb=pw.log()
 for _ in range(6):
  u=la-torch.logsumexp(kernel+w[:,None,:],-1);w=lb-torch.logsumexp(kernel+u[:,:,None],-2)
 plan=(kernel+u[:,:,None]+w[:,None,:]).exp()
 return (plan*sim).sum((1,2))

def local_loss(h,a,q,v,ids,weights):
 positive=ids[:,None]==ids[None,:]
 hard=F.normalize(h.detach().float(),dim=-1)@F.normalize(a.detach().float(),dim=-1).T
 row=hard_mask(hard,positive);col=hard_mask(hard.T,positive.T)
 allowed=row|col.T;ii,jj=allowed.nonzero(as_tuple=True)
 # Compute only selected positive/hard-negative pairs, retaining gradients.
 values=torch.cat([pair_ot(q[i],v[j],weights[i]) for i,j in zip(ii.split(1024),jj.split(1024))])/.07
 scores=torch.zeros(len(ids),len(ids),device=q.device,dtype=torch.float32).index_put((ii,jj),values)
 return .5*(multipos(scores,positive,row)+multipos(scores.T,positive.T,col))
