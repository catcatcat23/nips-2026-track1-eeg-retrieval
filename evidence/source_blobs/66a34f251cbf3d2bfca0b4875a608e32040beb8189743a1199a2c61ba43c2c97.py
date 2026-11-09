"""CPU regression checks; run with the project environment."""
import torch
from image_id_loss import ImageIdClipLoss
from neuraltrain.losses.losses import ClipLoss

torch.manual_seed(21)
torch.set_num_threads(2)
loss = ImageIdClipLoss()
old = ClipLoss(norm_kind='y', temperature=False, symmetric=False)
x = torch.randn(8,16,requires_grad=True)
y = torch.randn(8,16,requires_grad=True)
ids = torch.arange(8)
a,b = loss(x,y,ids),old(x,y)
torch.testing.assert_close(a,b)
for ga,gb in zip(torch.autograd.grad(a,(x,y)),torch.autograd.grad(b,(x,y))):
    torch.testing.assert_close(ga,gb,atol=1e-6,rtol=1e-5)
# Repeated identical images collapse to a unique gallery, but every EEG counts.
ids = torch.tensor([3,3,3,8,8,12,12,12])
y0 = torch.randn(3,16,requires_grad=True)
_,inv = ids.unique(return_inverse=True)
y = y0[inv]
expected = torch.nn.functional.cross_entropy(old.get_scores(x,y0),inv)
actual = loss(x,y,ids)
torch.testing.assert_close(actual,expected)
for ga,gb in zip(torch.autograd.grad(actual,(x,y0)),torch.autograd.grad(expected,(x,y0))):
    torch.testing.assert_close(ga,gb,atol=1e-6,rtol=1e-5)
# Stochastic adapter views: explicit grouped log-mean-exp reference.
y = torch.randn(8,16,requires_grad=True)
scores = old.get_scores(x,y)
grouped = torch.stack([torch.logsumexp(scores[:,ids==i],1)-torch.tensor(int((ids==i).sum())).float().log() for i in ids.unique()],1)
actual = loss(x,y,ids)
torch.testing.assert_close(actual,torch.nn.functional.cross_entropy(grouped,inv))
perm = torch.randperm(8)
torch.testing.assert_close(actual,loss(x[perm],y[perm],ids[perm]))
grad = torch.autograd.grad(actual,(x,y))
assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grad)
assert loss(x,y,torch.zeros(8,dtype=torch.long)).abs()<1e-6
with torch.autocast('cpu',dtype=torch.bfloat16):
    assert loss(x,y,ids).dtype == torch.float32
# Tiny multi-positive fit: only EEG vectors learn; repeated targets stay frozen.
ids = torch.arange(4).repeat_interleave(2)
target = torch.eye(4)[ids]
pred = torch.nn.Parameter(torch.randn(8,4))
opt = torch.optim.Adam([pred],lr=.1)
initial = float(loss(pred,target,ids).detach())
for _ in range(150):
    opt.zero_grad(); value=loss(pred,target,ids);value.backward();opt.step()
assert (pred.argmax(-1)==ids).all() and value<.02
print({'checks':'PASS','tiny_fit_initial':initial,'tiny_fit_final':float(value.detach()),'accuracy':1.0})
