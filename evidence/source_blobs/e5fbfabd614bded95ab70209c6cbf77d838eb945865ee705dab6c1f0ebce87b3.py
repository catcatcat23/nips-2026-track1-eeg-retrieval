"""Controlled v64 input / subject-supervision / LOSO extensions."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from augment_trials import MixedTrials, AugmentedModel
from data import DatasetEEG
from local_ot_train import loss_fn as original_loss, metrics, write

CFG = {}

class View(Dataset):
    def __init__(self, ds, keep):
        self.ds = ds
        self.indices = np.flatnonzero(keep)
        self.subjects = ds.subjects[self.indices]
        self.ids = ds.ids[self.indices]
        self.clean = ds.clean
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.ds[int(self.indices[i])]

class Trials(MixedTrials):
    def __init__(self, ds):
        if CFG.get('heldout_subject'):
            val_ids = np.array(json.loads(Path(CFG['validation_ids_path']).read_text()))
            ds = ConcatDataset([View(d, (d.subjects.astype(str) != CFG['heldout_subject']) & ~np.isin(d.ids,val_ids)) for d in ds.datasets])
        super().__init__(ds)
        self.fixed = CFG.get('fixed_repeats')
        if self.fixed:
            assert min(map(len, self.groups.values())) >= self.fixed, 'Not every group has four distinct trials'
    def __getitem__(self, i):
        if not self.fixed: return super().__getitem__(i)
        b = self.ds[i]
        pool = [j for j in self.groups[self.keys[i]] if j != i]
        chosen = [i] + np.random.choice(pool, self.fixed - 1, replace=False).tolist()
        assert len(set(chosen)) == self.fixed
        assert all(self.keys[j] == self.keys[i] for j in chosen)
        b['eeg'] = torch.stack([self.ds[j]['eeg'] for j in chosen]).mean(0)
        b['repeat_count'] = self.fixed
        return b

class IdentityModel(AugmentedModel):
    def __init__(self, manifest):
        super().__init__(manifest)
        # Preserve the base model RNG trajectory when adding a new head.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(333)
            self.identity_head = (nn.Identity() if CFG.get("identity_location") == "pooled512" else nn.Sequential(nn.LayerNorm(512), nn.Linear(512,256), nn.GELU(), nn.Linear(256,128)))
        self.blur_head.register_forward_pre_hook(self._capture)
        self.pooled_for_identity = None
    def _capture(self, module, inputs):
        self.pooled_for_identity = inputs[0]


def subject_loss(features, labels, temperature=.1):
    features = F.normalize(features.float(), dim=-1)
    logits = features @ features.T / temperature
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positive = (labels[:,None] == labels[None,:]) & ~diagonal
    valid = positive.any(1)
    if not valid.any(): return features.sum() * 0
    logp = logits - logits.masked_fill(diagonal, -torch.inf).logsumexp(1, keepdim=True)
    return -(logp.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1))[valid].mean()


def loss_fn(model, batch, weights):
    value, parts = original_loss(model, batch, weights)
    identity = value.new_zeros(())
    if CFG.get('identity_weight', 0):
        _, ids = np.unique(batch['subject'], return_inverse=True)
        identity = subject_loss(model.identity_head(model.pooled_for_identity), torch.tensor(ids,device=value.device), CFG['identity_temperature'])
        value = value + CFG['identity_weight'] * identity
        model.pooled_for_identity = None
    parts['identity'] = identity
    return value, parts


@torch.no_grad()
def evaluate_subset(model, cfg, split, heldout):
    model.eval()
    if split == 'test':
        full = DatasetEEG(cfg['source_root'], split, aux=False)
        ds = View(full, full.subjects.astype(str) == cfg['heldout_subject'])
    else:
        val_ids=np.array(json.loads(Path(cfg['validation_ids_path']).read_text()))
        parts=[]
        for name in ['train','val']:
            full=DatasetEEG(cfg['source_root'],name,aux=False)
            parts.append(View(full,(full.subjects.astype(str)!=cfg['heldout_subject']) & np.isin(full.ids,val_ids)))
        ds=ConcatDataset(parts)
        ds.ids=np.concatenate([d.ids for d in parts]);ds.subjects=np.concatenate([d.subjects for d in parts]);ds.clean=parts[0].clean
    assert len(ds)
    ids = np.unique(ds.ids)
    if split == 'test':
        assert len(ids) == 200
        blur = []
        for i in ids:
            with np.load(Path(cfg['test_blur_shards']) / f'{i:05d}.npz') as f:
                assert int(f['id']) == i
                blur.append(f['blur'].copy())
        blur = np.stack(blur)
    else:
        blur = np.load(Path(cfg['source_root'])/'features/blur.npy',mmap_mode='r')[ids].copy()
    gallery = F.normalize(model.image_adapter(torch.tensor(blur,device='cuda')),dim=-1).cpu().numpy()
    hparts,zparts=[],[]
    for b in DataLoader(ds,batch_size=128,num_workers=2):
        h,z,_=model.encode_eeg(b['eeg'].cuda())
        hparts.append(h.cpu().numpy());zparts.append(z.cpu().numpy())
    allblur=np.zeros_like(ds.clean);allblur[ids]=gallery
    return dict(blur=metrics(np.concatenate(hparts),ds.ids,ds.subjects,allblur),clean=metrics(np.concatenate(zparts),ds.ids,ds.subjects,ds.clean),split=split,subjects=np.unique(ds.subjects).astype(str).tolist())


def evaluate(model,cfg,weights):
    if cfg.get('heldout_subject'):return evaluate_subset(model,cfg,'val',False)
    from local_ot_train import evaluate as base_evaluate
    return base_evaluate(model,cfg,weights)


def audit(cfg,out):
    root=Path(cfg['source_root'])/'data'
    ids={s:np.load(root/f'{s}_ids.npy') for s in ['train','val','test']}
    subs={s:np.load(root/f'{s}_subjects.npy').astype(str) for s in ids}
    if cfg.get('heldout_subject'):
        held=cfg['heldout_subject'];assert held in subs['test']
        val_ids=np.array(json.loads(Path(cfg['validation_ids_path']).read_text()))
        joint_ids=np.concatenate([ids['train'],ids['val']]);joint_subs=np.concatenate([subs['train'],subs['val']])
        train_mask=(joint_subs!=held)&~np.isin(joint_ids,val_ids)
        val_mask=(joint_subs!=held)&np.isin(joint_ids,val_ids)
        assert not np.intersect1d(joint_ids[train_mask],joint_ids[val_mask]).size
        assert not np.any(joint_subs[train_mask]==held)
        assert not np.intersect1d(np.concatenate([ids['train'],ids['val']]),ids['test']).size
        train_subs=sorted(set(subs['train'])-{held});assert held not in train_subs
        weights=json.loads((out/'predictability/subjects.json').read_text())
        provenance=json.loads((out/'predictability/complete.json').read_text())
        assert provenance['validation_ids_excluded']==val_ids.tolist()
        assert all(s in weights for s in train_subs)
        write(out/'data_audit.json',dict(heldout=held,train_subjects=train_subs,train_count=int(train_mask.sum()),validation_count=int(val_mask.sum()),test_count=int((subs['test']==held).sum()),validation='seen subjects, fixed image-disjoint 10% training-image holdout',test='held-out subject, original 200 test images; evaluated only after selection',predictability='subject-specific train-only weights; held-out weights never used'))
    else:
        write(out/'data_audit.json',dict(train_count=sum(len(ids[s]) for s in ['train','val']),test_count=len(ids['test']),fixed_repeats=cfg.get('fixed_repeats'),identity_weight=cfg.get('identity_weight',0),selection='original v64 public test selection; not independent test'))


def preflight(cfg,out,model_class):
    from local_ot_train import LocalData
    torch.manual_seed(33)
    model=model_class(json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text())).cuda()
    model.perturbation_kind=cfg['augmentation']
    ds=Trials(ConcatDataset([LocalData(cfg,s) for s in ['train','val']]))
    indices=[]
    for subject in sorted(set(k[0] for k in ds.keys))[:4]:
        candidates=[i for i,k in enumerate(ds.keys) if k[0]==subject]
        used=set()
        for i in candidates:
            if ds.keys[i][1] in used:continue
            indices.append(i);used.add(ds.keys[i][1])
            if len(used)==2:break
    b=next(iter(DataLoader(torch.utils.data.Subset(ds,indices),batch_size=8)))
    if cfg.get('fixed_repeats'):assert (b['repeat_count']==4).all()
    weights_data=json.loads((out/'predictability/subjects.json').read_text())
    w=torch.tensor([weights_data[s]['weights'] for s in b['subject']],device='cuda')
    model.train();value,parts=loss_fn(model,b,w);value.backward()
    gradients={}
    for name in ['encoder','blur_head','recovery','image_adapter']+(['identity_head'] if cfg.get('identity_weight') and cfg.get('identity_location') != 'pooled512' else []):
        grads=[p.grad for p in getattr(model,name).parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        gradients[name]=sum(float(g.abs().sum()) for g in grads);assert gradients[name]>0
    if cfg.get('identity_location') == 'pooled512':
        assert isinstance(model.identity_head, nn.Identity)
        model.zero_grad(set_to_none=True)
        model(b['eeg'].cuda(), b['blur'].cuda(), b['local'].cuda())
        _, labels=np.unique(b['subject'],return_inverse=True)
        identity=subject_loss(model.pooled_for_identity,torch.tensor(labels,device='cuda'),cfg['identity_temperature'])
        identity.backward()
        g=[p.grad for p in model.encoder.parameters() if p.grad is not None]
        assert g and all(torch.isfinite(v).all() for v in g)
        gradients['identity_only_to_encoder']=sum(float(v.abs().sum()) for v in g)
        assert gradients['identity_only_to_encoder']>0
        model.pooled_for_identity=None
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    model.eval();initial=float(loss_fn(model,b,w)[0].detach())
    for step in range(100):
        opt.zero_grad();value,_=loss_fn(model,b,w);value.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        with torch.no_grad():
            h,z,a,_,_=model(b['eeg'].cuda(),b['blur'].cuda(),b['local'].cuda())
            hits=(F.normalize(h,dim=-1)@a.T).argmax(1)
            acc=float((b['id'].cuda()[hits]==b['id'].cuda()).float().mean())
        if step>=2 and acc==1 and float(value.detach())<initial*.5:break
    assert acc==1 and float(value.detach())<initial*.5, 'tiny-set overfit failed'
    write(out/'preflight.json',dict(passed=True,gradients=gradients,initial_loss=initial,final_loss=float(value.detach()),tiny_blur_top1=acc,steps=step+1,updates_discarded=True))


def final_loso(cfg,out,model_class):
    result=json.loads((out/'stage1/result.json').read_text())
    # One deployable variant and epoch selected on seen-subject validation only.
    variant=max(result['best'],key=result['best'].get);epoch=result['best_epoch'][variant]
    filename=f'epoch_{epoch:03d}.pt' if variant=='raw' else f'{variant}_epoch_{epoch:03d}.pt'
    ck=torch.load(out/'stage1'/filename,map_location='cpu',weights_only=False)
    model=model_class(ck['manifest']).cuda().eval();model.load_state_dict(ck['model'])
    scores=evaluate_subset(model,cfg,'test',True)
    write(out/'heldout_test.json',dict(variant=variant,epoch=epoch,selection='seen-subject validation blur single-trial Top5',metrics=scores))
