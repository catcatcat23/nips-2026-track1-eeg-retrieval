"""Three Alljoined identity controls over the frozen v67 training recipe."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import alljoined_v67 as base


def contrast(features, subjects, groups=None):
    f = F.normalize(features.float(), dim=-1)
    logits = f @ f.T / .1
    allowed = ~torch.eye(len(f), dtype=torch.bool, device=f.device)
    if groups is not None:
        allowed &= groups[:, None] == groups[None, :]
    positive = allowed & (subjects[:, None] == subjects[None, :])
    valid = positive.any(1)
    assert valid.any() and allowed[valid].any(1).all()
    logp = logits - logits.masked_fill(~allowed, -torch.inf).logsumexp(1, keepdim=True)
    return -(logp.masked_fill(~positive, 0).sum(1)[valid] / positive.sum(1)[valid]).mean()


class Sampler:
    def __init__(self, cfg, train, test):
        self.ds = train
        self.kind = cfg['identity_scope']
        self.rng = np.random.default_rng(810033)
        self.groups = {}
        for i, (im, sub) in enumerate(zip(train.ids, train.subjects)):
            self.groups.setdefault(int(im), {}).setdefault(str(sub), []).append(i)
        self.groups = {i: {s: ix for s, ix in g.items() if len(ix) >= 2} for i, g in self.groups.items()}
        self.groups = {i: g for i, g in self.groups.items() if len(g) >= 4}
        self.images = np.array(sorted(self.groups))
        assert len(self.images) == 16540 and not np.intersect1d(self.images, test.ids).size
        paths = json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text())['paths']
        concepts = {i: Path(p).parent.name.split('_', 1)[1] for i, p in enumerate(paths)}
        assert not {concepts[int(i)] for i in self.images} & {concepts[int(i)] for i in np.unique(test.ids)}
        by = {}
        for i in self.images: by.setdefault(concepts[int(i)], []).append(int(i))
        self.concepts = list(by.values())
        assert len(by) == 1654 and all(len(x) == 10 for x in self.concepts)
        self.common = [sorted(set.intersection(*(set(self.groups[i]) for i in ims))) for ims in self.concepts]
        assert all(len(s) >= 4 for s in self.common)

    def sample(self):
        records, labels = [], []
        if self.kind == 'concept':
            blocks = [(int(c), self.rng.choice(self.concepts[c], 2, replace=False), self.rng.choice(self.common[c], 4, replace=False)) for c in self.rng.choice(1654, 16, replace=False)]
        else:
            blocks = [(int(i), [int(i)], self.rng.choice(sorted(self.groups[int(i)]), 4, replace=False)) for i in self.rng.choice(self.images, 32, replace=False)]
        for group, images, subjects in blocks:
            for sub in subjects:
                for im in images:
                    for ix in self.rng.choice(self.groups[int(im)][sub], 2, replace=False):
                        assert self.ds.ids[ix] == im and str(self.ds.subjects[ix]) == sub
                        records.append(int(ix)); labels.append((group, sub))
        assert len(records) == len(set(records)) == 256
        _, sub = np.unique([s for _, s in labels], return_inverse=True)
        group = torch.tensor([g for g, _ in labels])
        sub = torch.tensor(sub)
        positive = (group[:, None] == group[None, :]) & (sub[:, None] == sub[None, :])
        assert torch.all(positive.sum(1) == (4 if self.kind == 'concept' else 2))
        x = torch.tensor(np.stack([self.ds.eeg(i) for i in records]), dtype=torch.float32)
        return x, group, sub


def pooled(model, x):
    kind = model.perturbation_kind
    model.perturbation_kind = 'none'
    try: x = model.preprocess(x)
    finally: model.perturbation_kind = kind
    return model.encoder(x, pos=model.positions.to(x)[None].expand(len(x), -1, -1), return_output=True)[-1].mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--check', choices=['cpu','gpu'])
    a = ap.parse_args(); cfg = json.loads(Path(a.config).read_text()); out = Path(cfg['run_root'])
    torch.set_num_threads(4)
    meta, train, test = base.load(cfg)
    sampler = Sampler(cfg, train, test)
    if a.check:
        for _ in range(3):
            x, groups, subjects = sampler.sample()
            f = torch.randn(256, 32, requires_grad=True)
            g = None if cfg['identity_scope']=='global' else groups
            loss = contrast(f, subjects, g); loss.backward(); assert torch.isfinite(f.grad).all()
            p = torch.randperm(256)
            torch.testing.assert_close(loss.detach(), contrast(f.detach()[p], subjects[p], None if g is None else g[p]))
        report = dict(passed=True, train_trials=len(train), test_trials=len(test), train_images=16540, test_images=200, subjects=len(np.unique(train.subjects)), concepts=1654, channels=meta['channels'], test_overlap=0, scope=cfg['identity_scope'], batch_size=256)
        if a.check == 'gpu':
            torch.manual_seed(33)
            model = base.LocalOTModel({'channels':meta['channels']}).cuda().eval()
            n = 16 if cfg['identity_scope']=='concept' else 8
            x=x[:n].cuda(); subjects=subjects[:n].cuda(); g=None if cfg['identity_scope']=='global' else groups[:n].cuda()
            initial=float(contrast(pooled(model,x),subjects,g).detach())
            loss=contrast(pooled(model,x),subjects,g);loss.backward()
            grads=[p.grad for p in model.encoder.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(t).all() for t in grads) and sum(float(t.abs().sum()) for t in grads)>0
            assert all(p.grad is None for p in model.blur_head.parameters())
            opt=torch.optim.AdamW(model.encoder.parameters(),lr=1e-4)
            floor=float(np.log(3)) if cfg['identity_scope']=='concept' else 0.
            for step in range(50):
                opt.zero_grad();loss=contrast(pooled(model,x),subjects,g);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                if step>2 and float(loss.detach())-floor < .5*(initial-floor):break
            assert float(loss.detach())-floor < .5*(initial-floor)
            report.update(initial=initial,final=float(loss.detach()),steps=step+1,updates_discarded=True)
        (out/f'identity_{a.check}_check.json').write_text(json.dumps(report,indent=2))
        return
    assert json.loads((out/'identity_gpu_check.json').read_text())['passed']
    original = base.loss_fn
    calls = 0
    def loss_fn(model, batch, weights):
        nonlocal calls
        features=[]
        hook=model.blur_head.register_forward_pre_hook(lambda module, inputs: features.append(inputs[0]))
        try: retrieval, parts=original(model,batch,weights)
        finally: hook.remove()
        if cfg['identity_scope']=='global':
            _, subjects=np.unique(batch['subject'],return_inverse=True)
            identity=contrast(features[0],torch.tensor(subjects,device='cuda'))
        else:
            x,g,s=sampler.sample()
            identity=contrast(pooled(model,x.cuda()),s.cuda(),g.cuda())
        weighted=cfg['identity_weight']*identity
        if calls<3 or calls%500==0:
            params=list(model.encoder.parameters())
            g=torch.autograd.grad(retrieval,params,retain_graph=True,allow_unused=True)
            h=torch.autograd.grad(weighted,params,retain_graph=True,allow_unused=True)
            aa=identity.new_zeros(());bb=aa.clone();ab=aa.clone()
            for u,v in zip(g,h):
                if u is not None:aa+=u.detach().float().square().sum()
                if v is not None:bb+=v.detach().float().square().sum()
                if u is not None and v is not None:ab+=(u.detach().float()*v.detach().float()).sum()
            row=dict(step=calls,retrieval=float(aa.sqrt()),identity=float(bb.sqrt()),ratio=float((bb/aa.clamp_min(1e-20)).sqrt()),cosine=float(ab/(aa*bb).sqrt().clamp_min(1e-20)))
            with (out/'gradient_balance.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('GRADIENT',row,flush=True)
        calls+=1
        parts['identity']=identity
        return retrieval+weighted,parts
    base.loss_fn=loss_fn
    # Reuse loaded arrays and records rather than holding two copies of the dataset.
    base.load=lambda config:(meta,train,test)
    sys.argv=[sys.argv[0],'--config',a.config]
    base.main()

if __name__=='__main__':main()
