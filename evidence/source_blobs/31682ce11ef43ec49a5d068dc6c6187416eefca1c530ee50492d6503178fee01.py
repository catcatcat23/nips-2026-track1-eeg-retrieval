"""Independent 256-trial, image-conditioned subject contrastive batch."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import DatasetEEG

class GroupedIdentity:
    def __init__(self, cfg):
        self.datasets = [DatasetEEG(cfg['source_root'], s, aux=False) for s in ['train', 'val']]
        self.groups = {}
        for d, ds in enumerate(self.datasets):
            for i, (image, subject) in enumerate(zip(ds.ids, ds.subjects)):
                self.groups.setdefault(int(image), {}).setdefault(str(subject), []).append((d, i))
        self.groups = {im: {s: rows for s, rows in subs.items() if len(rows) >= 2} for im, subs in self.groups.items()}
        self.groups = {im: subs for im, subs in self.groups.items() if len(subs) >= 4}
        self.images = np.array(sorted(self.groups))
        test_ids = np.load(Path(cfg['source_root'])/'data/test_ids.npy')
        assert not np.intersect1d(self.images, test_ids).size
        assert len(self.images) >= 32
        self.rng = np.random.default_rng(810033)

    def sample(self):
        eeg, images, subjects, records = [], [], [], []
        for im in self.rng.choice(self.images, 32, replace=False):
            groups = self.groups[int(im)]
            for s in self.rng.choice(sorted(groups), 4, replace=False):
                pool = groups[s]
                for j in self.rng.choice(len(pool), 2, replace=False):
                    d, i = pool[j]
                    eeg.append(torch.from_numpy(self.datasets[d].x[i].copy()))
                    images.append(int(im)); subjects.append(s); records.append((d, i))
        assert len(eeg) == len(set(records)) == 256
        _, labels = np.unique(subjects, return_inverse=True)
        return torch.stack(eeg), torch.tensor(images), torch.tensor(labels)


def conditional_loss(features, images, subjects, temperature=.1):
    f = F.normalize(features.float(), dim=-1)
    logits = f @ f.T / temperature
    allowed = (images[:, None] == images[None, :]) & ~torch.eye(len(f), device=f.device, dtype=torch.bool)
    positives = allowed & (subjects[:, None] == subjects[None, :])
    assert torch.all(positives.sum(1) == 1)
    assert torch.all(allowed.sum(1) == 7)
    logp = logits - logits.masked_fill(~allowed, -torch.inf).logsumexp(1, keepdim=True)
    return -logp.masked_fill(~positives, 0).sum(1).mean()


def pooled(model, x):
    # Official/model preprocessing retained, auxiliary channel masking disabled.
    kind = model.perturbation_kind
    model.perturbation_kind = 'none'
    try:
        x = model.preprocess(x)
    finally:
        model.perturbation_kind = kind
    pos = model.positions.to(x).unsqueeze(0).expand(len(x), -1, -1)
    return model.encoder(x, pos=pos, return_output=True)[-1].mean(1)


AUX = None

def auxiliary_backward(model, cfg):
    global AUX
    if AUX is None:
        AUX = GroupedIdentity(cfg)
    x, im, sub = AUX.sample()
    loss = conditional_loss(pooled(model, x.cuda()), im.cuda(), sub.cuda(), cfg['identity_temperature'])
    assert torch.isfinite(loss)
    (cfg['identity_weight'] * loss).backward()
    return loss.detach()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--cpu', action='store_true')
    a = p.parse_args(); cfg = json.loads(Path(a.config).read_text()); out = Path(cfg['run_root'])
    sampler = GroupedIdentity(cfg)
    for _ in range(5):
        x, im, sub = sampler.sample()
        f = torch.randn(256, 32, requires_grad=True)
        loss = conditional_loss(f, im, sub); loss.backward()
        assert torch.isfinite(f.grad).all()
    # Image groups are independent: changing all other groups cannot change an anchor's denominator.
    f = torch.randn(256,32)
    full = conditional_loss(f, im, sub)
    split = torch.stack([conditional_loss(f[im==v], im[im==v], sub[im==v]) for v in im.unique()]).mean()
    torch.testing.assert_close(full, split)
    report = dict(passed=True, images=len(sampler.images), batch_size=256, images_per_batch=32,
                  subjects_per_image=4, distinct_trials_per_subject=2, positives_per_anchor=1,
                  negatives_per_anchor=6, test_overlap=0, grouping_checks=5)
    (out/'grouped_cpu_check.json').write_text(json.dumps(report, indent=2))
    if a.cpu: return
    import extensions as ex
    ex.CFG.update(cfg)
    torch.set_num_threads(4); torch.manual_seed(33)
    model = ex.IdentityModel(json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text())).cuda()
    model.train(); model.perturbation_kind = cfg['augmentation']
    value = auxiliary_backward(model, cfg)
    grads = [v.grad for v in model.encoder.parameters() if v.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    magnitude = sum(float(g.abs().sum()) for g in grads); assert magnitude > 0
    for name in ['blur_head','recovery','image_adapter','local_projection']:
        assert all(p.grad is None for p in getattr(model,name).parameters())
    # Tiny real EEG overfit: two trials each from four people, one image.
    x, im, sub = x[:8].cuda(), im[:8].cuda(), sub[:8].cuda()
    model.zero_grad(set_to_none=True); model.eval()
    opt = torch.optim.AdamW(model.encoder.parameters(), lr=1e-4)
    initial = float(conditional_loss(pooled(model,x), im, sub).detach())
    for step in range(50):
        opt.zero_grad(); value = conditional_loss(pooled(model,x), im, sub)
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        if step >= 2 and float(value.detach()) < initial*.5: break
    assert float(value.detach()) < initial*.5
    report.update(encoder_gradient_abs=magnitude, initial=initial, final=float(value.detach()), steps=step+1, updates_discarded=True)
    (out/'grouped_gpu_check.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__': main()
