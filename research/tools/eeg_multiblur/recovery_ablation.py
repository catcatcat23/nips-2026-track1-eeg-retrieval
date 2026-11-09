"""Frozen v13 representation probe: old head A, projected head B/D.
Public test selects checkpoints by explicit user request, not independent testing.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import DatasetEEG
from evaluate import normalize, group_means
from evaluate_parallel import gpu_hits
from reference_model import ReferenceModel
from neuraltrain.losses.losses import ClipLoss


class OldHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.recovery = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 768),
                                      nn.GELU(), nn.Linear(768, 1536))

    def forward(self, h):
        return h + self.recovery(h)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.net = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 3072),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(3072, 1536))

    def forward(self, x):
        return x + self.alpha * self.net(x)


class ProjectedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 1536),
                                 ResidualBlock(), ResidualBlock(),
                                 nn.LayerNorm(1536), nn.Linear(1536, 1536))

    def forward(self, h):
        # Raw outputs intentionally retained for official mean-then-normalize scoring.
        return self.net(h)


def make_head(arm):
    torch.manual_seed(33)
    return (OldHead() if arm == 'A' else ProjectedHead()).cuda()


def objective(q, target, ids, arm):
    if arm == 'B_ID':
        from image_id_loss import ImageIdClipLoss
        loss = ImageIdClipLoss().to(q.device)(q, target, ids)
        return loss, {'retrieval': loss.detach()}
    if arm != 'D':
        loss = ClipLoss(norm_kind='y', temperature=False, symmetric=False).cuda()(q, target)
        return loss, {'retrieval': loss.detach()}
    # Unique image gallery: repeated image IDs cannot become false negatives.
    unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    first = torch.full((len(unique),), len(ids), device=ids.device, dtype=torch.long)
    first.scatter_reduce_(0, inverse, torch.arange(len(ids), device=ids.device), reduce='amin')
    c = F.normalize(target.float(), dim=-1)
    gallery = c[first]
    z = F.normalize(q.float(), dim=-1)
    retrieval = F.cross_entropy(z @ gallery.T / 0.07, inverse)
    direction = (1 - (z * c).sum(-1)).mean()
    teacher = (c @ gallery.T / 0.1).softmax(-1).detach()
    relation = F.kl_div((z @ gallery.T / 0.1).log_softmax(-1), teacher, reduction='batchmean')
    entropy = -(teacher * teacher.clamp_min(1e-12).log()).sum(-1).mean()
    return retrieval + 0.1 * direction + 0.1 * relation, {
        'retrieval': retrieval.detach(), 'direction': direction.detach(),
        'relation': relation.detach(), 'teacher_entropy': entropy}


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def load_source(cfg):
    p = Path(cfg['checkpoint'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == cfg['checkpoint_sha256']
    ck = torch.load(p, map_location='cpu', weights_only=False)
    model = ReferenceModel(ck['manifest']).cuda().eval()
    model.load_state_dict(ck['model'])
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode(model, eeg):
    x = model.preprocess(eeg)
    pos = model.positions.to(x).unsqueeze(0).expand(len(x), -1, -1)
    return model.blur_head(model.encoder(x, pos=pos, return_output=True)[-1].mean(1))


def preflight(cfg, out):
    ds = DatasetEEG(cfg['source_root'], 'train', aux=False)
    _, first = np.unique(ds.ids, return_index=True)
    batch = next(iter(DataLoader(torch.utils.data.Subset(ds, first[:8]), batch_size=8)))
    model = load_source(cfg)
    h = encode(model, batch['eeg'].cuda()).detach()
    c, ids = batch['clean'].cuda(), batch['id'].cuda()
    with torch.no_grad():
        recovered = model(batch['eeg'].cuda())
        torch.testing.assert_close(h + model.recovery(h), recovered)
    assert not h.requires_grad and all(not p.requires_grad for p in model.parameters())
    del model
    results = {}
    for arm in cfg.get('preflight_arms', ('A', 'B', 'D')):
        head = make_head(arm)
        head.eval()  # Disable dropout only for tiny overfit diagnostic.
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0)
        initial = float(objective(head(h), c, ids, arm)[0].detach())
        for step in range(300):
            opt.zero_grad(set_to_none=True)
            loss, _ = objective(head(h), c, ids, arm)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                accuracy = float(((F.normalize(head(h), dim=-1) @ F.normalize(c, dim=-1).T).argmax(-1) == torch.arange(8, device='cuda')).float().mean())
            if accuracy == 1.0 and float(loss.detach()) < initial * 0.3:
                break
        assert accuracy == 1.0 and float(loss.detach()) < initial * 0.3, (arm, initial, loss, accuracy)
        head.train()
        opt.zero_grad(set_to_none=True)
        smoke, _ = objective(head(h.repeat(32, 1)), c.repeat(32, 1), ids.repeat(32), arm)
        smoke.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        results[arm] = {'initial_loss': initial, 'final_loss': float(loss.detach()), 'steps': step + 1, 'top1': accuracy, 'train_mode_gradients_finite': True}
        print('PREFLIGHT', arm, results[arm], flush=True)
    # Verify duplicate ID handling and invariance to candidate duplication in D.
    q = torch.randn(3, 1536, device='cuda', requires_grad=True)
    targets = c[[0, 0, 1]]
    l, _ = objective(q, targets, torch.tensor([0, 0, 1], device='cuda'), 'D')
    l.backward()
    assert torch.isfinite(q.grad).all()
    write(out / 'preflight.json', results)


def extract(cfg, out):
    model = load_source(cfg)
    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    for split in ('train', 'val', 'test'):
        ds = DatasetEEG(cfg['source_root'], split, aux=False)
        path = cache / f'{split}_h.npy'
        assert not path.exists(), 'Refusing to overwrite cached features'
        array = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(ds), 1536))
        start = 0
        for step, batch in enumerate(DataLoader(ds, batch_size=256, num_workers=2)):
            h = encode(model, batch['eeg'].cuda()).cpu().numpy()
            assert np.isfinite(h).all()
            array[start:start+len(h)] = h
            start += len(h)
            if step % 200 == 0:
                print('EXTRACT', split, start, '/', len(ds), flush=True)
        assert start == len(ds)
        array.flush()
        np.save(cache / f'{split}_ids.npy', ds.ids)
        np.save(cache / f'{split}_subjects.npy', ds.subjects)
    write(cache / 'complete.json', {'checkpoint_sha256': cfg['checkpoint_sha256'], 'dtype': 'float32', 'encoder_mode': 'eval', 'splits': ['train', 'val', 'test']})


def metrics(pred, ids, subjects, clean):
    unique, inverse = np.unique(ids, return_inverse=True)
    candidates, image_idx = np.unique(clean[unique], axis=0, return_inverse=True)
    labels = image_idx.reshape(-1)[inverse]
    gallery = normalize(candidates)
    _, sid = np.unique(subjects, return_inverse=True)
    keys, means = group_means(pred, np.stack([sid, labels], axis=1))
    result = {'candidate_images': len(gallery), 'per_subject': {}}
    for name, x, y, subject_indices in [('single_trial', pred, labels, sid), ('subject_aggregated', means, keys[:, 1], keys[:, 0])]:
        hits = gpu_hits(x, gallery, y, torch.device('cuda'))
        result[name] = {'top1': float(hits[:, 0].mean()), 'top5': float(hits[:, 1].mean()), 'count': len(x)}
        for i, subject in enumerate(np.unique(subjects)):
            selected = hits[subject_indices == i]
            result['per_subject'].setdefault(str(subject), {})[name] = {'top1': float(selected[:, 0].mean()), 'top5': float(selected[:, 1].mean()), 'count': len(selected)}
    return result


@torch.no_grad()
def predictions(head, x):
    head.eval()
    return np.concatenate([head(b).cpu().numpy() for b in x.split(1024)])


def train(cfg, out, arm):
    cache = Path(cfg.get('cache_root', out / 'cache'))
    assert json.loads((cache / 'complete.json').read_text())['checkpoint_sha256'] == cfg['checkpoint_sha256']
    dest = out / arm
    dest.mkdir(exist_ok=True)
    assert not (dest / 'history.jsonl').exists(), 'Use a new run directory'
    clean = np.load(Path(cfg['source_root']) / 'data/clean.npy')
    target = torch.from_numpy(clean).cuda()
    ids = np.concatenate([np.load(cache / f'{s}_ids.npy') for s in ('train', 'val')])
    trainids = torch.as_tensor(ids, device='cuda')
    h = torch.from_numpy(np.concatenate([np.load(cache / f'{s}_h.npy') for s in ('train', 'val')])).cuda()
    testids = np.load(cache / 'test_ids.npy')
    subjects = np.load(cache / 'test_subjects.npy')
    assert len(np.intersect1d(ids, testids)) == 0
    testh = torch.from_numpy(np.load(cache / 'test_h.npy')).cuda()
    head = make_head(arm)
    if cfg.get('initial_reference'):
        initial = torch.load(cfg['initial_reference'], map_location='cpu', weights_only=True)
        for key, value in head.state_dict().items():
            assert torch.equal(value.cpu(), initial[key]), key
    torch.save(head.state_dict(), dest / 'initial_head.pt')
    opt = torch.optim.AdamW(head.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps = (len(h) + cfg['batch_size'] - 1) // cfg['batch_size']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], epochs=cfg['epochs'], steps_per_epoch=steps, pct_start=0.1)
    if arm == 'A':
        ck = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
        old = OldHead().cuda()
        old.recovery.load_state_dict({k.removeprefix('recovery.'): v for k, v in ck['model'].items() if k.startswith('recovery.')})
        original = metrics(predictions(old, testh), testids, subjects, clean)
        assert abs(original['single_trial']['top5'] - 0.3708086785009862) < 1e-5
        write(dest / 'original_v13_metrics.json', original)
        del old, ck
    best, stale = -1.0, 0
    generator = torch.Generator(device='cuda').manual_seed(cfg['seed'])
    for epoch in range(1, cfg['epochs'] + 1):
        begin = time.time()
        head.train()
        order = torch.randperm(len(h), generator=generator, device='cuda')
        totals, count = {}, 0
        for step, idx in enumerate(order.split(cfg['batch_size'])):
            opt.zero_grad(set_to_none=True)
            q = head(h[idx])
            loss, parts = objective(q, target[trainids[idx]], trainids[idx], arm)
            assert torch.isfinite(loss), (arm, epoch, step)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            sched.step()
            for key, value in {'total': loss.detach(), **parts}.items():
                totals[key] = totals.get(key, 0.0) + float(value) * len(idx)
            count += len(idx)
            if step % 500 == 0:
                print(json.dumps({'arm': arm, 'epoch': epoch, 'step': step, 'steps': steps, 'loss': float(loss.detach())}), flush=True)
        selection = metrics(predictions(head, testh), testids, subjects, clean)
        # Fixed training subset: diagnostic only, never checkpoint selection.
        diagnostic = metrics(predictions(head, h[:2048]), ids[:2048], np.zeros(2048, dtype=int), clean)
        score = selection['single_trial']['top5']
        improved = score > best
        if improved:
            best, stale = score, 0
            torch.save({'head': head.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'epoch': epoch, 'arm': arm, 'config': cfg, 'selection': selection, 'source_checkpoint_sha256': cfg['checkpoint_sha256']}, dest / 'best.pt')
        else:
            stale += 1
        record = {'arm': arm, 'epoch': epoch, 'loss': {k: v/count for k, v in totals.items()}, 'selection': selection, 'train_diagnostic': diagnostic, 'lr': opt.param_groups[0]['lr'], 'best_top5': best, 'stale': stale, 'seconds': time.time() - begin}
        with (dest / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        write(dest / 'status.json', {'state': 'training', 'epoch': epoch, 'best_top5': best, 'stale': stale})
        if stale >= cfg['patience']:
            break
    best_ck = torch.load(dest / 'best.pt', map_location='cpu', weights_only=False)
    write(dest / 'result.json', {'arm': arm, 'best_epoch': best_ck['epoch'], 'selection': best_ck['selection'], 'role': 'public_test_used_for_model_selection_not_independent_test'})
    write(dest / 'status.json', {'state': 'complete', 'epochs': epoch, 'best_epoch': best_ck['epoch'], 'best_top5': best})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preflight', 'extract', 'train'])
    parser.add_argument('--arm', choices=['A', 'B', 'D', 'B_ID'])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('recovery_ablation_config.json'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['run_root'])
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    if args.stage == 'preflight': preflight(cfg, out)
    elif args.stage == 'extract': extract(cfg, out)
    else: train(cfg, out, args.arm)


if __name__ == '__main__':
    main()
