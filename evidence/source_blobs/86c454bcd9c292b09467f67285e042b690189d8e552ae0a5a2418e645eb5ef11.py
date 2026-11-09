"""Matched clean-only REVE controls: passive EMA x trial mixing/channel masking."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Dataset
from reference_model import ReferenceBase
from data import DatasetEEG
from alljoined_v67 import Alljoined
from augment_trials import MixedTrials
from four_trials import FourTrials
from local_ot_train_ema import EMA
from neuraltrain.losses.losses import ClipLoss
from v33_recovery_stage2 import metrics, write


class CleanAlljoined(Alljoined):
    def __getitem__(self, i):
        k = int(self.ids[i])
        return dict(eeg=torch.from_numpy(np.array(self.eeg(i), dtype=np.float32)),
                    id=k, subject=str(self.subjects[i]),
                    clean=torch.from_numpy(self.clean[k].copy()))


class CleanModel(ReferenceBase):
    def __init__(self, channels, all_channels, augment):
        super().__init__(dict(sfreq=120, n_times=120, n_chans=len(channels),
                              ch_names=channels), 'resample200')
        names = [str(n).removeprefix('EEG ').split('-', 1)[0] for n in all_channels]
        self.register_buffer('alljoined_positions', self.encoder.get_positions(names))
        self.domain = 'things'
        self.augment = augment

    def forward(self, x):
        x = self.preprocess(x)
        if self.training and self.augment:
            x = x.clone()
            rows = (torch.rand(len(x), device=x.device) < .2).nonzero().flatten()
            cols = torch.randint(x.shape[1], (len(rows),), device=x.device)
            x[rows, cols] = 0
        p = self.positions if self.domain == 'things' else self.alljoined_positions
        assert len(p) == x.shape[1]
        tokens = self.encoder(x, pos=p.to(x).unsqueeze(0).expand(len(x), -1, -1),
                              return_output=True)[-1]
        return self.head(tokens.mean(1))


def seed():
    random.seed(33)
    np.random.seed(33)
    torch.manual_seed(33)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def digest(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, ds, domain):
    model.eval()
    model.domain = domain
    parts = []
    for b in DataLoader(ds, batch_size=256, num_workers=4):
        parts.append(model(b['eeg'].cuda()).cpu().numpy())
    pred = np.concatenate(parts)
    result = metrics(pred, ds.ids, ds.subjects, ds.clean)
    n = np.linalg.norm(pred, axis=1)
    result['prediction_norm'] = dict(mean=float(n.mean()), std=float(n.std()))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    mix_class = MixedTrials
    if cfg.get('mix_scope') == 'cross_subject':
        from cross_subject_mix import CrossSubjectTrials
        mix_class = CrossSubjectTrials
        if cfg.get('cross_subject_anchor_weight') == .7:
            from weighted_cross_subject_mix import WeightedCrossSubjectTrials
            mix_class = WeightedCrossSubjectTrials
    if cfg.get('fixed_repeats') == 4:
        assert cfg['augment'] and cfg.get('mix_scope', 'within_subject') == 'within_subject'
        mix_class = FourTrials
    root = Path(cfg['run_root'])
    seed()
    tc = json.loads(Path(cfg['things_config']).read_text())
    ac = json.loads(Path(cfg['alljoined_config']).read_text())
    manifest = json.loads((Path(tc['source_root']) / 'data/manifest.json').read_text())
    meta = json.loads(Path(ac['data_manifest']).read_text())
    clean = np.load(ac['clean_features'], mmap_mode='r')
    td = [DatasetEEG(tc['source_root'], s, aux=False) for s in ['train', 'val']]
    at = CleanAlljoined(meta['train'], clean, None, None)
    tests = dict(things=DatasetEEG(tc['source_root'], 'test', aux=False),
                 alljoined=CleanAlljoined(meta['test'], clean, None, None))
    train = dict(things=ConcatDataset(td), alljoined=ConcatDataset([at]))
    ti = np.unique(np.concatenate([d.ids for d in td]))
    assert np.array_equal(ti, np.unique(at.ids)) and len(ti) == 16540
    for d, ds in tests.items():
        assert len(np.unique(ds.ids)) == 200 and not np.intersect1d(ds.ids, ti).size
    assert np.array_equal(np.unique(tests['things'].ids), np.unique(tests['alljoined'].ids))
    assert Path(ac['clean_features']).resolve() == (Path(tc['source_root'])/'data/clean.npy').resolve()
    def make():
        torch.manual_seed(33)
        return CleanModel(manifest['channels'], meta['channels'], cfg['augment']).cuda()
    model = make()
    initial_hash = digest(model)
    write(root/'data_audit.json', dict(train={d:len(ds) for d,ds in train.items()},
          test={d:len(ds) for d,ds in tests.items()}, train_images=len(ti),
          test_images=200, image_overlap=0, initial_state_sha256=initial_hash,
          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    criterion = ClipLoss(norm_kind='y', temperature=False, symmetric=False)
    if args.preflight:
        results = {}
        for domain, base in [('things', td[0]), ('alljoined', at)]:
            probe = make().eval()
            probe.domain = domain
            _, ix = np.unique(base.ids, return_index=True)
            b = next(iter(DataLoader(torch.utils.data.Subset(base, ix[:8]), batch_size=8)))
            x, y = b['eeg'].cuda(), b['clean'].cuda()
            opt = torch.optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=0)
            for step in range(200):
                opt.zero_grad()
                z = probe(x)
                loss = criterion(z, y)
                if step == 0: initial = float(loss.detach())
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all()
                           for p in probe.parameters() if p.requires_grad)
                nn.utils.clip_grad_norm_(probe.parameters(), 1, error_if_nonfinite=True)
                opt.step()
                with torch.no_grad():
                    acc = float(((probe(x) @ nn.functional.normalize(y, dim=-1).T).argmax(1)
                                 == torch.arange(8, device='cuda')).float().mean())
                if acc == 1 and float(loss.detach()) < .3*initial: break
            assert acc == 1 and float(loss.detach()) < .3*initial
            probe.train()
            probe.zero_grad()
            criterion(probe(x), y).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in probe.parameters() if p.requires_grad)
            # Passive EMA equation and eval determinism.
            ema = EMA(probe, .9998)
            old = next(probe.parameters()).detach().clone()
            with torch.no_grad(): next(probe.parameters()).add_(.01)
            ema.update(probe)
            torch.testing.assert_close(ema.params[0], old*.9998+(old+.01)*.0002)
            probe.eval()
            with torch.no_grad(): torch.testing.assert_close(probe(x), probe(x), rtol=0, atol=0)
            results[domain] = dict(steps=step+1, initial_loss=initial,
                                    final_loss=float(loss.detach()), tiny_top1=acc)
            del probe, opt, ema
        if cfg['augment']:
            audits = {}
            for domain, ds in train.items():
                mix = mix_class(ds)
                audits[domain] = getattr(mix, 'audit', {'anchors':len(mix)})
                if cfg.get('cross_subject_anchor_weight') == .7:
                    counts = {1:0, 2:0, 4:0}
                    for trial in range(120):
                        anchor = trial % len(mix)
                        np.random.seed(733+trial)
                        r = np.random.random()
                        k = 1 if r < .5 else (2 if r < .75 else 4)
                        members = mix.members(anchor,k)
                        weights = {1:[1.],2:[.7,.3],4:[.7,.1,.1,.1]}[k]
                        stack = torch.stack([ds[j]['eeg'] for j in members])
                        expected = (stack*stack.new_tensor(weights).reshape(-1,1,1)).sum(0)
                        np.random.seed(733+trial)
                        actual = mix[anchor]
                        torch.testing.assert_close(actual['eeg'],expected,rtol=0,atol=0)
                        assert actual['id'] == ds[anchor]['id']
                        assert actual['subject'] == ds[anchor]['subject']
                        assert actual['repeat_count'] == k
                        counts[k] += 1
                    assert all(counts.values())
                    audits[domain]['verified_weighted_draws'] = counts
                for i in range(20):
                    b = mix[i]
                    assert b['repeat_count'] in [1,2,3,4] and torch.isfinite(b['eeg']).all()
                    if cfg.get('fixed_repeats') == 4:
                        assert b['repeat_count'] == 4
                        anchor = mix.valid[i]
                        pool = mix.groups[mix.keys[anchor]]
                        np.random.seed(177+i)
                        chosen = [anchor]+np.random.choice([j for j in pool if j != anchor],3,replace=False).tolist()
                        expected = torch.stack([ds[j]['eeg'] for j in chosen]).mean(0)
                        np.random.seed(177+i)
                        torch.testing.assert_close(mix[i]['eeg'], expected, rtol=0, atol=0)
            write(root/'augmentation_audit.json', audits)
        write(root/'preflight.json', dict(passed=True, tiny=results,
              initial_state_sha256=initial_hash, updates_discarded=True))
        print('PREFLIGHT PASSED', results, flush=True)
        return
    assert json.loads((root/'preflight.json').read_text())['initial_state_sha256'] == initial_hash
    loaders, samplers = {}, {}
    for d, ds in train.items():
        if cfg['augment']: ds = mix_class(ds)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0, seed=33, drop_last=True)
        samplers[d] = sampler
        loaders[d] = DataLoader(ds, batch_size=256, sampler=sampler, num_workers=4,
                                persistent_workers=True, pin_memory=True, drop_last=True,
                                generator=torch.Generator().manual_seed(33))
    steps = len(loaders['things'])
    assert steps <= len(loaders['alljoined'])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4,
              total_steps=cfg['epochs']*steps, pct_start=.1)
    ema = EMA(model, .9998)
    evaluator = make().eval()
    seed()
    dest = root/'stage1'
    dest.mkdir()
    best, stale, rankings = -1., 0, []
    for epoch in range(1, cfg['epochs']+1):
        for s in samplers.values(): s.set_epoch(epoch-1)
        its = {d:iter(l) for d,l in loaders.items()}
        model.train()
        total = {d:0. for d in loaders}
        start = time.time()
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            for d in loaders:
                model.domain = d
                b = next(its[d])
                loss = criterion(model(b['eeg'].cuda()), b['clean'].cuda())
                assert torch.isfinite(loss)
                (.5*loss).backward()
                total[d] += float(loss.detach())
            nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            opt.step()
            sched.step()
            ema.update(model)
            if step % 100 == 0:
                write(dest/'status.json', dict(state='training', epoch=epoch, step=step, steps=steps))
                print('TRAIN', epoch, step, steps, {d:v/(step+1) for d,v in total.items()}, flush=True)
        results = {}
        for variant in ['raw', 'ema0.9998']:
            write(dest/'status.json', dict(state='evaluating', epoch=epoch, variant=variant))
            m = model if variant == 'raw' else evaluator
            if variant != 'raw': m.load_state_dict(ema.state_dict(model), strict=True)
            results[variant] = {d:evaluate(m,ds,d) for d,ds in tests.items()}
            score = np.mean([r['single_trial']['top5'] for r in results[variant].values()])
            path = dest/f'{variant}_epoch_{epoch:03d}.pt'
            torch.save(dict(model=m.state_dict(), epoch=epoch, variant=variant, config=cfg,
                            metrics=results[variant], manifest=manifest,
                            alljoined_channels=meta['channels']), path)
            rankings.append(dict(epoch=epoch, variant=variant, score=float(score), path=path.name))
        rankings.sort(key=lambda r:-r['score'])
        new = rankings[0]['score']
        stale = 0 if new > best else stale+1
        best = new
        write(dest/'best.json', rankings[0])
        write(dest/'top3.json', rankings[:3])
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(model), optimizer=opt.state_dict(),
                        scheduler=sched.state_dict(), epoch=epoch, config=cfg, best=best, stale=stale,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                        numpy_rng=np.random.get_state(), python_rng=random.getstate()), dest/'last.pt')
        row = dict(epoch=epoch, loss={d:v/steps for d,v in total.items()}, metrics=results,
                   seconds=time.time()-start, stale=stale, lr=opt.param_groups[0]['lr'])
        with (dest/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print('EPOCH', epoch, 'BEST', rankings[0], 'stale', stale, flush=True)
        if stale >= cfg['patience']: break
    write(dest/'status.json', dict(state='complete', epoch=epoch, best=rankings[0]))


if __name__ == '__main__':
    main()
