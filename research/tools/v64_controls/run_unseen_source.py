"""Fresh paired source models: S3--S10 only; S1/S2 are never evaluated here.

Run with the immutable v91 dependencies on PYTHONPATH. This module changes the
source-subject filter and validation scope, not the v79/v91 optimization recipe.
"""
import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

import extensions as ex
import local_ot_train_ema as base
from augment_trials import MixedTrials
from data import DatasetEEG


EXCLUDED = ['Gifford2022Large/1', 'Gifford2022Large/2']
SOURCES = [f'Gifford2022Large/{i}' for i in range(3, 11)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def masks(subjects, ids, cfg):
    """The same allow-list is used by fitting and source validation."""
    source = np.isin(np.asarray(subjects).astype(str), cfg['source_subjects'])
    excluded = np.isin(np.asarray(subjects).astype(str), cfg['excluded_subjects'])
    assert not np.any(source & excluded)
    val_ids = np.asarray(json.loads(Path(cfg['validation_ids_path']).read_text()))
    is_val = np.isin(ids, val_ids)
    return source & ~is_val, source & is_val


class SourceTrials(MixedTrials):
    def __init__(self, ds):
        parts = [ex.View(d, masks(d.subjects, d.ids, ex.CFG)[0]) for d in ds.datasets]
        super().__init__(ConcatDataset(parts))
        assert set(k[0] for k in self.keys) == set(ex.CFG['source_subjects'])
        assert not set(k[0] for k in self.keys) & set(ex.CFG['excluded_subjects'])


@torch.no_grad()
def evaluate_source(model, cfg, weights):
    """Source image-heldout validation only. No target/test EEG is loaded."""
    model.eval()
    parts = []
    for split in ['train', 'val']:
        d = DatasetEEG(cfg['source_root'], split, aux=False)
        parts.append(ex.View(d, masks(d.subjects, d.ids, cfg)[1]))
    ds = ConcatDataset(parts)
    labels = np.concatenate([d.ids for d in parts])
    subjects = np.concatenate([d.subjects for d in parts]).astype(str)
    assert set(subjects) == set(cfg['source_subjects'])
    ids = np.unique(labels)
    val_ids = np.asarray(json.loads(Path(cfg['validation_ids_path']).read_text()))
    assert np.array_equal(ids, val_ids)
    clean = parts[0].clean
    blur = np.load(Path(cfg['source_root']) / 'features/blur.npy', mmap_mode='r')[ids].copy()
    gallery = F.normalize(model.image_adapter(torch.tensor(blur, device='cuda')), dim=-1).cpu().numpy()
    hs, zs = [], []
    for b in DataLoader(ds, batch_size=128, num_workers=2):
        h, z, _ = model.encode_eeg(b['eeg'].cuda())
        hs.append(h.cpu().numpy())
        zs.append(z.cpu().numpy())
    allblur = np.zeros_like(clean)
    allblur[ids] = gallery
    return dict(blur=ex.metrics(np.concatenate(hs), labels, subjects, allblur),
                clean=ex.metrics(np.concatenate(zs), labels, subjects, clean),
                split='source_image_disjoint_validation', subjects=sorted(set(subjects)))


def audit(cfg, out):
    assert cfg['excluded_subjects'] == EXCLUDED
    assert cfg['source_subjects'] == SOURCES
    assert not cfg.get('heldout_subject')
    for key in ['warm_start', 'resume', 'checkpoint', 'base_checkpoint', 'head_checkpoint', 'initial_reference']:
        assert not cfg.get(key), f'Retrieval checkpoint initialization is forbidden: {key}'
    assert cfg['seed'] == 33 and cfg['batch_size'] == 256
    assert cfg['checkpoint_group'] == 'single_trial'
    assert cfg['selection_split'] == 'val_source_subjects'
    assert cfg['training_splits'] == ['train', 'val']
    assert cfg.get('identity_weight') in [0., 1.]
    assert cfg['identity_location'] == 'pooled512'
    assert cfg['identity_temperature'] == .1
    assert cfg['initialization_kind'] == 'original_pretrained_reve_fresh_heads'
    assert os.environ.get('HF_HUB_OFFLINE') == '1', 'Offline pretrained loading required'
    hf = Path(os.environ['HF_HOME']) / 'hub/models--brain-bzh--reve-base'
    assert (hf / 'refs/main').read_text().strip() == cfg['pretrained_snapshot']
    assert (hf / 'snapshots' / cfg['pretrained_snapshot']).is_dir()
    snapshot = Path(cfg['source_snapshot'])
    recorded = json.loads(Path(cfg['source_hashes_path']).read_text())
    for name, digest in recorded.items():
        assert sha256(snapshot / name) == digest, f'Source changed: {name}'
    root = Path(cfg['source_root']) / 'data'
    ids = np.concatenate([np.load(root / f'{s}_ids.npy') for s in ['train', 'val']])
    subs = np.concatenate([np.load(root / f'{s}_subjects.npy') for s in ['train', 'val']]).astype(str)
    tr, va = masks(subs, ids, cfg)
    # Test image IDs are metadata used only for a disjointness assertion.
    testids = np.unique(np.load(root / 'test_ids.npy'))
    assert not np.intersect1d(ids[tr | va], testids).size
    assert not np.intersect1d(ids[tr], ids[va]).size
    assert set(subs[tr]) == set(SOURCES) == set(subs[va])
    assert not np.any(np.isin(subs[tr | va], EXCLUDED))
    val_ids = json.loads(Path(cfg['validation_ids_path']).read_text())
    weights = json.loads((out / 'predictability/subjects.json').read_text())
    provenance = json.loads((out / 'predictability/complete.json').read_text())
    assert set(weights) == set(SOURCES) == set(provenance['subjects'])
    assert provenance['validation_ids_excluded'] == val_ids
    assert provenance['test_used'] is False
    assert all(v['images'] == len(np.unique(ids[tr])) for v in weights.values())
    row = dict(excluded_subjects=EXCLUDED, source_subjects=SOURCES,
               train_count=int(tr.sum()), validation_count=int(va.sum()),
               train_images=len(np.unique(ids[tr])), validation_images=len(val_ids),
               target_eeg_read=False, target_test_evaluated=False,
               predictability='source-only entries from independently fitted per-subject train-image regressions',
               initialization='original pretrained REVE; fresh random retrieval/OT heads',
               initialization_kind=cfg['initialization_kind'],
               foundation_note='Exclusion concerns downstream training; REVE foundation pretraining is not audited',
               selection='source image-disjoint validation blur single-trial Top5; best Raw/EMA',
               validation_ids_sha256=sha256(cfg['validation_ids_path']),
               config_sha256=sha256(out / 'config.json'),
               source_hashes_sha256=sha256(cfg['source_hashes_path']),
               train_order_sha256=hashlib.sha256(ids[tr].tobytes() + subs[tr].tobytes()).hexdigest(),
               validation_order_sha256=hashlib.sha256(ids[va].tobytes() + subs[va].tobytes()).hexdigest(),
               manifest_sha256=sha256(root / 'manifest.json'))
    ex.write(out / 'data_audit.json', row)
    return row


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def initialization_audit(cfg, out):
    manifest = json.loads((Path(cfg['source_root']) / 'data/manifest.json').read_text())
    digests = {}
    for name, cls in [('baseline', ex.AugmentedModel), ('identity', ex.IdentityModel)]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(33)
            model = cls(manifest)
            if name == 'identity':
                assert isinstance(model.identity_head, torch.nn.Identity)
            digests[name] = state_digest(model)
            del model
    assert digests['baseline'] == digests['identity'], 'Paired initialization differs'
    ex.write(out / 'initialization_audit.json', dict(passed=True, state_sha256=digests,
             pretrained_snapshot=cfg['pretrained_snapshot'], retrieval_checkpoint_loaded=False))


def select_source(cfg, out):
    result = json.loads((out / 'stage1/result.json').read_text())
    variant = max(result['best'], key=result['best'].get)
    epoch = result['best_epoch'][variant]
    name = f'epoch_{epoch:03d}.pt' if variant == 'raw' else f'{variant}_epoch_{epoch:03d}.pt'
    ckpt = out / 'stage1' / name
    ex.write(out / 'selected_source.json', dict(checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt),
             variant=variant, epoch=epoch, source_validation_blur_top5=result['best'][variant],
             selection='source validation only', excluded_subjects=EXCLUDED,
             target_eeg_read=False, target_test_evaluated=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['run_root'])
    assert out.resolve().is_relative_to(Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments'))
    assert args.config.resolve() == (out / 'config.json').resolve()
    if not args.audit_only:
        assert not (out / 'stage1').exists(), 'Refusing to overwrite an existing source run'
    try:
        ex.CFG.clear()
        ex.CFG.update(cfg)
        torch.set_num_threads(4)
        torch.manual_seed(33)
        np.random.seed(33)
        random.seed(33)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        audit(cfg, out)
        if args.audit_only:
            return
        ex.Trials = SourceTrials
        ex.evaluate = evaluate_source
        cls = ex.IdentityModel if cfg['identity_weight'] else ex.AugmentedModel
        if args.preflight:
            initialization_audit(cfg, out)
            preflight_cfg = dict(cfg)
            if not cfg['identity_weight']:
                # The shared preflight uses this key to inspect identity_head;
                # the baseline intentionally has no identity branch.
                preflight_cfg.pop('identity_location', None)
            ex.preflight(preflight_cfg, out, cls)
            check = json.loads((out / 'preflight.json').read_text())
            check.update(excluded_subjects=EXCLUDED, config_sha256=sha256(args.config),
                         source_hashes_sha256=sha256(cfg['source_hashes_path']))
            ex.write(out / 'preflight.json', check)
            return
        check = json.loads((out / 'preflight.json').read_text())
        assert check['passed'] and check['excluded_subjects'] == EXCLUDED
        assert check['config_sha256'] == sha256(args.config)
        assert check['source_hashes_sha256'] == sha256(cfg['source_hashes_path'])
        assert json.loads((out / 'initialization_audit.json').read_text())['passed']
        base.LocalOTModel, base.MixedTrials = cls, SourceTrials
        base.loss_fn, base.evaluate = ex.loss_fn, evaluate_source
        base.train(cfg, out)
        select_source(cfg, out)
        (out / 'queue_status.txt').write_text('complete\n')
    except BaseException as err:
        ex.write(out / 'failure.json', dict(type=type(err).__name__, message=str(err),
                 exited_at=datetime.now(timezone.utc).isoformat()))
        (out / 'queue_status.txt').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
