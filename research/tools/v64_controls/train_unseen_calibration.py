"""Source-unseen subject calibration transfer; never fit on the evaluation person.

After source-only S3--S10 training: fit a pooled512 residual on S2 and evaluate
S1, and vice versa. A matched seen-source residual is the protocol control.
This is a strict held-out calibration transfer pilot, not a reproduction of
MLDG or a claim of subject-independent statistics from a single EEG.
"""
import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_origin(path, *, content=True):
    """Hash small inputs; pin large immutable EEG files by path/size/mtime."""
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    value = dict(path=str(path), bytes=stat.st_size)
    if content:
        value['sha256'] = digest(path)
    else:
        value['mtime_ns'] = stat.st_mtime_ns
    return value


def cache_origin(cfg, arm, plan, provenance):
    """One source fingerprint for both partial and completed feature caches."""
    snapshot = Path(cfg['source_snapshot'])
    recorded = json.loads(Path(cfg['source_hashes_path']).read_text())
    actual = {name: digest(snapshot / name) for name in recorded}
    assert actual == recorded, 'Immutable source snapshot changed'
    assert digest(__file__) == actual['train_unseen_calibration.py'], 'Runner differs from source snapshot'
    data = Path(cfg['source_root']) / 'data'
    return dict(schema=2, checkpoint=provenance, records_sha256=object_digest(plan),
                source_config=file_origin(arm / 'config.json'), source_code_sha256=actual,
                source_hash_manifest=file_origin(cfg['source_hashes_path']),
                data_manifest=file_origin(data / 'manifest.json'),
                validation_ids=file_origin(cfg['validation_ids_path']),
                metadata={f'{split}_{kind}': file_origin(data / f'{split}_{kind}.npy')
                          for split in ['train', 'val', 'test'] for kind in ['ids', 'subjects']},
                eeg={split: file_origin(data / f'{split}_eeg.npy', content=False)
                     for split in ['train', 'val', 'test']},
                eeg_integrity='resolved path, file size and mtime_ns; EEG bytes are not rehashed in full',
                dtype='float32', extraction='eval; independent single-trial pooled REVE512; no augmentation')


def validate_cached_array(cache, key, records, entry, fingerprint):
    assert entry['origin_fingerprint'] == fingerprint, f'Cache source changed: {key}'
    assert entry['file'] == f'{key}.npy' and entry['rows'] == len(records), key
    path = cache / entry['file']
    assert digest(path) == entry['sha256'], f'Cache content changed: {key}'
    array = np.load(path, mmap_mode='r')
    assert array.shape == (len(records), 512) and array.dtype == np.dtype('float32'), key


def trial_reusable(dest, origin):
    """Only a fully completed, exactly matched trial can be reused.

    An interrupted optimization trajectory is retained for inspection. Restart
    in a new experiment directory instead of replacing its best/history files.
    """
    if not dest.exists() or not any(dest.iterdir()):
        return False
    marker, result_path = dest / 'origin.json', dest / 'result.json'
    if not marker.is_file() or not result_path.is_file():
        raise FileExistsError(f'Incomplete trial retained; refusing to overwrite: {dest}')
    expected = object_digest(origin)
    saved = json.loads(marker.read_text())
    assert saved == dict(fingerprint=expected, origin=origin), f'Trial provenance changed: {dest}'
    result = json.loads(result_path.read_text())
    assert result['fingerprint'] == expected and result['state'] == 'complete', dest
    assert set(result['artifacts_sha256']) == {'best.pt', 'last.pt', 'history.json'}, dest
    for name, sha in result['artifacts_sha256'].items():
        assert digest(dest / name) == sha, f'Trial artifact changed: {dest / name}'
    return True


def plan_records(cfg):
    """Identical EEG rows across the baseline and identity source models."""
    data = Path(cfg['source_root']) / 'data'
    excluded = sorted(cfg['excluded_subjects'])
    assert excluded == ['Gifford2022Large/1', 'Gifford2022Large/2']
    val_ids = set(json.loads(Path(cfg['validation_ids_path']).read_text()))
    ids = {s: np.load(data / f'{s}_ids.npy') for s in ['train', 'val', 'test']}
    subs = {s: np.load(data / f'{s}_subjects.npy').astype(str) for s in ids}
    test_ids = set(map(int, ids['test']))
    groups = {}
    for split in ['train', 'val']:
        for row, (subject, image) in enumerate(zip(subs[split], ids[split])):
            groups.setdefault((str(subject), int(image)), []).append([split, row])
    source = sorted(set(subs['train']) - set(excluded))
    assert len(source) == 8
    all_subjects = source + excluded
    common = sorted(set.intersection(*[
        {image for subject, image in groups if subject == s} for s in all_subjects
    ]))
    assert len(common) == 16540 and not (set(common) & test_ids)
    assert val_ids <= set(common)
    rng = np.random.default_rng(930033)
    pools = {key: {split: [] for split in ['fit', 'dev']} for key in ['seen', 'novel01', 'novel02']}
    for image in common:
        split = 'dev' if image in val_ids else 'fit'
        # Equal in expectation over the eight source people; no target EEG used.
        chosen_source = source[int(rng.integers(len(source)))]
        for key, subject in [('seen', chosen_source), ('novel01', excluded[0]), ('novel02', excluded[1])]:
            choices = groups[(subject, image)]
            original_split, row = choices[int(rng.integers(len(choices)))]
            pools[key][split].append(dict(split=original_split, row=int(row), image=image, subject=subject))
    tests = {}
    for n, subject in enumerate(excluded, 1):
        rows = np.flatnonzero(subs['test'] == subject)
        assert len(np.unique(ids['test'][rows])) == 200
        tests[f'test{n:02d}'] = [dict(split='test', row=int(r), image=int(ids['test'][r]), subject=subject) for r in rows]
    arrays = {f'{pool}_{split}': rows for pool, splits in pools.items() for split, rows in splits.items()}
    arrays.update(tests)
    for split in ['fit', 'dev']:
        ref = [r['image'] for r in arrays[f'seen_{split}']]
        for pool in ['novel01', 'novel02']:
            assert ref == [r['image'] for r in arrays[f'{pool}_{split}']]
    assert all(r['subject'] in source for key, rows in arrays.items() if key.startswith('seen_') for r in rows)
    return dict(source_subjects=source, excluded_subjects=excluded, arrays=arrays,
                fit_images=len(pools['seen']['fit']), dev_images=len(pools['seen']['dev']),
                rule='one fixed real single trial per image; paired image/trial records across model arms; all valid test trials',
                selection_seed=930033)


def select_checkpoint(cfg, arm):
    result = json.loads((arm / 'stage1/result.json').read_text())
    variant = max(result['best'], key=result['best'].get)
    epoch = result['best_epoch'][variant]
    name = f'epoch_{epoch:03d}.pt' if variant == 'raw' else f'{variant}_epoch_{epoch:03d}.pt'
    path = arm / 'stage1' / name
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert sorted(ck['config']['excluded_subjects']) == sorted(cfg['excluded_subjects'])
    assert cfg['initialization_kind'] == ck['config']['initialization_kind'] == 'original_pretrained_reve_fresh_heads'
    return path, ck, dict(path=str(path), sha256=digest(path), variant=variant, epoch=epoch,
                         selection='S3--S10 image-disjoint validation blur single Top5 only')


def load_model(cfg, checkpoint):
    import extensions as ex
    ex.CFG.clear()
    ex.CFG.update(cfg)
    cls = ex.IdentityModel if cfg.get('identity_weight') else ex.AugmentedModel
    model = cls(checkpoint['manifest'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    return model.cuda()


@torch.no_grad()
def pooled(model, x):
    x = model.preprocess(x)
    positions = model.positions.to(x).unsqueeze(0).expand(len(x), -1, -1)
    return model.encoder(x, pos=positions, return_output=True)[-1].mean(1)


def extract(cfg, arm, root, plan, checkpoint, provenance):
    cache = arm / 'calibration_cache'
    cache.mkdir(exist_ok=True)
    origin = cache_origin(cfg, arm, plan, provenance)
    fingerprint = object_digest(origin)
    marker = dict(fingerprint=fingerprint, origin=origin)
    origin_path = cache / 'origin.json'
    if origin_path.exists():
        assert json.loads(origin_path.read_text()) == marker, 'Partial/complete cache source changed'
    else:
        assert not any(cache.iterdir()), 'Untracked partial cache retained; use a new experiment directory'
        write(origin_path, marker)
    manifest = dict(**marker, arrays={})
    old_path = cache / 'complete.json'
    if old_path.exists():
        old = json.loads(old_path.read_text())
        assert old['fingerprint'] == fingerprint and old['origin'] == origin
        assert set(old['arrays']) == set(plan['arrays']), 'Incomplete array set in complete.json'
        for key, entry in old['arrays'].items():
            validate_cached_array(cache, key, plan['arrays'][key], entry, fingerprint)
        return cache
    data = Path(cfg['source_root']) / 'data'
    eeg = {s: np.load(data / f'{s}_eeg.npy', mmap_mode='r') for s in ['train', 'val', 'test']}
    model = load_model(cfg, checkpoint)
    first = plan['arrays']['seen_fit'][:8]
    x = torch.tensor(np.stack([eeg[r['split']][r['row']] for r in first]), device='cuda')
    with torch.no_grad():
        e = pooled(model, x)
        h, z, _ = model.encode_eeg(x)
        torch.testing.assert_close(model.blur_head(e), h, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(model.recovery(model.blur_head(e)), z, atol=2e-5, rtol=2e-5)
    for key, records in plan['arrays'].items():
        path = cache / f'{key}.npy'
        done = cache / f'{key}.json'
        if done.exists():
            entry = json.loads(done.read_text())
            validate_cached_array(cache, key, records, entry, fingerprint)
            manifest['arrays'][key] = entry
            continue
        assert not path.exists(), f'Incomplete cache array retained; refusing to overwrite: {path}'
        values = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(records), 512))
        for start in range(0, len(records), 128):
            block = records[start:start + 128]
            x = torch.tensor(np.stack([eeg[r['split']][r['row']] for r in block]), device='cuda')
            e = pooled(model, x).float().cpu().numpy()
            assert np.isfinite(e).all()
            values[start:start + len(e)] = e
            if start % 2048 == 0:
                write(arm / 'pipeline_status.json', dict(state='extracting', array=key, done=start + len(e), total=len(records), pid=os.getpid()))
                print('EXTRACT', key, start + len(e), len(records), flush=True)
        values.flush()
        del values
        entry = dict(file=path.name, sha256=digest(path), rows=len(records), origin_fingerprint=fingerprint)
        write(done, entry)
        manifest['arrays'][key] = entry
    del model
    torch.cuda.empty_cache()
    write(old_path, manifest)
    return cache


def targets(cfg, model, plan, device, test_blur_shards):
    data = Path(cfg['source_root']) / 'data'
    clean = torch.tensor(np.load(data / 'clean.npy'), device=device)
    clean = F.normalize(clean.float(), dim=-1)
    raw_blur = np.load(Path(cfg['source_root']) / 'features/blur.npy', mmap_mode='r')
    ids = sorted({r['image'] for rows in plan['arrays'].values() for r in rows})
    test_ids = sorted({r['image'] for r in plan['arrays']['test01']})
    assert test_ids == sorted({r['image'] for r in plan['arrays']['test02']})
    raw = np.asarray(raw_blur).copy()
    for image in test_ids:
        with np.load(Path(test_blur_shards) / f'{image:05d}.npz') as f:
            assert int(f['id']) == image
            raw[image] = f['blur'].copy()
    blur = torch.zeros_like(clean)
    with torch.no_grad():
        for start in range(0, len(ids), 256):
            ii = ids[start:start + 256]
            blur[ii] = F.normalize(model.image_adapter(torch.tensor(raw[ii], device=device)), dim=-1)
    assert torch.isfinite(clean[ids]).all() and torch.isfinite(blur[ids]).all()
    return dict(clean=clean, blur=blur)


@torch.no_grad()
def evaluate(model, x, image_ids, banks, enabled=True):
    model.eval()
    device = next(model.parameters()).device
    unique, inverse = np.unique(np.asarray(image_ids), return_inverse=True)
    labels = torch.tensor(inverse, device=device)
    out = {}
    predictions = {s: [] for s in ['blur', 'clean']}
    ratio = []
    for block in x.split(256):
        p = model(block.to(device), enabled=enabled)
        for space in predictions:
            assert torch.isfinite(p[space]).all(), f'Non-finite {space} predictions'
            predictions[space].append(p[space])
        ratio.append((p['pooled'] - block.to(device)).norm(dim=1) / block.to(device).norm(dim=1).clamp_min(1e-8))
    counts = torch.bincount(labels, minlength=len(unique))
    for space, chunks in predictions.items():
        pred = torch.cat(chunks)
        gallery = banks[space][unique]
        assert torch.isfinite(gallery).all(), f'Non-finite {space} gallery'
        sums = pred.new_zeros(len(unique), pred.shape[1])
        sums.index_add_(0, labels, pred)
        means = sums / counts[:, None]
        out[space] = {}
        for name, q, y in [('single_trial', pred, labels), ('subject_aggregated', means, torch.arange(len(unique), device=device))]:
            hits1 = hits5 = 0
            ranks = 0.0
            for block, truth in zip(q.split(512), y.split(512)):
                score = block @ gallery.T
                assert torch.isfinite(score).all(), f'Non-finite {space} scores'
                top = score.topk(min(5, len(unique)), dim=1).indices
                hits1 += int((top[:, 0] == truth).sum())
                hits5 += int((top == truth[:, None]).any(1).sum())
                target = score.gather(1, truth[:, None])
                ranks += float((1 + (score > target).sum(1) + .5 * ((score == target).sum(1) - 1)).sum())
            out[space][name] = dict(top1=hits1 / len(y), top5=hits5 / len(y), mean_rank=ranks / len(y), count=len(y))
    out.update(repeats_min=int(counts.min()), repeats_max=int(counts.max()), candidate_count=len(unique),
               correction_ratio=float(torch.cat(ratio).mean()))
    return out


def calibration_origin(cfg, arm, plan, cache, provenance, settings):
    test_ids = sorted({r['image'] for r in plan['arrays']['test01']})
    root = Path(cfg['source_root'])
    return dict(schema=2, checkpoint=provenance, settings=settings,
                protocol=file_origin(arm.parent / 'calibration_protocol.json'),
                feature_cache=file_origin(cache / 'complete.json'),
                clean=file_origin(root / 'data/clean.npy'),
                training_blur=file_origin(root / 'features/blur.npy'),
                test_blur_shards={str(image): file_origin(Path(settings['test_blur_shards']) / f'{image:05d}.npz')
                                  for image in test_ids},
                records_sha256=object_digest(plan),
                runner_sha256=digest(__file__))


def calibration_preflight(model, x, image_ids, banks, settings):
    """Two disposable C updates on real fit rows and frozen native heads.

    Only the fitting split is used. Restore the zero-step C and RNG afterward;
    this acceptance check does not initialize or select a production trial.
    """
    assert x.ndim == 2 and x.shape == (8, 512)
    assert len(set(image_ids)) == len(image_ids) == 8
    device = next(model.parameters()).device
    x = x.to(device)
    labels = torch.arange(len(x), device=device)
    image_ids = torch.tensor(image_ids, device=device)
    original = {k: p.detach().clone() for k, p in model.calibrator.state_dict().items()}
    frozen = {k: p.detach().clone() for k, p in model.state_dict().items() if not k.startswith('calibrator.')}
    trainable = dict(model.calibrator.named_parameters())
    assert sum(p.numel() for p in trainable.values()) == 67136
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    diagnostics = []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(930033)
            for module in model.calibrator.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            nn.init.zeros_(model.calibrator.net[-1].weight)
            nn.init.zeros_(model.calibrator.net[-1].bias)
            model.eval()
            with torch.no_grad():
                baseline = model(x, enabled=False)
                zero = model(x, enabled=True)
                for key in baseline:
                    torch.testing.assert_close(zero[key], baseline[key], atol=0, rtol=0)
            optimizer = torch.optim.AdamW(trainable.values(), lr=settings['lr'], weight_decay=settings['weight_decay'])
            model.train()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                clean_loss = F.cross_entropy(prediction['clean'] @ banks['clean'][image_ids].T, labels)
                blur_loss = F.cross_entropy(prediction['blur'] @ banks['blur'][image_ids].T, labels)
                loss = clean_loss + blur_loss
                assert torch.isfinite(loss), 'Non-finite tiny-set loss'
                component_grads = {}
                for name, component in [('clean', clean_loss), ('blur', blur_loss)]:
                    gradient = torch.autograd.grad(component, model.calibrator.net[-1].weight, retain_graph=True)[0]
                    assert torch.isfinite(gradient).all() and gradient.norm() > 0, name
                    component_grads[name] = float(gradient.norm())
                loss.backward()
                grads = {}
                for name, parameter in trainable.items():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                    grads[name] = float(parameter.grad.norm())
                assert grads['net.3.weight'] > 0
                if step == 0:
                    assert grads['net.1.weight'] == 0, 'Zero-last-layer initialization violated'
                else:
                    assert all(value > 0 for value in grads.values()), grads
                norm = nn.utils.clip_grad_norm_(trainable.values(), 1, error_if_nonfinite=True)
                optimizer.step()
                diagnostics.append(dict(step=step + 1, loss=float(loss.detach()),
                                        clean_loss=float(clean_loss.detach()), blur_loss=float(blur_loss.detach()),
                                        gradient_norm=float(norm), per_parameter_gradient_norm=grads,
                                        component_gradient_norm=component_grads))
            model.eval()
            with torch.no_grad():
                expected = model(x)
                for key in baseline:
                    torch.testing.assert_close(model(x, enabled=False)[key], baseline[key], atol=0, rtol=0)
                order = torch.tensor([5, 1, 7, 0, 3, 6, 2, 4], device=device)
                permuted = model(x[order])
                singletons = [model(row[None]) for row in x]
                for key in expected:
                    torch.testing.assert_close(permuted[key], expected[key][order], atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(torch.cat([row[key] for row in singletons]), expected[key], atol=2e-5, rtol=2e-5)
                assert not torch.equal(expected['pooled'], baseline['pooled']), 'C did not update'
            with io.BytesIO() as buffer:
                torch.save(model.calibrator.state_dict(), buffer)
                model.calibrator.load_state_dict(original, strict=True)
                buffer.seek(0)
                model.calibrator.load_state_dict(torch.load(buffer, map_location=device, weights_only=True), strict=True)
            with torch.no_grad():
                reloaded = model(x)
                for key in expected:
                    torch.testing.assert_close(reloaded[key], expected[key], atol=0, rtol=0)
            for key, value in frozen.items():
                assert torch.equal(model.state_dict()[key], value), f'Frozen state changed: {key}'
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    finally:
        model.calibrator.load_state_dict(original, strict=True)
        model.zero_grad(set_to_none=True)
        model.eval()
    return dict(passed=True, samples=8, steps=2, device=str(device), torch_version=torch.__version__,
                data='first eight distinct seen_fit images; no development/test data',
                zero_step_tensor_exact=True, original_heads_unchanged=True, bypass_exact=True,
                save_reload_exact=True, batch_order_tolerance=2e-5,
                production_initialization_unchanged=True, gradients=diagnostics)


def train_calibrators(cfg, arm, plan, cache, ck, provenance, settings):
    from unseen_subject_calibrator import CalibratedReadout
    assert settings['selection_space'] == 'blur', 'Calibration selection must use the prespecified blur development rank'
    assert settings['epochs'] >= 1 and settings['patience'] >= 1 and settings['batch_size'] >= 2
    device = torch.device('cuda')
    base = CalibratedReadout.from_checkpoint_model_state(ck['model']).to(device)
    banks = targets(cfg, base, plan, device, settings['test_blur_shards'])
    frozen_before = {name: p.detach().clone() for name, p in base.named_parameters() if not p.requires_grad}
    arrays = {key: torch.from_numpy(np.load(cache / f'{key}.npy')) for key in plan['arrays']}
    ids = {key: [r['image'] for r in rows] for key, rows in plan['arrays'].items()}
    output = arm / 'calibration'
    output.mkdir(exist_ok=True)
    origin = calibration_origin(cfg, arm, plan, cache, provenance, settings)
    fingerprint = object_digest(origin)
    marker = dict(fingerprint=fingerprint, origin=origin)
    if (output / 'origin.json').exists():
        assert json.loads((output / 'origin.json').read_text()) == marker, 'Calibration inputs/settings changed'
    else:
        assert not any(output.iterdir()), 'Untracked calibration artifacts retained; use a new experiment directory'
        write(output / 'origin.json', marker)
    # Check every existing trial before recomputing baseline reports or fitting.
    reusable = {}
    for pool in ['seen', 'novel01', 'novel02']:
        for seed in settings['seeds']:
            trial_origin = dict(calibration_fingerprint=fingerprint, pool=pool, seed=seed)
            reusable[pool, seed] = trial_reusable(output / f'{pool}_seed{seed}', trial_origin)
    preflight = calibration_preflight(base, arrays['seen_fit'][:8], ids['seen_fit'][:8], banks, settings)
    write(output / 'preflight.json', dict(fingerprint=fingerprint, checkpoint=provenance,
                                        fit_records=plan['arrays']['seen_fit'][:8], **preflight))
    # Test metrics are first computed here, after source checkpoint selection.
    # They never enter training/early stopping or source selection.
    zero = {key: evaluate(base, arrays[key], ids[key], banks, enabled=False) for key in ['test01', 'test02']}
    write(output / 'zero.json', dict(metrics=zero, checkpoint=provenance, fingerprint=fingerprint))
    for key in ['test01', 'test02']:
        with torch.no_grad():
            actual = base(arrays[key][:16].to(device), enabled=True)
            expected = base(arrays[key][:16].to(device), enabled=False)
        for space in actual:
            torch.testing.assert_close(actual[space], expected[space], atol=0, rtol=0)
    finished = []
    for pool in ['seen', 'novel01', 'novel02']:
        tests = ['test01', 'test02'] if pool == 'seen' else ['test02' if pool == 'novel01' else 'test01']
        fit_subjects = {r['subject'] for r in plan['arrays'][f'{pool}_fit']}
        for key in tests:
            assert not fit_subjects & {r['subject'] for r in plan['arrays'][key]}
        for seed in settings['seeds']:
            dest = output / f'{pool}_seed{seed}'
            if reusable[pool, seed]:
                finished.append(str(dest))
                continue
            dest.mkdir(exist_ok=True)
            trial_origin = dict(calibration_fingerprint=fingerprint, pool=pool, seed=seed)
            trial_fingerprint = object_digest(trial_origin)
            write(dest / 'origin.json', dict(fingerprint=trial_fingerprint, origin=trial_origin))
            torch.manual_seed(seed)
            model = CalibratedReadout.from_checkpoint_model_state(ck['model']).to(device)
            params = [p for p in model.parameters() if p.requires_grad]
            assert sum(p.numel() for p in params) == 67136
            opt = torch.optim.AdamW(params, lr=settings['lr'], weight_decay=settings['weight_decay'])
            fit = arrays[f'{pool}_fit']
            dev = arrays[f'{pool}_dev']
            fit_ids = torch.tensor(ids[f'{pool}_fit'], device=device)
            dev_ids = ids[f'{pool}_dev']
            assert not set(fit_ids.tolist()) & set(dev_ids)
            assert not set(fit_ids.tolist() + dev_ids) & set(ids['test01'])
            assert len(set(fit_ids.tolist())) == len(fit_ids)
            dev_zero = evaluate(model, dev, dev_ids, banks)
            best = dev_zero['blur']['single_trial']['mean_rank']
            best_epoch = 0
            stale = 0
            torch.save(dict(calibrator=model.calibrator.state_dict(), epoch=0, checkpoint=provenance, seed=seed, pool=pool), dest / 'best.pt')
            history = [dict(epoch=0, development=dev_zero, best_epoch=0)]
            gen = torch.Generator().manual_seed(seed + 930000)
            for epoch in range(1, settings['epochs'] + 1):
                write(arm / 'pipeline_status.json', dict(state='calibrating', pool=pool, seed=seed, epoch=epoch, completed=len(finished), total=3 * len(settings['seeds']), pid=os.getpid()))
                model.train()
                losses, norms, ratios = [], [], []
                order = torch.randperm(len(fit), generator=gen)
                for ix in order.split(settings['batch_size']):
                    if len(ix) < 2:
                        continue
                    x = fit[ix].to(device)
                    image = fit_ids[ix.to(device)]
                    pred = model(x, enabled=True)
                    truth = torch.arange(len(ix), device=device)
                    lc = F.cross_entropy(pred['clean'] @ banks['clean'][image].T, truth)
                    lb = F.cross_entropy(pred['blur'] @ banks['blur'][image].T, truth)
                    loss = lc + lb
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    assert torch.isfinite(loss)
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                    norms.append(float(nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)))
                    opt.step()
                    losses.append([float(loss.detach()), float(lc.detach()), float(lb.detach())])
                    ratios.append(float(((pred['pooled'] - x).norm(dim=1) / x.norm(dim=1).clamp_min(1e-8)).mean().detach()))
                development = evaluate(model, dev, dev_ids, banks)
                score = development['blur']['single_trial']['mean_rank']
                if score < best:
                    best, best_epoch, stale = score, epoch, 0
                    torch.save(dict(calibrator=model.calibrator.state_dict(), epoch=epoch, checkpoint=provenance, seed=seed, pool=pool), dest / 'best.pt')
                else:
                    stale += 1
                row = dict(epoch=epoch, loss=np.mean(losses, axis=0).tolist(), loss_names=['total', 'clean', 'blur'],
                           grad_norm=float(np.mean(norms)), correction_ratio=float(np.mean(ratios)),
                           development=development, best_epoch=best_epoch, stale=stale)
                history.append(row)
                write(dest / 'history.json', history)
                torch.save(dict(calibrator=model.calibrator.state_dict(), optimizer=opt.state_dict(), epoch=epoch), dest / 'last.pt')
                print('CALIBRATE', pool, seed, epoch, 'dev_blur_mean_rank', score, 'best_epoch', best_epoch, flush=True)
                if stale >= settings['patience']:
                    break
            for name, p in model.named_parameters():
                if name in frozen_before:
                    assert p.grad is None and torch.equal(p.detach(), frozen_before[name]), name
            model.calibrator.load_state_dict(torch.load(dest / 'best.pt', map_location=device, weights_only=False)['calibrator'])
            # One final evaluation of the dev-selected checkpoint; no test selection.
            result = dict(state='complete', fingerprint=trial_fingerprint,
                          pool=pool, seed=seed, best_epoch=best_epoch, trained_epochs=epoch,
                          development_selection='minimum blur mean correct-image rank; ties retain earlier epoch; epoch0 allowed',
                          checkpoint=provenance, train_subjects=sorted(fit_subjects), fit_images=len(fit),
                          fit_trials=len(fit), dev_images=len(dev), trainable_parameters=sum(p.numel() for p in params),
                          tests={key: dict(zero=zero[key], calibrated=evaluate(model, arrays[key], ids[key], banks)) for key in tests},
                          artifacts_sha256={name: digest(dest / name) for name in ['best.pt', 'last.pt', 'history.json']})
            write(dest / 'result.json', result)
            finished.append(str(dest))
            del model, opt
            torch.cuda.empty_cache()
    write(output / 'status.json', dict(state='complete', fingerprint=fingerprint, completed=len(finished), total=3 * len(settings['seeds'])))


def main(args):
    cfg = json.loads(args.config.read_text())
    arm = Path(cfg['run_root'])
    root = arm.parent
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = json.loads((root / 'calibration_protocol.json').read_text())
    plan = plan_records(cfg)
    plan_path = root / 'calibration_records.json'
    # Planner is created once before any workers launch; never race on a shared file.
    assert plan_path.exists() and json.loads(plan_path.read_text()) == plan
    cp, ck, provenance = select_checkpoint(cfg, arm)
    write(arm / 'calibration_source.json', provenance)
    cache = extract(cfg, arm, root, plan, ck, provenance)
    train_calibrators(cfg, arm, plan, cache, ck, provenance, settings)
    write(arm / 'pipeline_status.json', dict(state='complete', pid=os.getpid(), completed_at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        cfg = json.loads(args.config.read_text())
        write(Path(cfg['run_root']) / 'pipeline_status.json', dict(state='failed', error=repr(exc), failed_at=time.time(), pid=os.getpid()))
        raise
