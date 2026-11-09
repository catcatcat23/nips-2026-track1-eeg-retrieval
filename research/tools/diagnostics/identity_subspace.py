"""Frozen-head, image-disjoint identity-subspace interventions on saved probes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
CACHE = BASE / 'subject_identity_probe_20261101'
RUNS = {'v70': 'reve_v70_clean_ema_control', 'v71': 'reve_v71_clean_ema_control',
        'v72': 'reve_v72_cross_subject_mix'}


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def fit(x, y, split):
    scaler = StandardScaler().fit(x[split == 0])
    z = scaler.transform(x)
    best = None
    scores = []
    for c in [.001, .01, .1, 1.]:
        clf = LogisticRegression(C=c, solver='lbfgs', max_iter=1000,
                                 tol=1e-5, random_state=33).fit(z[split == 0], y[split == 0])
        acc = float(clf.score(z[split == 1], y[split == 1]))
        scores.append({'C': c, 'validation_accuracy': acc})
        if best is None or acc > best[0]:
            best = acc, clf
    clf = best[1]
    # Convert the fitted standardized-coordinate classifier to raw feature coordinates.
    w = clf.coef_ / scaler.scale_[None, :]
    b = clf.intercept_ - w @ scaler.mean_
    np.testing.assert_allclose(x @ w.T + b, clf.decision_function(z), atol=1e-5, rtol=1e-5)
    return w, b, scaler, clf, scores


def metrics(z, ids, subjects, gallery, candidates):
    scores = torch.nn.functional.normalize(z.float(), dim=-1) @ gallery.T
    labels = torch.as_tensor(np.searchsorted(candidates, ids), device=z.device)
    positive = scores.gather(1, labels[:, None]).squeeze(1)
    wrong = scores.clone(); wrong.scatter_(1, labels[:, None], -torch.inf)
    rank = 1 + (wrong >= positive[:, None]).sum(1)
    margin = positive - wrong.max(1).values
    # Conservative ties, applied consistently to all conditions.
    return {k: v.cpu().numpy() for k, v in {'rank': rank, 'margin': margin,
            'top1': rank <= 1, 'top5': rank <= 5}.items()}


def summarize(values, subjects):
    per = {str(s): {k: float(v[subjects == s].mean()) for k, v in values.items()}
           for s in np.unique(subjects)}
    return {'per_subject': per, 'macro': {k: float(np.mean([p[k] for p in per.values()]))
                                         for k in values}}


def run(out):
    torch.set_num_threads(4); torch.manual_seed(33)
    torch.backends.cuda.matmul.allow_tf32 = False
    old_results = json.loads((CACHE / 'results.json').read_text())
    cfg = json.loads((BASE / RUNS['v70'] / 'config.json').read_text())
    tc = json.loads(Path(cfg['things_config']).read_text())
    ac = json.loads(Path(cfg['alljoined_config']).read_text())
    clean_paths = {'things': Path(tc['source_root']) / 'data/clean.npy',
                   'alljoined': Path(ac['clean_features'])}
    protocol = {'models': RUNS, 'checkpoint': 'epoch10 EMA0.9998', 'layer': 'pooled512',
                'domains': list(clean_paths), 'fit_validation_test': 'existing image-disjoint 50/25/25 split',
                'aggregation': 'mean predictions over 4 cached trials per person/image; not official 80',
                'gallery': 'all 200 original test images, including non-query images',
                'alphas': [0, .25, .5, .75, 1], 'random_subspaces': 10, 'donor_repeats': 3,
                'scope': 'exploratory public-test diagnostic; no model update; no causal physiology claim',
                'rank': 'full numerical rank for each model; report rank before comparing',
                'ties': 'conservative rank', 'random_control': 'per-sample matched perturbation norm',
                'same_cross': 'natural and cross scaled to same-person perturbation norm',
                'checkpoint_note': 'same checkpoint and cached selections as original identity probe'}
    write(out / 'protocol.json', protocol)
    all_results = []
    for name, run_name in RUNS.items():
        ckpath = BASE / run_name / 'stage1/ema0.9998_epoch_010.pt'
        ck = torch.load(ckpath, map_location='cpu', weights_only=False)
        weight = ck['model']['head.weight'].float().cuda()
        bias = ck['model']['head.bias'].float().cuda(); del ck
        for domain, cleanpath in clean_paths.items():
            prefix = f'{name}_{domain}'
            write(out / 'status.json', {'state': 'running', 'model': name, 'domain': domain, 'phase': 'fit_probe'})
            sel = np.load(CACHE / f'{domain}_selection.npz')
            y, split, ids = sel['subject'], sel['split'], sel['image']
            features = np.load(CACHE / f'{prefix}_features.npz')
            x = features['pooled'].astype(np.float64)
            assert set(ids[split == 0]).isdisjoint(ids[split == 2])
            assert set(ids[split == 1]).isdisjoint(ids[split == 2])
            w, b, scaler, clf, candidates_c = fit(x, y, split)
            wc = w - w.mean(0, keepdims=True)
            _, singular, vh = np.linalg.svd(wc, full_matrices=False)
            rank = int((singular > max(wc.shape) * np.finfo(wc.dtype).eps * singular[0]).sum())
            assert 0 < rank <= len(np.unique(y)) - 1
            q = vh[:rank].T; mu = x[split == 0].mean(0, keepdims=True)
            np.testing.assert_allclose(q.T @ q, np.eye(rank), atol=1e-8)
            te = split == 2; h = x[te]; labels = y[te]; image_ids = ids[te]
            identity = (h - mu) @ q @ q.T
            np.testing.assert_allclose((h - mu - identity) @ wc.T, 0, atol=1e-7)
            np.testing.assert_allclose(mu + identity + (h - mu - identity), h, atol=1e-10)
            np.savez(out / f'{prefix}_probe.npz', weight=w, bias=b, mean=mu, Q=q,
                     singular=singular, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                     standardized_weight=clf.coef_, standardized_bias=clf.intercept_, classes=clf.classes_)
            acc = float((np.argmax(h @ w.T + b, 1) == labels).mean())
            previous = next(r['result']['test_accuracy'] for r in old_results
                            if r['model'] == name and r['domain'] == domain and r['layer'] == 'pooled')
            print(prefix, 'probe', acc, 'previous', previous, 'rank', rank, flush=True)
            # Validate cached pooled features reproduce the actual saved clean outputs.
            def head(a):
                return torch.as_tensor(a, dtype=torch.float32, device='cuda') @ weight.T + bias
            with torch.no_grad():
                torch.testing.assert_close(head(h).cpu(), torch.from_numpy(features['projection_clean'][te]),
                                           atol=2e-4, rtol=2e-4)
            clean = np.load(cleanpath, mmap_mode='r'); candidates = np.unique(ids)
            assert len(candidates) == 200
            gallery = torch.nn.functional.normalize(torch.tensor(np.array(clean[candidates]), device='cuda').float(), dim=-1)
            groups = [np.flatnonzero((labels == s) & (image_ids == im))
                      for s in np.unique(labels) for im in np.unique(image_ids)]
            assert all(len(g) == 4 for g in groups)
            agg_ids = np.array([image_ids[g[0]] for g in groups])
            agg_sub = np.array([labels[g[0]] for g in groups])
            def evaluate(tag, changed, metadata):
                delta = changed - h
                with torch.no_grad():
                    z = head(changed)
                    single = metrics(z, image_ids, labels, gallery, candidates)
                    means = torch.stack([z[g.tolist()].mean(0) for g in groups])
                    agg = metrics(means, agg_ids, agg_sub, gallery, candidates)
                np.savez_compressed(out / f'{prefix}_{tag}.npz', **single, image=image_ids, subject=labels,
                                    delta_norm=np.linalg.norm(delta, axis=1), feature_norm=np.linalg.norm(h, axis=1),
                                    **{'agg_' + k: v for k, v in agg.items()}, agg_subject=agg_sub, agg_image=agg_ids)
                row = {'model': name, 'domain': domain, 'condition': tag, **metadata,
                       'single': summarize(single, labels), 'aggregate4': summarize(agg, agg_sub),
                       'probe_accuracy': float((np.argmax(changed @ w.T + b, 1) == labels).mean()),
                       'delta_norm_mean': float(np.linalg.norm(delta, axis=1).mean())}
                all_results.append(row); write(out / 'results.json', all_results)
            evaluate('original', h, {})
            for alpha in protocol['alphas']:
                evaluate(f'remove_a{alpha}', h - alpha * identity, {'alpha': alpha})
            for seed in range(10):
                rng = np.random.default_rng(100 + seed)
                qr = np.linalg.qr(rng.standard_normal((h.shape[1], rank)))[0]
                delta = -((h - mu) @ qr) @ qr.T
                delta *= np.linalg.norm(identity, axis=1, keepdims=True) / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
                evaluate(f'random_remove_{seed}', h + delta, {'random_seed': seed})
            for seed in range(3):
                rng = np.random.default_rng(500 + seed); same, cross = [], []
                for j, (s, im) in enumerate(zip(labels, image_ids)):
                    same.append(rng.choice(np.flatnonzero((labels == s) & (image_ids == im) & (np.arange(len(h)) != j))))
                    donor_subject = rng.choice(np.unique(labels[labels != s]))
                    cross.append(rng.choice(np.flatnonzero((labels == donor_subject) & (image_ids == im))))
                same = np.array(same); cross = np.array(cross)
                np.savez(out / f'{prefix}_donors_{seed}.npz', source_cache_indices=np.flatnonzero(te),
                         same_cache_indices=np.flatnonzero(te)[same], cross_cache_indices=np.flatnonzero(te)[cross])
                ds = ((h[same] - h) @ q) @ q.T; dc = ((h[cross] - h) @ q) @ q.T
                matched = dc * np.linalg.norm(ds, axis=1, keepdims=True) / np.maximum(np.linalg.norm(dc, axis=1, keepdims=True), 1e-12)
                for alpha in protocol['alphas']:
                    for tag, delta in [('same', ds), ('cross', dc), ('cross_matched', matched)]:
                        evaluate(f'{tag}_s{seed}_a{alpha}', h + alpha * delta, {'alpha': alpha, 'donor_seed': seed})
            write(out / f'{prefix}_checks.json', {'passed': True, 'rank': rank, 'probe_accuracy': acc,
                  'previous_probe_accuracy': previous, 'C': float(clf.C), 'validation_candidates': candidates_c,
                  'checkpoint': str(ckpath), 'query_trials': len(h), 'query_images': len(np.unique(image_ids)),
                  'candidate_images': len(candidates), 'head_replay': True})
            print('COMPLETE', prefix, flush=True)
    write(out / 'status.json', {'state': 'complete', 'conditions': len(all_results)})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    run(args.out)
