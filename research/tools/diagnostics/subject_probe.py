"""Frozen image-disjoint subject probes; no retrieval model updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from clean_ema_control import CleanModel, CleanAlljoined, seed
from data import DatasetEEG


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def sample(ds):
    subjects, labels = np.unique(ds.subjects.astype(str), return_inverse=True)
    groups = {}
    for i, (im, s) in enumerate(zip(ds.ids, labels)):
        groups.setdefault((int(im), int(s)), []).append(i)
    common = sorted(im for im in np.unique(ds.ids)
                    if all(len(groups.get((int(im), s), [])) >= 4 for s in range(len(subjects))))
    assert len(common) >= 100, len(common)
    rng = np.random.default_rng(20261101)
    images = rng.permutation(common)
    n = len(images)
    split_images = [images[:n//2], images[n//2:3*n//4], images[3*n//4:]]
    ix, splits = [], []
    for part, ims in enumerate(split_images):
        for im in ims:
            for s in range(len(subjects)):
                ix.extend(rng.choice(groups[(int(im), s)], 4, replace=False).tolist())
                splits.extend([part]*4)
    ix = np.asarray(ix)
    assert len(np.unique(ix)) == len(ix)
    return ix, labels[ix], np.asarray(splits), dict(
        subjects=subjects.tolist(), images=[v.tolist() for v in split_images],
        repeats_per_subject_image=4, rows=len(ix), image_overlap=0,
        protocol='Public retrieval test images split 50/25/25% for identity probe only; not session-disjoint')


def fit_probe(x, y, split, permuted=False):
    tr, va, te = [split == i for i in range(3)]
    scaler = StandardScaler().fit(x[tr])
    a, b, c = [scaler.transform(x[m]) for m in [tr, va, te]]
    train_y = y[tr].copy()
    if permuted:
        np.random.default_rng(17).shuffle(train_y)
    candidates = []
    best = None
    for penalty in [.001, .01, .1, 1.]:
        clf = LogisticRegression(C=penalty, solver='lbfgs', max_iter=1000,
                                 tol=1e-5, random_state=33)
        clf.fit(a, train_y)
        score = accuracy_score(y[va], clf.predict(b))
        candidates.append(dict(C=penalty, validation_accuracy=float(score),
                               iterations=clf.n_iter_.tolist()))
        if best is None or score > best[0]:
            best = score, penalty, clf
    pred = best[2].predict(c)
    cm = confusion_matrix(y[te], pred)
    return dict(C=best[1], validation_accuracy=float(best[0]),
                test_accuracy=float(accuracy_score(y[te], pred)),
                balanced_accuracy=float(balanced_accuracy_score(y[te], pred)),
                per_subject_accuracy=(cm.diagonal()/cm.sum(1)).tolist(),
                confusion_matrix=cm.tolist(), candidates=candidates,
                chance=1/len(np.unique(y)), permuted_train_labels=permuted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    seed()
    base = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
    cfg = json.loads((base/'reve_v70_clean_ema_control/config.json').read_text())
    tc = json.loads(Path(cfg['things_config']).read_text())
    ac = json.loads(Path(cfg['alljoined_config']).read_text())
    manifest = json.loads((Path(tc['source_root'])/'data/manifest.json').read_text())
    meta = json.loads(Path(ac['data_manifest']).read_text())
    ds = dict(things=DatasetEEG(tc['source_root'], 'test', aux=False),
              alljoined=CleanAlljoined(meta['test'], np.load(ac['clean_features'], mmap_mode='r'), None, None))
    selections = {}
    for domain, data in ds.items():
        ix, y, split, audit = sample(data)
        selections[domain] = ix, y, split
        save(root/f'{domain}_audit.json', audit)
        np.savez(root/f'{domain}_selection.npz', indices=ix, subject=y, split=split, image=data.ids[ix])
    results = []
    runs = dict(pretrained=None, v70='reve_v70_clean_ema_control',
                v71='reve_v71_clean_ema_control', v72='reve_v72_cross_subject_mix')
    for name, run in runs.items():
        model = CleanModel(manifest['channels'], meta['channels'], False).cuda().eval()
        model.requires_grad_(False)
        checkpoint = None
        if run:
            checkpoint = base/run/'stage1/ema0.9998_epoch_010.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
        for domain, data in ds.items():
            save(root/'status.json', dict(state='running', model=name, domain=domain, phase='extract'))
            ix, y, split = selections[domain]
            pools, outputs = [], []
            positions = model.positions if domain == 'things' else model.alljoined_positions
            model.domain = domain
            with torch.inference_mode():
                for b in DataLoader(Subset(data, ix), batch_size=128, num_workers=2):
                    x = model.preprocess(b['eeg'].cuda())
                    e = model.encoder(x, pos=positions.to(x).unsqueeze(0).expand(len(x), -1, -1),
                                      return_output=True)[-1].mean(1)
                    pools.append(e.cpu().numpy())
                    if run:
                        z = model.head(e)
                        assert torch.isfinite(z).all()
                        if not outputs:
                            torch.testing.assert_close(z, model(b['eeg'].cuda()))
                        outputs.append(z.cpu().numpy())
            features = dict(pooled=np.concatenate(pools))
            if run:
                features['projection_clean'] = np.concatenate(outputs)
            np.savez(root/f'{name}_{domain}_features.npz', **features)
            save(root/'status.json', dict(state='running', model=name, domain=domain, phase='linear_probe'))
            for layer, x in features.items():
                assert np.isfinite(x).all()
                row = dict(model=name, domain=domain, layer=layer, checkpoint=str(checkpoint),
                           result=fit_probe(x, y, split), shuffled_control=fit_probe(x, y, split, True))
                results.append(row)
                save(root/'results.json', results)
                print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    save(root/'status.json', dict(state='complete', comparisons=len(results)))


if __name__ == '__main__':
    main()
