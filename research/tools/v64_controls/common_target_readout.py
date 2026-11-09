"""Matched cached-REVE probes; no test-driven model selection."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/identity_reference_loso_pilot')

def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(fold, arm):
    root = ROOT / f'fold_{fold:02d}_{arm}'
    cfg = json.loads((root / 'config.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    cache = torch.load(root / 'features.pt', map_location='cpu', weights_only=False)
    data = Path(cfg['source_root']) / 'data'
    ids = np.load(data / 'test_ids.npy')
    subjects = np.load(data / 'test_subjects.npy').astype(str)
    ids = ids[subjects == cfg['heldout_subject']]
    assert cache['etest'].shape == (len(ids), 512)
    assert cache['etrain'].shape == (1000, 512)
    assert np.array_equal(cache['images'], protocol['images'])
    assert cache['records'] == [tuple(r) for r in protocol['records']]
    assert not np.intersect1d(cache['images'], ids).size
    assert not np.intersect1d(cache['images'], json.loads(Path(cfg['validation_ids_path']).read_text())).size
    for split_index, split in enumerate(['train', 'val']):
        ii = np.load(data / f'{split}_ids.npy')
        ss = np.load(data / f'{split}_subjects.npy').astype(str)
        for k, (d, row) in enumerate(cache['records']):
            if d == split_index:
                assert ii[row] == cache['images'][k] and ss[row] == cfg['heldout_subject']
    blur_path = Path(cfg['source_root']) / 'features/blur.npy'
    target = np.load(blur_path, mmap_mode='r')[cache['images']].copy()
    candidates = np.unique(ids)
    gallery = []
    for k in candidates:
        with np.load(Path(cfg['test_blur_shards']) / f'{k:05d}.npz') as f:
            gallery.append(f['blur'].copy())
    gallery = np.stack(gallery)
    assert target.shape == (1000, 1536) and gallery.shape == (200, 1536)
    assert len(np.unique(gallery, axis=0)) == 200
    for x in [cache['etrain'], cache['etest'], torch.from_numpy(target), torch.from_numpy(gallery)]:
        assert torch.isfinite(x).all()
    return cache, F.normalize(torch.from_numpy(target), dim=-1), F.normalize(torch.from_numpy(gallery), dim=-1), torch.from_numpy(np.searchsorted(candidates, ids)), dict(checkpoint=protocol['checkpoint'], feature_cache=str(root/'features.pt'), feature_sha256=digest(root/'features.pt'), candidate_ids=candidates.tolist(), test_trials=len(ids), image_pool=cache['images'].tolist(), records=protocol['records'], heldout=cfg['heldout_subject'])

def head(kind):
    layers = [nn.LayerNorm(512, elementwise_affine=False)]
    layers += [nn.Linear(512, 1536)] if kind == 'linear' else [nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1536)]
    return nn.Sequential(*layers)

@torch.no_grad()
def evaluate(model, x, labels, gallery):
    model.eval()
    pred = torch.cat([model(chunk) for chunk in x.split(512)])
    sums = torch.zeros(200, 1536)
    sums.index_add_(0, labels, pred)
    counts = torch.bincount(labels, minlength=200)
    means = sums / counts[:, None]
    def hits(z, y):
        top = (z @ gallery.T).topk(5, dim=1).indices
        return dict(top1=float((top[:, 0] == y).float().mean()), top5=float((top == y[:, None]).any(1).float().mean()), count=len(y))
    return dict(single_trial=hits(pred, labels), subject_aggregated=hits(means, torch.arange(200)), repeats_min=int(counts.min()), repeats_max=int(counts.max()))

def self_test():
    model = nn.Identity()
    g = torch.eye(200, 1536)
    y = torch.arange(200).repeat_interleave(2)
    scores = evaluate(model, g[y], y, g)
    assert scores['single_trial']['top1'] == scores['subject_aggregated']['top5'] == 1
    assert evaluate(model, g[(y+1)%200], y, g)['single_trial']['top1'] == 0

def main(a):
    torch.set_num_threads(4)
    self_test()
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    write(out/'status.json', dict(state='preflight', pid=__import__('os').getpid()))
    settings = dict(task='common frozen blur target readout; no image adapter', folds=[1,2], arms=['baseline','identity'], heads=['linear','mlp256'], budgets=[50,200,1000], seeds=[33,34,35,36,37], steps=200, report_steps=[0,50,200], batch=50, lr=.001, weight_decay=.01, optimizer='AdamW', loss='one-way image-normalized cross entropy; temperature=1', eeg_normalization='per-sample LayerNorm512, no learned affine', device='cpu', selection='fixed final step200; no test selection', aggregation='mean raw predictions within heldout subject and image, all available repeats', sampling='nested subsets of existing shared1000-image cache, one fixed trial per image; seed varies subset/order/init, not pool/trial acquisition; budget1000 has identical support set across seeds', scope='diagnostic probe, not original native blur metric or zero-shot system')
    write(out/'config.json', settings)
    pairs = {}
    manifest = {}
    for fold in [1,2]:
        pair = {arm:load(fold,arm) for arm in settings['arms']}
        b,i = pair.values()
        assert np.array_equal(b[0]['images'],i[0]['images']) and b[0]['records']==i[0]['records']
        assert torch.equal(b[1],i[1]) and torch.equal(b[2],i[2]) and torch.equal(b[3],i[3])
        pairs[fold]=pair
        manifest[str(fold)]={arm:v[-1] for arm,v in pair.items()}
    write(out/'manifest.json',manifest)
    print('PREFLIGHT PASS: matched images/trials/targets/gallery; disjoint calibration and test; finite features; metric self-tests',flush=True)
    if a.preflight:
        write(out/'status.json',dict(state='preflight_passed'))
        return
    start=time.time();done=0
    for fold,pair in pairs.items():
        for kind in settings['heads']:
            for budget in settings['budgets']:
                for seed in settings['seeds']:
                    subset=np.random.default_rng(seed).permutation(1000)[:budget]
                    for arm,(cache,targets,gallery,labels,_) in pair.items():
                        torch.manual_seed(seed)
                        model=head(kind)
                        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                        gen=torch.Generator().manual_seed(seed+10000)
                        x=cache['etrain'][subset].float(); y=targets[subset].float(); xt=cache['etest'].float()
                        key=f's{fold}_{arm}_{kind}_n{budget}_seed{seed}'
                        write(out/'status.json',dict(state='running',current=key,completed=done,total=120))
                        scores={'0':evaluate(model,xt,labels,gallery)};losses=[]
                        for step in range(1,201):
                            model.train(); ix=torch.randperm(budget,generator=gen)[:50]
                            loss=F.cross_entropy(model(x[ix])@y[ix].T,torch.arange(len(ix)))
                            opt.zero_grad();loss.backward()
                            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
                            if step in [50,200]:scores[str(step)]=evaluate(model,xt,labels,gallery)
                        write(out/f'{key}.json',dict(scores=scores,losses=losses,subset_indices=subset.tolist(),image_ids=cache['images'][subset].tolist(),parameter_count=sum(p.numel() for p in model.parameters())))
                        torch.save(model.state_dict(),out/f'{key}.pt')
                        done+=1
                        print(key, scores['200'], 'completed',done,'elapsed_s',round(time.time()-start),flush=True)
    write(out/'status.json',dict(state='complete',completed=done,elapsed_seconds=time.time()-start))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    try:main(a)
    except Exception as exc:
        if a.out.exists():write(a.out/'failure.json',dict(error=repr(exc)))
        raise
