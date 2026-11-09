"""Unsupervised identity clustering on fixed held-out probe images."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def scores(y, c, k):
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (y, c), 1)
    a, b = linear_sum_assignment(-counts)
    return dict(ari=float(adjusted_rand_score(y, c)),
                nmi=float(normalized_mutual_info_score(y, c)),
                matched_accuracy=float(counts[a, b].sum()/len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    source, dest = Path(args.source), Path(args.output)
    dest.mkdir(exist_ok=True)
    results = []
    for domain in ['things', 'alljoined']:
        sel = np.load(source/f'{domain}_selection.npz')
        mask = sel['split'] == 2
        y = sel['subject'][mask]
        k = len(np.unique(y))
        assert np.array_equal(np.unique(y), np.arange(k))
        assert len(np.unique(sel['image'][mask])) == 50
        assert len(set(np.bincount(y))) == 1
        for name in ['pretrained', 'v70', 'v71', 'v72']:
            cache = np.load(source/f'{name}_{domain}_features.npz')
            for layer in cache.files:
                original = np.ascontiguousarray(cache[layer][mask], dtype=np.float32)
                assert np.isfinite(original).all()
                for mode in ['raw', 'l2']:
                    x = original if mode == 'raw' else normalize(original)
                    write(dest/'status.json', dict(state='running', domain=domain,
                          model=name, layer=layer, mode=mode))
                    runs = []
                    for seed in [33, 34, 35]:
                        km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                                    max_iter=300, random_state=seed, algorithm='lloyd')
                        c = km.fit_predict(x)
                        shuffled = np.random.default_rng(seed).permutation(y)
                        result = dict(seed=seed, **scores(y,c,k),
                                      shuffled=scores(shuffled,c,k),
                                      inertia=float(km.inertia_), iterations=int(km.n_iter_),
                                      cluster_sizes=np.bincount(c,minlength=k).tolist())
                        runs.append(result)
                        np.save(dest/f'{name}_{domain}_{layer}_{mode}_seed{seed}_labels.npy',c)
                    row = dict(domain=domain, model=name, layer=layer, mode=mode,
                               rows=len(y), clusters=k, runs=runs,
                               summary={m:dict(mean=float(np.mean([r[m] for r in runs])),
                                               std=float(np.std([r[m] for r in runs],ddof=1)))
                                        for m in ['ari','nmi','matched_accuracy']})
                    results.append(row)
                    write(dest/'results.json',results)
                    print(json.dumps({a:b for a,b in row.items() if a!='runs'}),flush=True)
    lines=['| Dataset | Model | Layer | Scaling | ARI mean ± seed SD | NMI mean ± seed SD | Matched accuracy |',
           '|---|---|---|---|---:|---:|---:|']
    for r in results:
        s=r['summary']
        lines.append(f"| {r['domain']} | {r['model']} | {r['layer']} | {r['mode']} | "
                     f"{s['ari']['mean']:.4f} ± {s['ari']['std']:.4f} | "
                     f"{s['nmi']['mean']:.4f} ± {s['nmi']['std']:.4f} | "
                     f"{100*s['matched_accuracy']['mean']:.2f}% |")
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    write(dest/'status.json',dict(state='complete',comparisons=len(results)))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
