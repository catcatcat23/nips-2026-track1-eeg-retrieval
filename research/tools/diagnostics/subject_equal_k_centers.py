"""Equal-count visualization of cached EEG means, with linear-space centers."""
from subject_pca import *

OUT = BASE / 'subject_equal_k_centers_20261101'

def main():
    OUT.mkdir(exist_ok=True)
    record = {}
    for domain in ['things', 'alljoined']:
        meta = np.load(SRC / f'{domain}_metadata.npy')
        sampling = json.loads((SRC / f'{domain}_sampling.json').read_text())
        fixed = meta[:, 0] == sampling['fixed_image']
        m = meta[fixed]
        subjects = np.unique(m[:, 1])
        colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(subjects)))
        rng = np.random.default_rng(20261104)
        count = min(int(((m[:, 1] == s) & (m[:, 2] == k)).sum())
                    for s in subjects for k in range(1, 5))
        picks = {(int(s), k): rng.choice(np.flatnonzero((m[:, 1] == s) & (m[:, 2] == k)), count, replace=False)
                 for s in subjects for k in range(1, 5)}
        record[domain] = {'per_subject_per_panel': count, 'total_per_panel': count * len(subjects),
                          'selected_rows': {f'{s}_{k}': idx.tolist() for (s,k),idx in picks.items()}}
        for method in ['tsne', 'pca']:
            fig, axes = plt.subplots(4, 4, figsize=(15, 14))
            for row, name in enumerate(MODELS):
                x = np.load(SRC / f'{domain}_{name}_pooled.npy')[fixed]
                if method == 'tsne':
                    xy = np.load(BASE / 'subject_mean_tsne_20261101' / f'{domain}_{name}.npz')['xy']
                    centers = None
                else:
                    pca = PCA(n_components=2, svd_solver='full').fit(x[m[:, 2] == 1])
                    xy = pca.transform(x)
                    centers = pca.transform(np.stack([x[(m[:, 1] == s) & (m[:, 2] == 1)].mean(0) for s in subjects]))
                lo, hi = xy.min(0), xy.max(0)
                pad = .05 * (hi - lo)
                for col, k in enumerate(range(1, 5)):
                    ax = axes[row, col]
                    for si, s in enumerate(subjects):
                        a = xy[picks[(int(s), k)]]
                        ax.scatter(a[:, 0], a[:, 1], c=[colors[si]], s=18, alpha=.7, linewidths=0)
                        if centers is not None:
                            ax.scatter(*centers[si], c=[colors[si]], s=135, marker='*', edgecolors='black', linewidths=.6)
                    ax.set(xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]),
                           title=f'{name} | mean {k} trial(s) | n={count * len(subjects)}')
                    ax.set_xticks([]); ax.set_yticks([])
            handles = [Line2D([], [], marker='o', linestyle='', color=colors[i], label=sampling['subjects'][s].split('/')[-1]) for i,s in enumerate(subjects)]
            fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=9)
            detail = 'Joint PCA50 + t-SNE coordinates reused; no identity labels used in fitting' if method == 'tsne' else 'PCA fit on all single trials; stars = mean of ALL single-trial features (not mean EEG)'
            fig.suptitle(f'{domain} | image {sampling["fixed_image"]} | equal count: {count} points/person/panel\n{detail}\nEEG is averaged BEFORE encoding for k=2/3/4; same axes within each row', fontsize=12)
            fig.tight_layout(rect=(0,.06,1,.93))
            for ext in ['png','pdf']:
                fig.savefig(OUT / f'{domain}_{method}.{ext}', dpi=300)
            plt.close(fig)
        print(domain, count, flush=True)
    record['notes'] = 'Fixed image; descriptive visualization only. Sampling independent of features and identical across models. Disjoint raw chunks within each k, shared source trials across k. Equal plot counts obtained by subsampling, not extra trials. PCA stars average encoder pooled features, not final retrieval output; not evidence of retrieval accuracy.'
    (OUT / 'protocol.json').write_text(json.dumps(record, indent=2))

if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
