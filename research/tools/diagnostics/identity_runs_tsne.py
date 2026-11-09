"""Matched single-trial subject clustering for v80-v84, without training."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from data import DatasetEEG
from local_ot_model import LocalOTModel

BASE = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
OUT = BASE / 'identity_v80_v84_k1_20261105'
RUNS = dict(v80='reve_v80_v64_identity_direct', v81='reve_v81_grouped_identity_dualbatch',
            v82='reve_v82_grouped_identity_weight1', v83='reve_v83_concept_identity_weight1',
            v84='reve_v84_global_identity_weight1')
IMAGES = [30, 73]
MATCHED = False

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2))

def select():
    selected = {}
    for name, folder in RUNS.items():
        root = BASE / folder
        rows = [json.loads(s) for s in (root/'stage1/history.jsonl').read_text().splitlines() if s.strip()]
        options = []
        for r in rows:
            for variant, metrics in r['metrics'].items():
                if MATCHED and (r['epoch'] != 10 or variant != 'ema0.9998'): continue
                prefix = '' if variant == 'raw' else variant + '_'
                p = root/'stage1'/f'{prefix}epoch_{r["epoch"]:03d}.pt'
                if p.exists():
                    options.append((metrics['blur']['subject_aggregated']['top5'], -r['epoch'], variant, p))
        score, neg_epoch, variant, p = max(options, key=lambda t: (t[0], t[1]))
        selected[name] = dict(checkpoint=str(p), epoch=-neg_epoch, variant=variant, selection_score=score,
                              latest_complete_epoch=rows[-1]['epoch'], interim=(name=='v83'))
    return selected

def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(33); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = json.loads((OUT/'selection.json').read_text()) if (OUT/'selection.json').exists() else select()
    save('selection.json', selected)
    reference = {}
    for name, choice in selected.items():
        if all((OUT/f'{name}_image{im}_features.npz').exists() for im in IMAGES): continue
        save('status.json', dict(state='extracting', model=name))
        state = torch.load(choice['checkpoint'], map_location='cpu', weights_only=False)
        model = LocalOTModel(state['manifest']).eval().requires_grad_(False)
        model.load_state_dict(state['model'], strict=True)
        ds = DatasetEEG(state['config']['source_root'], 'test', aux=False)
        del state
        model.cuda()
        subjects, sy = np.unique(ds.subjects.astype(str), return_inverse=True)
        for im in IMAGES:
            ix = np.flatnonzero(ds.ids == im)
            assert len(ix)>30 and len(np.unique(sy[ix]))==10
            if im in reference: assert np.array_equal(reference[im], ix)
            reference[im]=ix
            parts = {k: [] for k in ['pooled', 'blur']}
            with torch.inference_mode():
                for start in range(0, len(ix), 128):
                    x = torch.stack([ds[int(i)]['eeg'] for i in ix[start:start+128]]).cuda()
                    x = model.preprocess(x)
                    pos = model.positions.to(x)[None].expand(len(x), -1, -1)
                    e = model.encoder(x, pos=pos, return_output=True)[-1].mean(1)
                    h = model.blur_head(e)
                    for key, t in [('pooled',e), ('blur',h)]: parts[key].append(t.cpu().numpy())
            features = {key: np.concatenate(v) for key,v in parts.items()}
            assert all(np.isfinite(v).all() for v in features.values())
            np.savez(OUT/f'{name}_image{im}_features.npz', **features, subject=sy[ix], subjects=subjects, trial_indices=ix)
        del model, ds
        torch.cuda.empty_cache()
    records=[]
    for im in IMAGES:
        for layer in ['pooled', 'blur']:
            fig, axes=plt.subplots(len(selected),2,figsize=(12,4.4*len(selected)))
            palette=plt.get_cmap('tab10')(np.arange(10))
            for row,(name,choice) in enumerate(selected.items()):
                save('status.json',dict(state='plotting',image=im,layer=layer,model=name))
                data=np.load(OUT/f'{name}_image{im}_features.npz')
                x=data[layer]; y=data['subject']; k=len(np.unique(y))
                cache=OUT/f'{name}_image{im}_{layer}_tsne.npz'
                if cache.exists():
                    c=np.load(cache); xy=c['xy']; pred=c['cluster']
                else:
                    xy=TSNE(perplexity=30,n_components=2,init='pca',max_iter=2000,random_state=5,n_jobs=4).fit_transform(x)
                    pred=KMeans(n_clusters=k,n_init=20,random_state=5).fit_predict(xy)
                    np.savez(cache,xy=xy,cluster=pred,subject=y)
                counts=np.zeros((k,k),dtype=int);np.add.at(counts,(pred,y),1)
                a,b=linear_sum_assignment(-counts); mapping=np.empty(k,dtype=int); mapping[a]=b
                scores={}
                for space, labels in [('tsne2d',pred), ('original',KMeans(n_clusters=k,n_init=20,random_state=5).fit_predict(x))]:
                    metric=dict(model=name,image=im,layer=layer,space=space,n=len(y),
                                ARI=float(adjusted_rand_score(y,labels)),NMI=float(normalized_mutual_info_score(y,labels)))
                    if space=='original':metric['subject_silhouette']=float(silhouette_score(x,y))
                    records.append(metric); scores[space]=metric
                for col,lab in enumerate([y,mapping[pred]]):
                    ax=axes[row,col]
                    for s in range(k): ax.scatter(*xy[lab==s].T,s=10,alpha=.65,color=palette[s],linewidths=0)
                    info=f'{name} ep{choice["epoch"]} {choice["variant"]}'+(' (INTERIM)' if choice['interim'] else '')
                    title=info+' | '+('Subject' if col==0 else 'KMeans')
                    m=scores['tsne2d' if col else 'original']
                    title+=f'\n{"2D" if col else "Original-space"} ARI={m["ARI"]:.3f}, NMI={m["NMI"]:.3f}'
                    ax.set_title(title,fontsize=11);ax.set_xticks([]);ax.set_yticks([])
            note='All ep10 EMA .9998; v64: no identity loss' if MATCHED else 'v83 still training'
            fig.suptitle(f'THINGS test image {im} | k=1: all {len(y)} single trials\n{layer} ({x.shape[1]}D) -> direct t-SNE -> KMeans (10 clusters)\nSame subject colors across rows; independent embeddings; {note}',fontsize=13)
            handles=[Line2D([],[],marker='o',linestyle='',color=palette[s],label=str(data['subjects'][s]).split('/')[-1]) for s in range(10)]
            fig.legend(handles=handles,loc='lower center',ncol=10,fontsize=9)
            fig.tight_layout(rect=(0,.025,1,.95))
            for ext in ['png','pdf']:fig.savefig(OUT/f'image{im}_{layer}.{ext}',dpi=180)
            plt.close(fig)
            save('metrics.json',records)
    save('protocol.json',dict(images=IMAGES,selection='matched ep10 EMA .9998' if MATCHED else 'maximum available blur aggregate Top5; ties earliest epoch; v83 interim',
        input='all single trials from fixed test images; no augmentation or averaging',
        features='REVE pooled512 and EEG blur prediction1536, FP32 eval mode, raw unnormalized',
        tsne='direct feature input, PCA initialization only, perplexity30, 2000 iterations, seed5, 2D',
        clustering='KMeans10 n_init20 seed5 in original and 2D spaces; labels used only for metrics/color matching',
        interpretation='two preselected illustrative images; not evidence of dataset-wide superiority'))
    save('status.json',dict(state='complete'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--matched',action='store_true');args=ap.parse_args()
    if args.matched:
        MATCHED=True
        OUT=BASE/'identity_v64_v80_v84_matched_ep10_20261105'
        RUNS={'v64':'reve_v64_stage1_ema',**{k:v for k,v in RUNS.items() if k!='v83'}}
    with threadpool_limits(limits=4): main()
