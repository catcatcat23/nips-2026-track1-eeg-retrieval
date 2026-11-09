"""Fixed images and checkpoint: identity clustering across EEG feature stages."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from data import DatasetEEG
from local_ot_model import LocalOTModel
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
OUT=BASE/'blur_layer_subjects_20261102'

def write(path, value):
    path.write_text(json.dumps(value, indent=2))

def main():
    torch.manual_seed(33);np.random.seed(33);torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    path=BASE/'reve_v64_stage1_ema/stage1/ema0.9998_epoch_010.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    cfg=state['config'];model=LocalOTModel(state['manifest']).eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True);del state
    ds=DatasetEEG(cfg['source_root'],'test',aux=False)
    subjects,sy=np.unique(ds.subjects.astype(str),return_inverse=True)
    metrics=[]
    for im in [6,30,73]:
        cache=OUT/f'image_{im}_features.npz'
        if cache.exists():
            a=np.load(cache);features={k:a[k] for k in ['pooled','blur','clean']};labels=a['subject']
        else:
            ix=np.flatnonzero(ds.ids==im);labels=sy[ix];parts={k:[] for k in ['pooled','blur','clean']}
            with torch.inference_mode():
                for start in range(0,len(ix),32):
                    batch=torch.stack([ds[int(i)]['eeg'] for i in ix[start:start+32]])
                    x=model.preprocess(batch);p=model.positions.to(x)
                    e=model.encoder(x,pos=p[None].expand(len(x),-1,-1),return_output=True)[-1].mean(1)
                    h=model.blur_head(e);z=model.recovery(h)
                    for k,t in [('pooled',e),('blur',h),('clean',z)]:parts[k].append(t.numpy())
                    write(OUT/'status.json',dict(state='extracting_cpu',image=im,rows=min(start+32,len(ix)),total=len(ix)))
            features={k:np.concatenate(v) for k,v in parts.items()}
            np.savez(cache,**features,subject=labels,trial_indices=ix)
        fig,axes=plt.subplots(1,3,figsize=(17,6));palette=plt.get_cmap('tab10')(np.arange(len(subjects)))
        for col,(layer,x) in enumerate(features.items()):
            for norm in ['raw','l2']:
                inp=x if norm=='raw' else normalize(x)
                for rs in [5,33,34]:
                    pred=KMeans(n_clusters=len(subjects),n_init=10,random_state=rs).fit_predict(inp)
                    metrics.append(dict(image=im,layer=layer,space='original',normalization=norm,seed=rs,ARI=float(adjusted_rand_score(labels,pred)),NMI=float(normalized_mutual_info_score(labels,pred))))
            xy=TSNE(perplexity=30,n_components=2,init='pca',max_iter=2000,random_state=5,verbose=1,n_jobs=4).fit_transform(x)
            pred=KMeans(n_clusters=len(subjects),n_init=20,random_state=5).fit_predict(xy)
            ari=adjusted_rand_score(labels,pred);nmi=normalized_mutual_info_score(labels,pred)
            metrics.append(dict(image=im,layer=layer,space='tsne2d',normalization='raw',seed=5,ARI=float(ari),NMI=float(nmi)))
            np.savez(OUT/f'image_{im}_{layer}_xy.npz',xy=xy,subject=labels,cluster=pred)
            for s in range(len(subjects)):
                z=xy[labels==s];axes[col].scatter(*z.T,color=palette[s],alpha=.6,s=10,linewidths=0)
            axes[col].set_title(f'{layer} ({x.shape[1]}D) | ARI={ari:.3f}, NMI={nmi:.3f}')
            axes[col].set_xticks([]);axes[col].set_yticks([])
        handles=[Line2D([],[],marker='o',linestyle='',color=palette[s],label=str(subjects[s]).split('/')[-1]) for s in range(len(subjects))]
        fig.legend(handles=handles,loc='lower center',ncol=10)
        fig.suptitle(f'v64 epoch10 EMA .9998 | test image {im} | all {len(labels)} single trials\nDirect t-SNE (no PCA preprocessing); true subject colors; independently fitted panels')
        fig.tight_layout(rect=(0,.08,1,.90))
        for ext in ['png','pdf']:fig.savefig(OUT/f'image_{im}.{ext}',dpi=300)
        plt.close(fig);write(OUT/'metrics.json',metrics)
    write(OUT/'protocol.json',dict(checkpoint=str(path),images=[6,30,73],selection='All three previously discussed images, fixed before observing blur results',features='same forward pass pooled512, EEG blur prediction1536, clean1536; no image-side features',scope='layer diagnostic within v64, not v71 augmentation ablation',clustering='original raw and L2 with seeds5/33/34, plus raw direct tSNE2D seed5; K=10',precision='FP32 CPU, eval mode, no EEG augmentation'))
    write(OUT/'status.json',dict(state='complete'))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
