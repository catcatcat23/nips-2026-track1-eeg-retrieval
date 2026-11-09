"""Compact plots of the already-computed matched-epoch embeddings."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/identity_alljoined_matched_ep8_20261106')
def main():
    rows=json.loads((ROOT/'metrics.json').read_text())
    models=['v67','v85','v86','v87']
    titles=['v67: no identity loss','v85: global identity','v86: same-image identity','v87: same-concept identity']
    palette=plt.get_cmap('tab20')(np.arange(20))
    for image in [30,73]:
        fig,axes=plt.subplots(2,4,figsize=(18,9))
        for r,layer in enumerate(['pooled','blur']):
            for col,model in enumerate(models):
                d=np.load(ROOT/f'{model}_image{image}_{layer}_tsne.npz')
                xy=d['xy']; y=d['subject'];ax=axes[r,col]
                for s in range(20):ax.scatter(*xy[y==s].T,s=7,alpha=.65,color=palette[s],linewidths=0)
                m=next(m for m in rows if m['model']==model and m['image']==image and m['layer']==layer and m['space']=='original')
                ax.set_title(titles[col]+f'\nOriginal ARI={m["ARI"]:.3f}; silhouette={m["subject_silhouette"]:.3f}',fontsize=10)
                ax.set_xticks([]);ax.set_yticks([])
                if col==0:ax.set_ylabel('REVE pooled (512D)' if r==0 else 'EEG blur prediction (1536D)',fontsize=12)
        names=np.load(ROOT/f'{models[0]}_image{image}_features.npz')['subjects']
        handles=[Line2D([],[],marker='o',linestyle='',color=palette[i],label=str(s).split("/")[-1]) for i,s in enumerate(names)]
        fig.legend(handles=handles,loc='lower center',ncol=10,title='True subject identity (same color in every panel)')
        fig.suptitle(f'Alljoined test image {image} | all {len(y)} single trials (k=1) | matched epoch 8, EMA .9998\nDirect t-SNE, perplexity 30, 2000 iterations, seed 5; each panel fitted independently',fontsize=14)
        fig.tight_layout(rect=(0,.08,1,.92))
        for ext in ['png','pdf']:fig.savefig(ROOT/f'overview_image{image}.{ext}',dpi=300)
        plt.close(fig)
if __name__=='__main__':main()
