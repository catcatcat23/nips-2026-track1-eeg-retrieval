"""Compact plots of the already-computed matched-epoch embeddings."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/identity_v64_v80_v84_matched_ep10_20261105')
def main():
    rows=json.loads((ROOT/'metrics.json').read_text())
    models=['v64','v80','v81','v82','v84']
    titles=['v64: no identity loss','v80: global, weight 0.1','v81: same image, weight 0.1','v82: same image, weight 1.0','v84: global, weight 1.0']
    palette=plt.get_cmap('tab10')(np.arange(10))
    for image in [30,73]:
        fig,axes=plt.subplots(2,5,figsize=(20,8))
        for r,layer in enumerate(['pooled','blur']):
            for col,model in enumerate(models):
                d=np.load(ROOT/f'{model}_image{image}_{layer}_tsne.npz')
                xy=d['xy']; y=d['subject'];ax=axes[r,col]
                for s in range(10):ax.scatter(*xy[y==s].T,s=7,alpha=.65,color=palette[s],linewidths=0)
                m=next(m for m in rows if m['model']==model and m['image']==image and m['layer']==layer and m['space']=='original')
                ax.set_title(titles[col]+f'\nOriginal ARI={m["ARI"]:.3f}; silhouette={m["subject_silhouette"]:.3f}',fontsize=10)
                ax.set_xticks([]);ax.set_yticks([])
                if col==0:ax.set_ylabel('REVE pooled (512D)' if r==0 else 'EEG blur prediction (1536D)',fontsize=12)
        handles=[Line2D([],[],marker='o',linestyle='',color=palette[i],label=s) for i,s in enumerate(['1','10','2','3','4','5','6','7','8','9'])]
        fig.legend(handles=handles,loc='lower center',ncol=10,title='True subject identity (same color in every panel)')
        fig.suptitle(f'THINGS test image {image} | all {len(y)} single trials (k=1) | matched epoch 10, EMA .9998\nDirect t-SNE, perplexity 30, 2000 iterations, seed 5; each panel fitted independently',fontsize=14)
        fig.tight_layout(rect=(0,.08,1,.92))
        for ext in ['png','pdf']:fig.savefig(ROOT/f'overview_image{image}.{ext}',dpi=180)
        plt.close(fig)
if __name__=='__main__':main()
