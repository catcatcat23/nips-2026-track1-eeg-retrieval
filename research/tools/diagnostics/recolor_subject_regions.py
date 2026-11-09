"""Permute subject colors across panels by centroid position, never change xy/labels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
SRC=BASE/'direct_tsne_full_aggregate_20261101'
OUT=BASE/'subject_region_colors_20261102'

def main():
    OUT.mkdir(exist_ok=True)
    mappings={}
    for domain in ['things','alljoined']:
        subjects=np.load(BASE/'subject_full_aggregate_20261101'/f'{domain}_v70.npz')['subjects']
        palette=plt.get_cmap('tab10' if len(subjects)==10 else 'tab20')(np.arange(len(subjects)))
        for dim in [2,3]:
            fig=plt.figure(figsize=(14,18))
            references={}
            for row,model in enumerate(['v70','v71','v72']):
                for col,condition in enumerate(['single','mean']):
                    d=np.load(SRC/f'{domain}_{model}_{condition}_{dim}d.npz')
                    xy=d['xy'];labels=d['subject'];ids=np.unique(labels)
                    # Normalize geometry only for matching; displayed coordinates stay unchanged.
                    z=(xy-xy.mean(0))/max(float(np.sqrt(np.mean((xy-xy.mean(0))**2))),1e-12)
                    centers=np.stack([z[labels==s].mean(0) for s in ids])
                    if row==0:
                        references[condition]=centers
                        colors=np.arange(len(ids))
                    else:
                        a,b=linear_sum_assignment(cdist(centers,references[condition],metric='sqeuclidean'))
                        colors=np.empty(len(ids),dtype=int);colors[a]=b
                    mappings[f'{domain}_{dim}d_{model}_{condition}']={str(subjects[s]):int(colors[i]) for i,s in enumerate(ids)}
                    ax=fig.add_subplot(3,2,row*2+col+1,projection='3d' if dim==3 else None)
                    handles=[]
                    for i,s in enumerate(ids):
                        p=xy[labels==s];color=palette[colors[i]]
                        ax.scatter(*p.T,color=color,s=10,alpha=.6,linewidths=0)
                        handles.append(Line2D([],[],color=color,marker='o',linestyle='',markersize=4,label=str(subjects[s]).split('/')[-1]))
                    ax.set_xticks([]);ax.set_yticks([])
                    if dim==3:ax.set_zticks([]);ax.view_init(elev=20,azim=35)
                    ax.set_title(f'{model} | '+('one image, single trials' if condition=='single' else 'all images, averaged features')+f'\nn={len(xy)}',fontsize=12)
                    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.015),ncol=5,fontsize=8,frameon=False,handletextpad=.3,columnspacing=.7)
            fig.suptitle(f'{domain} | direct 512D -> t-SNE {dim}D | true subject groups\nColors matched by centroid location to v70 within each column\nSame color across panels does NOT necessarily mean the same subject',fontsize=13,y=.99)
            fig.tight_layout(rect=(0,.02,1,.94),h_pad=4)
            for ext in ['png','pdf']:fig.savefig(OUT/f'{domain}_{dim}d_subject_regions.{ext}',dpi=300)
            plt.close(fig)
    (OUT/'color_mapping.json').write_text(json.dumps(mappings,indent=2))
    (OUT/'protocol.json').write_text(json.dumps(dict(source=str(SRC),change='subject-level color permutation only; each subject retains one color within each panel',matching='Hungarian minimum squared distance of subject centroids after panel centering/isotropic RMS scaling; reference v70 separately per domain/dimension/condition',unchanged='coordinates, subject membership, KMeans, metrics, and feature data',limitation='Display convenience only; matched colors imply neither shared identity nor correspondence of high-dimensional representations'),indent=2))

if __name__=='__main__':main()
