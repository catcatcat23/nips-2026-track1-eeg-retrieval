"""Fixed-image PCA: fit raw single-trial pooled features; project means unchanged."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
SRC=BASE/'controlled_tsne_20261101'
OUT=BASE/'subject_pca_20261101'
MODELS=['pretrained','v70','v71','v72']

def main():
    OUT.mkdir(exist_ok=True)
    results={}
    for domain in ['things','alljoined']:
        meta=np.load(SRC/f'{domain}_metadata.npy')
        sampling=json.loads((SRC/f'{domain}_sampling.json').read_text())
        fixed=meta[:,0]==sampling['fixed_image']
        single=fixed&(meta[:,2]==1)
        subjects=sampling['subjects']; colors=plt.get_cmap('tab20')(np.linspace(0,1,len(subjects)))
        fig,axes=plt.subplots(1,4,figsize=(18,4.8))
        comp,grid=plt.subplots(4,4,figsize=(14,13))
        results[domain]={}
        for row,name in enumerate(MODELS):
            x=np.load(SRC/f'{domain}_{name}_pooled.npy')
            pca=PCA(n_components=2,svd_solver='full').fit(x[single])
            xy=pca.transform(x[fixed]); m=meta[fixed]
            ev=pca.explained_variance_ratio_
            results[domain][name]={'single_trials':int(single.sum()),'explained_variance_ratio':ev.tolist(),'image_id':sampling['fixed_image']}
            np.savez(OUT/f'{domain}_{name}.npz',xy=xy,metadata=m,components=pca.components_,mean=pca.mean_,explained_variance_ratio=ev)
            lo=xy.min(0);hi=xy.max(0);pad=(hi-lo)*.05
            for k in range(1,5):
                ax=grid[row,k-1]; pick=m[:,2]==k
                targets=[ax,axes[row]] if k==1 else [ax]
                for target in targets:
                    for s in range(len(subjects)):
                        points=xy[pick&(m[:,1]==s)]
                        target.scatter(points[:,0],points[:,1],s=10,alpha=.65,color=colors[s],linewidths=0)
                    target.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);target.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
                    target.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)',fontsize=9)
                    target.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)',fontsize=9)
                    target.tick_params(labelsize=8)
                    target.spines[['top','right']].set_visible(False)
                ax.set_title(f'{name} | k={k} | n={pick.sum()}',fontsize=10)
            axes[row].set_title(f'{name} | n={single.sum()}',fontsize=11)
        handles=[Line2D([],[],color=colors[i],marker='o',linestyle='',label=s.split('/')[-1]) for i,s in enumerate(subjects)]
        for f,label in [(fig,'single_trials'),(comp,'averaging_comparison')]:
            f.legend(handles=handles,loc='lower center',ncol=10,fontsize=9)
            f.suptitle(f'{domain} | fixed image {sampling["fixed_image"]} | color = subject\nRaw 512-D pooled features; PCA fitted on single trials only; no labels used',fontsize=12)
            f.tight_layout(rect=(0,.10 if f is fig else .065,1,.88 if f is fig else .94))
            for ext in ['png','pdf']:f.savefig(OUT/f'{domain}_{label}.{ext}',dpi=300)
            plt.close(f)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    (OUT/'protocol.json').write_text(json.dumps({'source':str(SRC),'features':'raw encoder pooled 512-D; centered by PCA, no L2 or standardization','fit':'all subjects, fixed image 30, single trials only, separately per model/domain','transform':'same PCA applied to k=1/2/3/4 inputs; shared axes within model','sampling':'all valid single trials in existing cache; disjoint chunks within k; raw pool shared across k','models':'pretrained and epoch10 EMA .9998 v70/v71/v72'},indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
