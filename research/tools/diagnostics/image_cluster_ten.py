"""Image identity clustering of all subject/image averaged EEG features."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
OUT=BASE/'image_cluster_ten_20261102'

def main():
    OUT.mkdir(exist_ok=True);results=[]
    for domain in ['things','alljoined']:
        fig,axes=plt.subplots(3,2,figsize=(14,17))
        palette=plt.get_cmap('tab10')(np.arange(10))
        selected=None
        for row,name in enumerate(['v70','v71','v72']):
            a=np.load(BASE/'subject_full_aggregate_20261101'/f'{domain}_{name}.npz')
            if selected is None:
                selected=np.sort(np.random.default_rng(5).choice(np.unique(a['pairs'][:,1]),10,replace=False))
            keep=np.isin(a['pairs'][:,1],selected)
            images,y=np.unique(a['pairs'][keep,1],return_inverse=True)
            raw=a['mean'][keep];k=10
            xy=TSNE(n_components=2,perplexity=30,init='pca',max_iter=2000,random_state=5,verbose=0).fit_transform(raw)
            for space,x in [('raw512',raw),('l2_512',normalize(raw)),('tsne2d',xy)]:
                pred=KMeans(n_clusters=k,n_init=10,random_state=5,max_iter=300).fit_predict(x)
                table=np.zeros((k,k),int);np.add.at(table,(pred,y),1)
                r,c=linear_sum_assignment(-table);mapping=np.empty(k,int);mapping[r]=c
                record=dict(domain=domain,model=name,space=space,n=len(y),k=k,ARI=float(adjusted_rand_score(y,pred)),NMI=float(normalized_mutual_info_score(y,pred)),matched_fraction=float((mapping[pred]==y).mean()))
                results.append(record)
                if space=='tsne2d':display=mapping[pred];score=record;clusters=pred
            np.savez(OUT/f'{domain}_{name}.npz',xy=xy,image_ids=images,labels=y,subject=a['pairs'][keep,0],cluster=clusters,cluster_display=display,palette=palette)
            for col,labels in enumerate([y,display]):
                ax=axes[row,col]
                ax.scatter(xy[:,0],xy[:,1],c=palette[labels],s=32,alpha=.8,linewidths=0)
                ax.set_title(f'{name} | '+('true image ID' if col==0 else 'KMeans K=10')+f' | n={len(y)}'+('' if col==0 else f'\nARI={score["ARI"]:.3f}, NMI={score["NMI"]:.3f}'))
                ax.set_xticks([]);ax.set_yticks([])
            if row==0:
                axes[row,0].legend(handles=[plt.Line2D([],[],marker='o',linestyle='',color=palette[i],label=str(int(img))) for i,img in enumerate(images)],title='Image ID',ncol=5,fontsize=8)
            print(domain,name,score,flush=True)
            (OUT/'results.json').write_text(json.dumps(results,indent=2))
        fig.suptitle(f'{domain} | 10 test images selected randomly (seed=5) | mean of single-trial REVE features\nDirect raw512 -> t-SNE 2D; no PCA preprocessing | colors denote the same image across models\nImage IDs: {selected.tolist()}',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94))
        for ext in ['png','pdf']:fig.savefig(OUT/f'{domain}_image_clusters.{ext}',dpi=300)
        plt.close(fig)
    (OUT/'protocol.json').write_text(json.dumps(dict(features='same cached raw512 averaged within subject/image, all valid test trials; v70/v71/v72 epoch10 EMA.9998',labels='10 randomly selected exact image IDs, not semantic categories',projection='refit direct512 tSNE2D seed5 perplexity30 iter2000, no PCA preprocessing; independently fitted models',clustering='KMeans K10 n_init10 seed5 max_iter300 on raw512, L2 normalized512 and tSNE2D; labels only for scoring/color matching',limitations='single clustering seed, test-selected training checkpoints; no significance claims; this is encoder-feature image clustering, not candidate-image retrieval accuracy'),indent=2))
    (OUT/'status.json').write_text('{"state":"complete"}')

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
