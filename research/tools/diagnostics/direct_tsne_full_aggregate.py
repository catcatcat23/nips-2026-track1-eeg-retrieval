"""Direct raw512 t-SNE then KMeans: single-image trials versus all-image aggregate diagnostics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
import sklearn
BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
SRC=BASE/'subject_full_aggregate_20261101'
OUT=BASE/'direct_tsne_full_aggregate_20261101'
MODELS=['v70','v71','v72']
def main():
    OUT.mkdir(exist_ok=True);results=[]
    for domain in ['things','alljoined']:
        for dim in [2,3]:
            fig=plt.figure(figsize=(17,12))
            for row,name in enumerate(MODELS):
                data=np.load(SRC/f'{domain}_{name}.npz');k=len(data['subjects'])
                palette=plt.get_cmap('tab20')(np.linspace(0,1,k))
                for ci,condition in enumerate(['single','mean']):
                    (OUT/'status.json').write_text(json.dumps(dict(state='running',domain=domain,dimension=dim,model=name,condition=condition)))
                    x=data[condition];y=data['single_subject'] if condition=='single' else data['pairs'][:,0]
                    cached=OUT/f'{domain}_{name}_{condition}_{dim}d.npz'
                    if cached.exists():
                        saved=np.load(cached);xy=saved['xy'];pred=saved['cluster']
                    else:
                        xy=TSNE(perplexity=30,n_components=dim,init='pca',max_iter=2000,random_state=5,verbose=1,learning_rate='auto',metric='euclidean',method='barnes_hut',angle=.5,n_jobs=4).fit_transform(x)
                        pred=KMeans(n_clusters=k,n_init=20,random_state=5,max_iter=300).fit_predict(xy)
                    counts=np.zeros((k,k),dtype=int);np.add.at(counts,(pred,y),1)
                    a,b=linear_sum_assignment(-counts);mapping=np.empty(k,dtype=int);mapping[a]=b
                    mapped=mapping[pred]
                    record=dict(domain=domain,dimensions=dim,model=name,condition=condition,n=len(y),k=k,ARI=float(adjusted_rand_score(y,pred)),NMI=float(normalized_mutual_info_score(y,pred)),matched_accuracy=float((mapped==y).mean()))
                    results.append(record);print(json.dumps(record),flush=True)
                    np.savez(OUT/f'{domain}_{name}_{condition}_{dim}d.npz',xy=xy,subject=y,cluster=pred,cluster_display=mapped)
                    for offset,lab in enumerate([y,mapped]):
                        ax=fig.add_subplot(3,4,row*4+ci*2+offset+1,projection='3d' if dim==3 else None)
                        for s in range(k):
                            p=xy[lab==s]
                            ax.scatter(*p.T,color=palette[s],s=7,alpha=.6,linewidths=0)
                        ax.set_xticks([]);ax.set_yticks([])
                        if dim==3:ax.set_zticks([]);ax.view_init(elev=20,azim=35)
                        title=f'{name} | {condition} | '+('subject' if offset==0 else 'KMeans')
                        ax.set_title(title+'\n'+(f'n={len(y)}' if offset==0 else f'ARI={record["ARI"]:.3f}, NMI={record["NMI"]:.3f}'),fontsize=10)
                    (OUT/'results.json').write_text(json.dumps(results,indent=2))
            fig.suptitle(f'{domain} | single image 30 / all-image feature means | direct raw512 -> t-SNE {dim}D -> KMeans\nEach condition fitted independently; cluster colors matched to identities AFTER clustering',fontsize=13)
            fig.tight_layout(rect=(0,0,1,.95))
            for ext in ['png','pdf']:fig.savefig(OUT/f'{domain}_{dim}d.{ext}',dpi=300)
            plt.close(fig)
    (OUT/'protocol.json').write_text(json.dumps(dict(sklearn_version=sklearn.__version__,source=str(SRC),features='raw512; no PCA preprocessing or L2; PCA initialization only',tsne=dict(perplexity=30,dimensions=[2,3],max_iter=2000,seed=5,learning_rate='auto',method='barnes_hut',metric='euclidean',angle=.5),kmeans=dict(k='known subject count 10 or20',n_init=20,seed=5,max_iter=300),labels='not used for fit; Hungarian matching for color and descriptive matched accuracy only',limitations='single trials: image30; aggregate: all test images; all valid trials per subject/image; one seed; t-SNE changes geometry; this is low-dimensional clustering, not high-dimensional classifier accuracy; each condition independently fitted'),indent=2))
    (OUT/'status.json').write_text(json.dumps(dict(state='complete')))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
