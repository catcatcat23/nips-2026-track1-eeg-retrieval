"""Shared t-SNE for fixed-image trial means; original-space pairwise diagnostics."""
from subject_pca import *
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
OUT=BASE/'subject_mean_tsne_20261101'

def distances(x,labels):
    u=normalize(x.astype(np.float64)); subjects=np.unique(labels)
    # Equal weight per subject and subject pair; within-person self-pairs excluded.
    means=[];within=[]
    for s in subjects:
        a=u[labels==s];n=len(a);assert n>1
        means.append(a.mean(0))
        within.append(float((n*n-(a.sum(0)**2).sum())/(n*(n-1))))
    means=np.stack(means); between=1-means@means.T
    w=float(np.mean(within));b=float(between[np.triu_indices(len(subjects),1)].mean())
    return dict(within_cosine_distance=w,between_cosine_distance=b,between_over_within=b/w,n=len(x),min_per_subject=int(min((labels==s).sum() for s in subjects)))

def main():
    OUT.mkdir(exist_ok=True);results={}
    for domain in ['things','alljoined']:
        meta=np.load(SRC/f'{domain}_metadata.npy')
        sampling=json.loads((SRC/f'{domain}_sampling.json').read_text())
        eligible=meta[:,0]==sampling['fixed_image'];m=meta[eligible]
        subjects=sampling['subjects'];colors=plt.get_cmap('tab20')(np.linspace(0,1,len(subjects)))
        fig,axes=plt.subplots(4,4,figsize=(14,13));results[domain]={}
        for row,name in enumerate(MODELS):
            x=np.load(SRC/f'{domain}_{name}_pooled.npy')[eligible]
            pca=PCA(n_components=50,svd_solver='full').fit(x[m[:,2]==1])
            xy=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',max_iter=1000,random_state=33,n_jobs=4).fit_transform(pca.transform(x))
            np.savez(OUT/f'{domain}_{name}.npz',xy=xy,metadata=m,pca_variance=pca.explained_variance_ratio_)
            lo=xy.min(0);hi=xy.max(0);pad=.05*(hi-lo);results[domain][name]={}
            for col,k in enumerate([1,2,3,4]):
                pick=m[:,2]==k;ax=axes[row,col]
                for s in range(len(subjects)):
                    points=xy[pick&(m[:,1]==s)]
                    ax.scatter(points[:,0],points[:,1],s=9,alpha=.8,color=colors[s],linewidths=0)
                ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
                ax.set_xticks([]);ax.set_yticks([]);ax.spines[['top','right']].set_visible(False)
                ax.set_title(f'{name} | k={k} | n={pick.sum()}',fontsize=10)
                results[domain][name][str(k)]=distances(x[pick],m[pick,1])
            print(domain,name,json.dumps(results[domain][name]),flush=True)
        handles=[Line2D([],[],color=colors[i],marker='o',linestyle='',label=s.split('/')[-1]) for i,s in enumerate(subjects)]
        fig.legend(handles=handles,loc='lower center',ncol=10,fontsize=9)
        fig.suptitle(f'{domain} | fixed image 30 | color = subject\nPCA50 fit on single trials; joint t-SNE across k=1/2/3/4; shared axes within each row',fontsize=12)
        fig.tight_layout(rect=(0,.065,1,.94))
        for ext in ['png','pdf']:fig.savefig(OUT/f'{domain}_mean_comparison.{ext}',dpi=300)
        plt.close(fig)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    (OUT/'protocol.json').write_text(json.dumps(dict(source=str(SRC),image=30,averaging='Raw EEG averaged within subject/image before encoder preprocessing; cached 512D pooled features',sampling='Disjoint chunks within k; shared raw pool across k; all valid trials; decreasing plotted n as k grows',projection='Raw features without L2 -> PCA50 fitted on k1 -> joint TSNE all k per model/domain; perplexity30 seed33 1000iterations',diagnostic='Cosine distances in original512D; equal subject weight for within nonself pairs, equal subject-pair weight for between pairs; descriptive only; no significance test',models='pretrained, v70/v71/v72 ep10 EMA.9998'),indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
