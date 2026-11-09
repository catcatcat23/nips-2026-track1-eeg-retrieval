from subject_pca import *
from sklearn.manifold import TSNE
OUT=BASE/'subject_pca50_tsne_20261101'

def main():
    OUT.mkdir(exist_ok=True)
    results={}
    for domain in ['things','alljoined']:
        meta=np.load(SRC/f'{domain}_metadata.npy')
        sampling=json.loads((SRC/f'{domain}_sampling.json').read_text())
        pick=(meta[:,0]==sampling['fixed_image'])&(meta[:,2]==1)
        subjects=sampling['subjects']; labels=meta[pick,1]
        colors=plt.get_cmap('tab20')(np.linspace(0,1,len(subjects)))
        fig,axes=plt.subplots(1,4,figsize=(18,5))
        results[domain]={}
        for ax,name in zip(axes,MODELS):
            x=np.load(SRC/f'{domain}_{name}_pooled.npy')[pick]
            pca=PCA(n_components=50,svd_solver='full')
            q=pca.fit_transform(x)
            xy=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',max_iter=1000,random_state=33,n_jobs=4).fit_transform(q)
            for s in range(len(subjects)):
                points=xy[labels==s]
                ax.scatter(points[:,0],points[:,1],s=12,alpha=.8,color=colors[s],linewidths=0)
            ax.set_title(f'{name} | n={len(x)}',fontsize=12)
            ax.set_xticks([]);ax.set_yticks([])
            ax.spines[['top','right']].set_visible(False)
            np.savez(OUT/f'{domain}_{name}.npz',xy=xy,metadata=meta[pick],pca_variance=pca.explained_variance_ratio_)
            results[domain][name]={'n':len(x),'pca50_variance':float(pca.explained_variance_ratio_.sum())}
            print(domain,name,results[domain][name],flush=True)
        handles=[Line2D([],[],color=colors[i],marker='o',linestyle='',label=s.split('/')[-1]) for i,s in enumerate(subjects)]
        fig.legend(handles=handles,loc='lower center',ncol=10,fontsize=9)
        fig.suptitle(f'{domain} | fixed image {sampling["fixed_image"]} | all subjects | color = subject\nSingle-trial raw pooled features: 512-D -> PCA 50-D -> t-SNE 2-D',fontsize=13)
        fig.tight_layout(rect=(0,.14,1,.88))
        for ext in ['png','pdf']:fig.savefig(OUT/f'{domain}_single_trials.{ext}',dpi=300)
        plt.close(fig)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    (OUT/'protocol.json').write_text(json.dumps(dict(source=str(SRC),image_id=30,k=1,normalization='PCA centering only; no L2, same as PCA-only control',perplexity=30,seed=33,iterations=1000,init='pca',learning_rate='auto',labels='color only; no supervised fitting'),indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
