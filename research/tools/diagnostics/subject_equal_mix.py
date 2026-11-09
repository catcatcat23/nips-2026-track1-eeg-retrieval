"""Equal anchor count, fixed 50/25/25 input mixture; no training or channel masking."""
from repetition_tsne import *
from subject_mean_tsne import distances
BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
SRC=BASE/'controlled_tsne_20261101'
OUT=BASE/'subject_equal_mix_20261101'
MODELS=dict(pretrained=None,v70='reve_v70_clean_ema_control',v71='reve_v71_clean_ema_control',v72='reve_v72_cross_subject_mix')

def main():
    OUT.mkdir(exist_ok=True);seed()
    cfg=json.loads((BASE/'reve_v70_clean_ema_control/config.json').read_text())
    tc=json.loads(Path(cfg['things_config']).read_text());ac=json.loads(Path(cfg['alljoined_config']).read_text())
    tm=json.loads((Path(tc['source_root'])/'data/manifest.json').read_text());am=json.loads(Path(ac['data_manifest']).read_text())
    datasets=dict(things=DatasetEEG(tc['source_root'],'test',aux=False),alljoined=CleanAlljoined(am['test'],np.load(ac['clean_features'],mmap_mode='r'),None,None))
    results={}
    for domain,ds in datasets.items():
        meta=np.load(SRC/f'{domain}_metadata.npy');audit=json.loads((SRC/f'{domain}_sampling.json').read_text())
        rows=np.flatnonzero((meta[:,0]==30)&(meta[:,2]==1));labels=meta[rows,1]
        anchors=np.array([audit['indices'][i][0] for i in rows]);rng=np.random.default_rng(20261103)
        ks=np.ones(len(rows),dtype=int);chosen=[None]*len(rows);inputs=[];avgrows=[]
        for s in np.unique(labels):
            loc=np.flatnonzero(labels==s);n=len(loc)
            n1=int(round(n*.5));n2=int(round(n*.25))
            ks[loc]=rng.permutation([1]*n1+[2]*n2+[4]*(n-n1-n2))
            for j in loc:
                extra=rng.choice(anchors[loc][anchors[loc]!=anchors[j]],ks[j]-1,replace=False)
                ix=[int(anchors[j])]+extra.tolist();chosen[j]=ix
                if ks[j]>1:
                    avgrows.append(j);inputs.append(torch.stack([ds[int(i)]['eeg'] for i in ix]).mean(0))
        avgrows=np.array(avgrows);inputs=torch.stack(inputs)
        write(OUT/f'{domain}_sampling.json',dict(subjects=audit['subjects'],anchors=anchors.tolist(),source_indices=chosen,k=ks.tolist(),labels=labels.tolist(),counts={str(k):int((ks==k).sum()) for k in [1,2,4]}))
        results[domain]={};colors=plt.get_cmap('tab20')(np.linspace(0,1,len(audit['subjects'])))
        fig,axes=plt.subplots(4,2,figsize=(11,16))
        for row,(name,run) in enumerate(MODELS.items()):
            write(OUT/'status.json',dict(state='running',domain=domain,model=name,phase='encoder_cpu',new_inputs=len(inputs)))
            baseline=np.load(SRC/f'{domain}_{name}_pooled.npy')[rows];mixed=baseline.copy()
            model=CleanModel(tm['channels'],am['channels'],False).eval().requires_grad_(False)
            if run:
                state=torch.load(BASE/run/'stage1/ema0.9998_epoch_010.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(state['model'],strict=True);del state
            p=model.positions if domain=='things' else model.alljoined_positions
            out=[]
            with torch.inference_mode():
                for batch in inputs.split(32):
                    x=model.preprocess(batch)
                    out.append(model.encoder(x,pos=p.to(x)[None].expand(len(x),-1,-1),return_output=True)[-1].mean(1).numpy())
            mixed[avgrows]=np.concatenate(out);del model
            assert np.isfinite(mixed).all()
            np.savez(OUT/f'{domain}_{name}_features.npz',baseline=baseline,mixed=mixed,labels=labels,k=ks)
            # Identical singleton features appear only once in joint fitting, then share coordinates.
            joint=np.concatenate([baseline,mixed[avgrows]])
            q=PCA(n_components=50,svd_solver='full').fit(baseline).transform(joint)
            xy=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',max_iter=1000,random_state=33,n_jobs=4).fit_transform(q)
            base_xy=xy[:len(rows)];mix_xy=base_xy.copy();mix_xy[avgrows]=xy[len(rows):]
            np.savez(OUT/f'{domain}_{name}_xy.npz',baseline=base_xy,mixed=mix_xy,labels=labels,k=ks)
            lo=xy.min(0);hi=xy.max(0);pad=.05*(hi-lo)
            for col,(points,title) in enumerate([(base_xy,'100% single'),(mix_xy,'50% single + 25% mean2 + 25% mean4')]):
                ax=axes[row,col]
                for s in np.unique(labels):
                    pts=points[labels==s];ax.scatter(*pts.T,s=10,alpha=.8,color=colors[s],linewidths=0)
                ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_xticks([]);ax.set_yticks([])
                ax.set_title(f'{name} | {title}\nn={len(rows)}',fontsize=10)
            results[domain][name]=dict(baseline=distances(baseline,labels),mixed=distances(mixed,labels))
            write(OUT/'results.json',results);print(domain,name,results[domain][name],flush=True)
        handles=[Line2D([],[],color=colors[i],marker='o',linestyle='',label=s.split('/')[-1]) for i,s in enumerate(audit['subjects'])]
        fig.legend(handles=handles,loc='lower center',ncol=10,fontsize=8)
        fig.suptitle(f'{domain} | fixed image 30 | color = subject | equal anchor count\nShared PCA50 + joint t-SNE within each row; no channel mask',fontsize=12)
        fig.tight_layout(rect=(0,.055,1,.95))
        for ext in ['png','pdf']:fig.savefig(OUT/f'{domain}_comparison.{ext}',dpi=300)
        plt.close(fig)
    write(OUT/'protocol.json',dict(image=30,mixture='50/25/25 per subject, nearest integer counts; shuffled assignment seed20261103',sampling='one input per original anchor; mean includes anchor and k-1 distinct same-person/image trials; sources may repeat between inputs as in augmentation',features='same cached single trials, freshly encoded EEG averages; frozen eval; raw512 pooled',projection='PCA50 fit baseline; TSNE joint baseline plus newly averaged samples, shared singleton points deduplicated; seed33 perplexity30 1000iter',limitations='one image, one sampling realization; no mask, no training, not independent observations, no significance claim'))
    write(OUT/'status.json',dict(state='complete'))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
