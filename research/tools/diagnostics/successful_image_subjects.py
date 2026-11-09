"""Select by shared clean retrieval success, never by t-SNE appearance."""
from repetition_tsne import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
OUT=BASE/'successful_image_subjects_20261102'
RUNS=dict(v70='reve_v70_clean_ema_control',v71='reve_v71_clean_ema_control',v72='reve_v72_cross_subject_mix')

def main():
    seed();OUT.mkdir(exist_ok=True)
    cfg=json.loads((BASE/RUNS['v70']/'config.json').read_text())
    tc=json.loads(Path(cfg['things_config']).read_text());ac=json.loads(Path(cfg['alljoined_config']).read_text())
    tm=json.loads((Path(tc['source_root'])/'data/manifest.json').read_text());am=json.loads(Path(ac['data_manifest']).read_text())
    ds=DatasetEEG(tc['source_root'],'test',aux=False)
    ids=np.unique(ds.ids);gallery=normalize(np.array(ds.clean[ids]))
    scores=[];per_model={}
    for name,run in RUNS.items():
        a=np.load(BASE/'subject_full_aggregate_20261101'/f'things_{name}.npz')
        st=torch.load(BASE/run/'stage1/ema0.9998_epoch_010.pt',map_location='cpu',weights_only=False)['model']
        w=st['head.weight'].numpy();b=st['head.bias'].numpy()
        logits=(a['mean']@w.T+b)@gallery.T
        correct=ids[logits.argmax(1)]==a['pairs'][:,1]
        score=np.array([correct[a['pairs'][:,1]==i].mean() for i in ids]);scores.append(score)
        per_model[name]=score.tolist();del st
    scores=np.stack(scores)
    # Maximize worst model accuracy, then mean accuracy; break ties by smallest image ID.
    order=np.lexsort((ids,-scores.mean(0),-scores.min(0)));chosen=int(ids[order[0]])
    write(OUT/'selection.json',dict(image=chosen,criterion='maximize minimum across models of per-image subject-aggregated clean Top1; then mean; smallest image ID tie-break; no tSNE selection',gallery_size=len(ids),image_ids=ids.tolist(),per_model_aggregate_top1=per_model,selected_scores={n:float(scores[j,order[0]]) for j,n in enumerate(RUNS)},limitation='Deliberately selected successful example; not representative evidence of overall identity separation'))
    ix=np.flatnonzero(ds.ids==chosen);subjects,sy=np.unique(ds.subjects.astype(str),return_inverse=True);labels=sy[ix]
    x=torch.stack([ds[int(i)]['eeg'] for i in ix]);features={};metrics={}
    for name,run in RUNS.items():
        model=CleanModel(tm['channels'],am['channels'],False)
        st=torch.load(BASE/run/'stage1/ema0.9998_epoch_010.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(st['model'],strict=True);del st
        model.eval().requires_grad_(False).cuda();out=[]
        with torch.inference_mode():
            for batch in x.split(128):
                z=model.preprocess(batch.cuda());p=model.positions.to(z)
                out.append(model.encoder(z,pos=p[None].expand(len(z),-1,-1),return_output=True)[-1].mean(1).cpu())
            e=torch.cat(out);pred=model.head(e.cuda()).cpu().numpy();rank=np.argsort(-(pred@gallery.T),axis=1)
        features[name]=e.numpy();metrics[name]=dict(single_top1=float((ids[rank[:,0]]==chosen).mean()),single_top5=float((ids[rank[:,:5]]==chosen).any(1).mean()))
        np.savez(OUT/f'{name}_features.npz',features=e.numpy(),subject=labels,trial_indices=ix,image=chosen)
        del model;torch.cuda.empty_cache()
        print(name,chosen,metrics[name],flush=True)
    write(OUT/'retrieval.json',metrics)
    mappings={}
    for dim in [2,3]:
        fig=plt.figure(figsize=(13,15));reference=None;palette=plt.get_cmap('tab10')(np.arange(len(subjects)))
        for row,(name,e) in enumerate(features.items()):
            xy=TSNE(perplexity=30,n_components=dim,init='pca',max_iter=2000,random_state=5,verbose=1,n_jobs=4).fit_transform(e)
            z=(xy-xy.mean(0))/np.sqrt(np.mean((xy-xy.mean(0))**2))
            cent=np.stack([z[labels==s].mean(0) for s in range(len(subjects))])
            if reference is None:reference=cent;color_ids=np.arange(len(subjects))
            else:
                a,b=linear_sum_assignment(cdist(cent,reference));color_ids=np.empty(len(subjects),int);color_ids[a]=b
            cluster=KMeans(n_clusters=len(subjects),n_init=20,random_state=5).fit_predict(xy)
            table=np.zeros((len(subjects),len(subjects)),int);np.add.at(table,(cluster,labels),1)
            a,b=linear_sum_assignment(-table);mapping=np.empty(len(subjects),int);mapping[a]=b
            ari=adjusted_rand_score(labels,cluster);nmi=normalized_mutual_info_score(labels,cluster)
            mappings[f'{name}_{dim}d']={str(subjects[s]):int(color_ids[s]) for s in range(len(subjects))}
            np.savez(OUT/f'{name}_{dim}d.npz',xy=xy,subject=labels,cluster=cluster,color_ids=color_ids,ARI=ari,NMI=nmi)
            for col,lab in enumerate([labels,mapping[cluster]]):
                ax=fig.add_subplot(3,2,row*2+col+1,projection='3d' if dim==3 else None)
                for s in range(len(subjects)):
                    points=xy[lab==s];ax.scatter(*points.T,color=palette[color_ids[s]],s=10,alpha=.6,linewidths=0)
                ax.set_xticks([]);ax.set_yticks([])
                if dim==3:ax.set_zticks([]);ax.view_init(20,35)
                ax.set_title(f'{name} | '+('subject' if col==0 else f'KMeans | ARI={ari:.3f}, NMI={nmi:.3f}'))
                if col==0:
                    handles=[Line2D([],[],marker='o',linestyle='',color=palette[color_ids[s]],label=str(subjects[s]).split('/')[-1]) for s in range(len(subjects))]
                    ax.legend(handles=handles,ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.01),frameon=False)
        fig.suptitle(f'THINGS | selected successful test image {chosen} | all {len(ix)} single trials\nDirect raw512 -> t-SNE {dim}D; region-matched colors: same color can mean different subjects\nImage selected by aggregate clean retrieval, not by visual separation',fontsize=11)
        fig.tight_layout(rect=(0,.02,1,.93),h_pad=3)
        for ext in ['png','pdf']:fig.savefig(OUT/f'things_{dim}d.{ext}',dpi=300)
        plt.close(fig)
    write(OUT/'color_mapping.json',mappings);write(OUT/'status.json',dict(state='complete'))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
