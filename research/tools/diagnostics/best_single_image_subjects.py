"""Select by shared clean retrieval success, never by t-SNE appearance."""
from repetition_tsne import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
OUT=BASE/'best_single_image_subjects_20261102'
RUNS=dict(v70='reve_v70_clean_ema_control',v71='reve_v71_clean_ema_control',v72='reve_v72_cross_subject_mix')

def main():
    seed();OUT.mkdir(exist_ok=True)
    cfg=json.loads((BASE/RUNS['v70']/'config.json').read_text())
    tc=json.loads(Path(cfg['things_config']).read_text());ac=json.loads(Path(cfg['alljoined_config']).read_text())
    tm=json.loads((Path(tc['source_root'])/'data/manifest.json').read_text());am=json.loads(Path(ac['data_manifest']).read_text())
    ds=DatasetEEG(tc['source_root'],'test',aux=False)
    ids=np.unique(ds.ids);gallery=normalize(np.array(ds.clean[ids]))
    from torch.utils.data import DataLoader
    cache=OUT/'v71_per_image_single.npz'
    if cache.exists():
        saved=np.load(cache);hits=saved['hits'];counts=saved['counts']
    else:
        model=CleanModel(tm['channels'],am['channels'],False)
        st=torch.load(BASE/RUNS['v71']/'stage1/ema0.9998_epoch_010.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(st['model'],strict=True);del st
        model.eval().requires_grad_(False).cuda()
        g=torch.from_numpy(gallery).cuda();image_ids=torch.from_numpy(ids).cuda()
        hits=np.zeros(len(ids),np.int64);counts=np.zeros(len(ids),np.int64);seen=0
        with torch.inference_mode():
            for batch in DataLoader(ds,batch_size=256,shuffle=False,num_workers=2,pin_memory=True):
                logits=model(batch['eeg'].cuda(non_blocking=True))@g.T
                pred=image_ids[logits.argmax(1)].cpu().numpy();true=batch['id'].numpy()
                idx=np.searchsorted(ids,true)
                np.add.at(hits,idx,(pred==true).astype(np.int64));np.add.at(counts,idx,1)
                seen+=len(true)
                if seen%25600==0:
                    write(OUT/'status.json',dict(state='scoring_all_test_trials',model='v71',rows=seen,total=len(ds)))
                    print('scoring',seen,len(ds),flush=True)
        assert counts.sum()==len(ds) and (counts>0).all()
        np.savez(cache,image_ids=ids,hits=hits,counts=counts)
        del model;torch.cuda.empty_cache()
    scores=hits/counts
    order=np.lexsort((ids,-scores));chosen=int(ids[order[0]])
    write(OUT/'selection.json',dict(image=chosen,criterion='maximum v71 per-image single-trial clean Top1 across all valid trials and subjects; smallest image ID tie-break; no visualization-based selection',gallery_size=len(ids),image_ids=ids.tolist(),v71_single_top1=scores.tolist(),selected_top1=float(scores[order[0]]),limitation='Selected using v71 test performance; illustrative case, not unbiased model comparison'))
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
        fig.suptitle(f'THINGS | selected successful test image {chosen} | all {len(ix)} single trials\nDirect raw512 -> t-SNE {dim}D; region-matched colors: same color can mean different subjects\nImage selected by highest v71 single-trial clean Top1; not by visual separation',fontsize=11)
        fig.tight_layout(rect=(0,.02,1,.93),h_pad=3)
        for ext in ['png','pdf']:fig.savefig(OUT/f'things_{dim}d.{ext}',dpi=300)
        plt.close(fig)
    write(OUT/'color_mapping.json',mappings);write(OUT/'status.json',dict(state='complete'))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
