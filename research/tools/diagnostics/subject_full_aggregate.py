"""One-image single trials versus all-image means of individual encoder outputs."""
from repetition_tsne import *
from torch.utils.data import DataLoader

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
OUT=BASE/'subject_full_aggregate_20261101'
RUNS=dict(v70='reve_v70_clean_ema_control',v71='reve_v71_clean_ema_control',v72='reve_v72_cross_subject_mix')

def main():
    OUT.mkdir(exist_ok=True); seed()
    cfg=json.loads((BASE/RUNS['v70']/'config.json').read_text())
    tc=json.loads(Path(cfg['things_config']).read_text());ac=json.loads(Path(cfg['alljoined_config']).read_text())
    tm=json.loads((Path(tc['source_root'])/'data/manifest.json').read_text());am=json.loads(Path(ac['data_manifest']).read_text())
    datasets=dict(things=DatasetEEG(tc['source_root'],'test',aux=False),alljoined=CleanAlljoined(am['test'],np.load(ac['clean_features'],mmap_mode='r'),None,None))
    for name,run in RUNS.items():
        model=None
        for domain,ds in datasets.items():
            path=OUT/f'{domain}_{name}.npz'
            if path.exists(): continue
            if model is None:
                model=CleanModel(tm['channels'],am['channels'],False)
                state=torch.load(BASE/run/'stage1/ema0.9998_epoch_010.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(state['model'],strict=True);del state
                model.eval().requires_grad_(False).cuda()
            subjects,sy=np.unique(ds.subjects.astype(str),return_inverse=True)
            pairs,group=np.unique(np.stack([sy,ds.ids],axis=1),axis=0,return_inverse=True)
            sums=np.zeros((len(pairs),512),dtype=np.float64);counts=np.zeros(len(pairs),dtype=np.int64)
            singles=[];labels=[];offset=0
            p=model.positions if domain=='things' else model.alljoined_positions
            loader=DataLoader(ds,batch_size=256,shuffle=False,num_workers=2,pin_memory=True)
            with torch.inference_mode():
                for batch in loader:
                    x=model.preprocess(batch['eeg'].cuda(non_blocking=True))
                    e=model.encoder(x,pos=p.to(x)[None].expand(len(x),-1,-1),return_output=True)[-1].mean(1).cpu().numpy()
                    assert np.isfinite(e).all()
                    end=offset+len(e);g=group[offset:end]
                    np.add.at(sums,g,e);np.add.at(counts,g,1)
                    mask=np.asarray(ds.ids[offset:end])==30
                    if mask.any():singles.append(e[mask]);labels.append(sy[offset:end][mask])
                    offset=end
                    if offset%25600==0:
                        write(OUT/'status.json',dict(state='extracting',model=name,domain=domain,rows=offset,total=len(ds)))
                        print(name,domain,offset,len(ds),flush=True)
            assert offset==len(ds) and (counts>0).all()
            np.savez(path,mean=(sums/counts[:,None]).astype(np.float32),pairs=pairs,counts=counts,single=np.concatenate(singles),single_subject=np.concatenate(labels),subjects=subjects)
        if model is not None:del model;torch.cuda.empty_cache()
    for domain in datasets:
        fig,axes=plt.subplots(3,2,figsize=(13,16))
        for row,name in enumerate(RUNS):
            a=np.load(OUT/f'{domain}_{name}.npz');colors=plt.get_cmap('tab20')(np.linspace(0,1,len(a['subjects'])))
            for col,(x,y) in enumerate([(a['single'],a['single_subject']),(a['mean'],a['pairs'][:,0])]):
                q=PCA(n_components=50,svd_solver='full').fit_transform(x)
                xy=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',max_iter=2000,random_state=5,n_jobs=4).fit_transform(q)
                np.savez(OUT/f'{domain}_{name}_{col}_xy.npz',xy=xy,subject=y)
                for s in range(len(colors)):
                    z=xy[y==s];axes[row,col].scatter(*z.T,c=[colors[s]],s=9,alpha=.7,linewidths=0)
                axes[row,col].set_title(f'{name} | '+('image 30, single trials' if col==0 else 'all test images, mean of trial features')+f'\nn={len(x)}')
                axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        handles=[Line2D([],[],marker='o',linestyle='',color=colors[s],label=str(a['subjects'][s]).split('/')[-1]) for s in range(len(colors))]
        fig.legend(handles=handles,loc='lower center',ncol=10,fontsize=9)
        fig.suptitle(f'{domain} | REVE pooled 512D | epoch 10 EMA .9998\nRaw features -> PCA50 -> t-SNE; panels fitted separately; color = true subject',fontsize=13)
        fig.tight_layout(rect=(0,.05,1,.95))
        for ext in ['png','pdf']:fig.savefig(OUT/f'{domain}_comparison.{ext}',dpi=300)
        plt.close(fig)
    write(OUT/'protocol.json',dict(models=RUNS,checkpoint='stage1/ema0.9998_epoch_010.pt',split='test',left='image30 all valid individual trials',right='all test images; mean individual raw512 encoder features within subject/image; no L2; no EEG averaging',projection='separate PCA50 then TSNE perplexity30 seed5 2000iterations per panel',limitations='Different image sets between columns; no direct attribution of visual differences to averaging; descriptive single seed. Counts stored in NPZ.',precision='FP32, no autocast, TF32 disabled'))
    write(OUT/'status.json',dict(state='complete'))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
