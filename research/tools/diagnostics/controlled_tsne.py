"""Predeclared subject/image views, with disjoint groups within each k."""
from repetition_tsne import *


def sample_views(ds, previous):
    subjects, sy = np.unique(ds.subjects.astype(str),return_inverse=True)
    images=previous['images']
    groups={}
    for i,(im,s) in enumerate(zip(ds.ids,sy)):
        if int(im) in images and (int(im)==images[0] or s==0):
            groups.setdefault((int(im),int(s)),[]).append(i)
    arrays, meta, source=[],[],[]
    rng=np.random.default_rng(20261102)
    for (im,s), ix in sorted(groups.items()):
        ix=rng.permutation(ix)
        for k in [1,2,3,4]:
            for j in range(len(ix)//k):
                chosen=ix[j*k:(j+1)*k]
                arrays.append(torch.stack([ds[int(i)]['eeg'] for i in chosen]).mean(0))
                meta.append([im,s,k,j]);source.append(chosen.tolist())
    return torch.stack(arrays),np.array(meta),dict(subjects=subjects.tolist(),images=images,
        fixed_image=images[0],fixed_subject=subjects[0],indices=source,
        overlap='No reuse within each averaging count; counts share the same raw pool. Descriptive paired views, not independent data.')


def project(x):
    x=normalize(x)
    q=PCA(n_components=min(50,len(x)-1,x.shape[1]),random_state=33).fit_transform(x)
    return TSNE(perplexity=min(30,(len(x)-1)/3),init='pca',learning_rate='auto',
                max_iter=1000,random_state=33,n_jobs=4).fit_transform(q)


def plots(root,domain,meta,a,features):
    names=list(features)
    for kind in ['identity','image','repetition']:
        fig,axes=plt.subplots(4,4,figsize=(14,13))
        values=np.arange(len(a['subjects'])) if kind=='identity' else np.array(a['images'])
        cmap=plt.get_cmap('tab20' if kind=='identity' else 'tab10')
        colors=cmap(np.linspace(0,1,len(values))) if kind=='identity' else cmap(np.arange(len(values)))
        for row,name in enumerate(names):
            e=features[name]
            eligible=(meta[:,0]==a['fixed_image']) if kind=='identity' else (meta[:,1]==0)
            joint=None
            if kind=='repetition':
                joint=project(e[eligible]);np.save(root/f'{domain}_{name}_joint_repetitions.npy',joint)
            for col,k in enumerate([1,2,3,4]):
                ax=axes[row,col]; pick=eligible&(meta[:,2]==k)
                xy=joint[meta[eligible,2]==k] if joint is not None else project(e[pick])
                lab=meta[pick,1] if kind=='identity' else meta[pick,0]
                for i,v in enumerate(values):
                    ax.scatter(*xy[lab==v].T,s=9,c=[colors[i]],alpha=.8,linewidths=0)
                if joint is not None:
                    lo=joint.min(0);hi=joint.max(0); pad=.05*(hi-lo)
                    ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
                ax.set_title(f'{name} | k={k} | n={len(xy)}',fontsize=10)
                ax.set_xticks([]);ax.set_yticks([])
                np.savez(root/f'{domain}_{kind}_{name}_k{k}.npz',xy=xy,labels=lab,metadata=meta[pick])
        labels=[s.split('/')[-1] for s in a['subjects']] if kind=='identity' else [f'image {i}' for i in values]
        handles=[Line2D([],[],color=colors[i],marker='o',linestyle='',label=label) for i,label in enumerate(labels)]
        fig.legend(handles=handles,loc='lower center',ncol=min(10,len(labels)),fontsize=9)
        title={'identity':f'One fixed image ({a["fixed_image"]}); color = subject',
               'image':f'One fixed subject ({a["fixed_subject"].split("/")[-1]}); color = image',
               'repetition':'One fixed subject; same t-SNE coordinates across k within each row; color = image'}[kind]
        fig.suptitle(f'{domain}: {title}\nEncoder pooled features; L2 -> PCA -> t-SNE; no label-guided fitting',fontsize=12)
        fig.tight_layout(rect=(0,.065,1,.94))
        for ext in ['png','pdf']:fig.savefig(root/f'{domain}_{kind}.{ext}',dpi=300)
        plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.output);seed();base=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
    cfg=json.loads((base/'reve_v70_clean_ema_control/config.json').read_text())
    tc=json.loads(Path(cfg['things_config']).read_text());ac=json.loads(Path(cfg['alljoined_config']).read_text())
    tm=json.loads((Path(tc['source_root'])/'data/manifest.json').read_text());am=json.loads(Path(ac['data_manifest']).read_text())
    ds=dict(things=DatasetEEG(tc['source_root'],'test',aux=False),alljoined=CleanAlljoined(am['test'],np.load(ac['clean_features'],mmap_mode='r'),None,None))
    models=dict(pretrained=None,v70='reve_v70_clean_ema_control',v71='reve_v71_clean_ema_control',v72='reve_v72_cross_subject_mix')
    for domain,data in ds.items():
        old=json.loads((base/f'repetition_tsne_20261101/{domain}_sampling.json').read_text())
        inputs,meta,a=sample_views(data,old);write(root/f'{domain}_sampling.json',a)
        np.save(root/f'{domain}_metadata.npy',meta);features={}
        for name,run in models.items():
            write(root/'status.json',dict(state='running',domain=domain,model=name,phase='encoder_cpu',inputs=len(inputs)))
            model=CleanModel(tm['channels'],am['channels'],False).eval().requires_grad_(False)
            if run:
                state=torch.load(base/run/'stage1/ema0.9998_epoch_010.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(state['model'],strict=True);del state
            p=model.positions if domain=='things' else model.alljoined_positions
            out=[]
            with torch.inference_mode():
                for b in inputs.split(32):
                    x=model.preprocess(b)
                    out.append(model.encoder(x,pos=p.to(x)[None].expand(len(x),-1,-1),return_output=True)[-1].mean(1).numpy())
            features[name]=np.concatenate(out);np.save(root/f'{domain}_{name}_pooled.npy',features[name]);del model
            print(domain,name,'extracted',len(inputs),flush=True)
        write(root/'status.json',dict(state='running',domain=domain,phase='plotting'))
        plots(root,domain,meta,a,features)
    write(root/'status.json',dict(state='complete'))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
