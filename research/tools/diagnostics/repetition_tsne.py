"""Same-subject image EEG averages -> frozen encoder -> joint t-SNE."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from clean_ema_control import CleanModel, CleanAlljoined, seed
from data import DatasetEEG


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2))


def prepare(ds, allowed):
    subjects, sy = np.unique(ds.subjects.astype(str), return_inverse=True)
    groups = {}
    for i, (im,s) in enumerate(zip(ds.ids, sy)):
        groups.setdefault((int(im),int(s)),[]).append(i)
    eligible = [int(im) for im in allowed if all(len(groups.get((int(im),s),[]))>=30 for s in range(len(subjects)))]
    rng=np.random.default_rng(20261101)
    images=rng.permutation(eligible)[:6]
    assert len(images)==6
    arrays,meta,indices=[],[],[]
    for im in images:
        for s in range(len(subjects)):
            ix=rng.choice(groups[(int(im),s)],30,replace=False)
            offset=0
            for k in [1,2,3,4]:
                for rep in range(3):
                    chosen=ix[offset:offset+k]; offset+=k
                    arrays.append(torch.stack([ds[int(j)]['eeg'] for j in chosen]).mean(0))
                    meta.append([int(im),s,k,rep]); indices.append(chosen.tolist())
            assert offset==30
    return torch.stack(arrays),np.asarray(meta),dict(subjects=subjects.tolist(),images=images.tolist(),
           indices=indices,protocol='30 disjoint raw trials per subject/image: 3 independent samples each for k=1,2,3,4')


def distances(x, meta):
    u=normalize(x); d=np.clip(1-u@u.T,0,2)
    im,s,k,_=meta.T
    upper=np.triu(np.ones(d.shape,dtype=bool),1)
    same_s=s[:,None]==s; same_i=im[:,None]==im
    result={}
    for a in [1,2,3,4]:
        for b in range(a,5):
            pair=upper&(((k[:,None]==a)&(k==b))|((k[:,None]==b)&(k==a)))
            result[f'{a}-{b}']={}
            for label,mask in [('same_person_same_image',same_s&same_i),
                               ('same_person_other_image',same_s&~same_i),
                               ('other_person_same_image',~same_s&same_i)]:
                z=d[pair&mask]
                result[f'{a}-{b}'][label]=dict(mean=float(z.mean()),pairs=len(z))
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    root=Path(args.output); seed()
    base=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
    cfg=json.loads((base/'reve_v70_clean_ema_control/config.json').read_text())
    tc=json.loads(Path(cfg['things_config']).read_text()); ac=json.loads(Path(cfg['alljoined_config']).read_text())
    tm=json.loads((Path(tc['source_root'])/'data/manifest.json').read_text()); am=json.loads(Path(ac['data_manifest']).read_text())
    datasets=dict(things=DatasetEEG(tc['source_root'],'test',aux=False),
        alljoined=CleanAlljoined(am['test'],np.load(ac['clean_features'],mmap_mode='r'),None,None))
    prepared={}
    for domain,ds in datasets.items():
        audit=json.loads((base/f'subject_identity_probe_20261101/{domain}_audit.json').read_text())
        x,meta,a=prepare(ds,audit['images'][2]); prepared[domain]=(x,meta,a)
        write(root/f'{domain}_sampling.json',a); np.save(root/f'{domain}_metadata.npy',meta)
    models=dict(pretrained=None,v70='reve_v70_clean_ema_control',v71='reve_v71_clean_ema_control',v72='reve_v72_cross_subject_mix')
    embeddings={}; metrics=[]
    for name,run in models.items():
        model=CleanModel(tm['channels'],am['channels'],False).eval().requires_grad_(False)
        if run:
            state=torch.load(base/run/'stage1/ema0.9998_epoch_010.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True); del state
        for domain,(data,meta,a) in prepared.items():
            write(root/'status.json',dict(state='running',model=name,domain=domain,phase='encoder_cpu'))
            p=model.positions if domain=='things' else model.alljoined_positions
            out=[]
            with torch.inference_mode():
                for batch in data.split(32):
                    x=model.preprocess(batch)
                    e=model.encoder(x,pos=p.to(x)[None].expand(len(x),-1,-1),return_output=True)[-1].mean(1)
                    out.append(e.numpy())
            e=np.concatenate(out); assert np.isfinite(e).all()
            np.save(root/f'{name}_{domain}_pooled.npy',e)
            metrics.append(dict(model=name,domain=domain,distances=distances(e,meta)))
            write(root/'distances.json',metrics)
            write(root/'status.json',dict(state='running',model=name,domain=domain,phase='tsne'))
            q=PCA(n_components=50,random_state=33).fit_transform(normalize(e))
            xy=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',max_iter=1000,
                    random_state=33,n_jobs=4).fit_transform(q)
            np.save(root/f'{name}_{domain}_tsne.npy',xy); embeddings[name,domain]=xy
            print(name,domain,'completed',len(e),flush=True)
        del model
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    markers={1:'o',2:'^',3:'s',4:'X'}
    for domain,(data,meta,a) in prepared.items():
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        colors=plt.get_cmap('tab20')(np.linspace(0,1,len(a['subjects'])))
        for col,name in enumerate(models):
            xy=embeddings[name,domain]
            for row in range(2):
                ax=axes[row,col]; sel=np.ones(len(meta),bool) if row==0 else meta[:,0]==a['images'][0]
                for s in range(len(a['subjects'])):
                    for k in markers:
                        pick=sel&(meta[:,1]==s)&(meta[:,2]==k)
                        ax.scatter(*xy[pick].T,s=12 if row==0 else 28,c=[colors[s]],marker=markers[k],alpha=.7,linewidths=0)
                ax.set_title(name+(' | six images' if row==0 else ' | one image'))
                ax.set_xticks([]); ax.set_yticks([])
        handles=[Line2D([],[],color=colors[s],marker='o',linestyle='',label=str(a['subjects'][s])) for s in range(len(colors))]
        handles += [Line2D([],[],color='black',marker=markers[k],linestyle='',label=f'{k} trial(s)') for k in markers]
        fig.legend(handles=handles,loc='lower center',ncol=8,fontsize=9)
        fig.suptitle(f'{domain}: frozen REVE pooled features; color = subject, marker = averaging count\nL2 -> PCA(50) -> t-SNE; disjoint source trials; panels fitted separately',fontsize=12)
        fig.tight_layout(rect=(0,.13,1,.92))
        fig.savefig(root/f'{domain}_tsne.png',dpi=300); fig.savefig(root/f'{domain}_tsne.pdf'); plt.close(fig)
    write(root/'status.json',dict(state='complete',models=4,domains=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
