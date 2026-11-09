"""Materialize official single-trial windows and unchanged clean targets."""
import json
from pathlib import Path
import numpy as np
from neuralbench.data import Data
from torch.utils.data import DataLoader
from neuralbench.experiment_config import merge_task_config


def competition_data_config(official):
    # Retain local data locations, image caches and the predefined split.
    # Replace the entire REVE-specific neuro config with the competition task.
    dc=json.loads(json.dumps(official['data']))
    task=merge_task_config('eeg','image',None)['data']
    dc['neuro']=task['neuro']
    dc['start']=task['start'];dc['duration']=task['duration']
    dc.update(num_workers=4, batch_size=256, train_shuffle=False)
    return dc


def main():
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    root=Path(cfg['run_root'])/'data'; root.mkdir(parents=True, exist_ok=True)
    official=json.loads(Path(cfg['official_config']).read_text())
    dc=competition_data_config(official)
    if (root/'complete.json').exists():
        saved=json.loads((root/'manifest.json').read_text())
        assert saved.get('preprocessing_contract')=='competition_eeg_image_v1', 'Legacy EEG cache; use a fresh run directory'
        assert saved['neuro_config']==dc['neuro'], 'Preprocessing config changed; rebuild in a fresh directory'
        print('Competition data already exported', flush=True); return
    assert not any(root.iterdir()), 'Refusing to overwrite partial or legacy data'
    data=Data(**dc)
    loaders=data.prepare()
    frames={s:l.dataset.triggers for s,l in loaders.items()}
    print('TRIGGER COLUMNS',frames['train'].columns.tolist(),flush=True)
    paths=sorted(set(str(p) for f in frames.values() for p in f['filepath']))
    assert all(Path(p).is_file() for p in paths)
    lookup={p:i for i,p in enumerate(paths)}
    clean=np.zeros((len(paths),1536),dtype='float32'); seen=np.zeros(len(paths),bool)
    manifest={'paths':paths,'splits':{},'official_config':cfg['official_config'],
              'preprocessing_contract':'competition_eeg_image_v1',
              'channels':list(data.neuro._channels),'neuro_config':dc['neuro'],
              'start':dc['start'],'duration':dc['duration']}
    for split,loader in loaders.items():
        f=frames[split]; ids=np.array([lookup[str(p)] for p in f['filepath']],dtype='int64')
        subjects=f['subject'].astype(str).to_numpy()
        np.save(root/f'{split}_ids.npy',ids); np.save(root/f'{split}_subjects.npy',subjects.astype(str))
        f.to_csv(root/f'{split}_trials.csv',index=False)
        out=None; offset=0
        for batch in loader:
            x=batch.data['neuro'].numpy().astype('float32')
            y=batch.data['target'].numpy().reshape(len(x),-1)
            assert x.ndim==3 and y.shape[1]==1536 and np.isfinite(x).all() and np.isfinite(y).all()
            assert x.shape[-1]==round(dc['duration']*dc['neuro']['frequency'])==120
            if out is None:
                out=np.lib.format.open_memmap(root/f'{split}_eeg.npy',mode='w+',dtype='float32',shape=(len(ids),*x.shape[1:]))
                manifest['input_shape']=list(x.shape[1:])
            bi=ids[offset:offset+len(x)]
            for j,k in enumerate(bi):
                if seen[k]:
                    assert np.allclose(clean[k],y[j],rtol=1e-5,atol=1e-5),f'Inconsistent target {k}'
                else: clean[k]=y[j];seen[k]=True
            out[offset:offset+len(x)]=x;offset+=len(x)
            if offset%25600==0: print(split,offset,'/',len(ids),flush=True)
        assert offset==len(ids)
        out.flush()
        manifest['splits'][split]={'trials':offset,'images':len(np.unique(ids))}
    assert seen.all()
    train_ids=set(np.load(root/'train_ids.npy').tolist()); val_ids=set(np.load(root/'val_ids.npy').tolist());test_ids=set(np.load(root/'test_ids.npy').tolist())
    assert not (train_ids|val_ids)&test_ids
    manifest['val_images_absent_from_train']=len(val_ids-train_ids)
    np.save(root/'clean.npy',clean)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    (root/'complete.json').write_text(json.dumps({'status':'verified','images':len(paths)}))
    print('EXPORT COMPLETE',manifest['splits'],flush=True)

if __name__=='__main__':main()
