"""Stage native-channel Alljoined EEG with the exact official EEG extractor."""
import argparse,hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from scipy.fft import dctn

ROOT=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v18_alljoined')
BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v13_reference')
DATA=Path('/mnt/DATA-4/anteng/eeg-retrieval/data')

def dump(path,obj):path.write_text(json.dumps(obj,indent=2))

def signature(path):
    with Image.open(path) as im:
        im=im.convert('RGB');shape=im.size
        exact=hashlib.sha256(str(shape).encode()+im.tobytes()).hexdigest()
        gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
        z=dctn(gray,norm='ortho')[:8,:8].reshape(-1)[1:]
        return exact,(z>np.median(z)).astype(np.uint8)

def inventory():
    for name in ['Xu2024Alljoined','Xu2025Alljoined']:
        assert (DATA/(name+'.unpacked.json')).exists(),name
    manifest=json.loads((BASE/'data/manifest.json').read_text());paths=manifest['paths'][:]
    sigs=list(map(signature,paths))
    exact={s[0]:i for i,s in enumerate(sigs)}
    testids=np.unique(np.load(BASE/'data/test_ids.npy'));testhash=np.stack([sigs[i][1] for i in testids]);testexact={sigs[i][0] for i in testids}
    source_map={};excluded=[];jobs=[]
    import pandas as pd
    for name,nm in [('Xu2024Alljoined','nm000133'),('Xu2025Alljoined','nm000134')]:
        folder=DATA/name/'download'/nm
        stimuli=folder/'stimuli'
        for events in sorted(folder.glob('sub-*/ses-*/eeg/*_events.tsv')):
            frame=pd.read_csv(events,sep='\t')
            valid=frame['stim_file'].notna() & ~frame['onset'].duplicated(keep=False)
            if name=='Xu2025Alljoined':valid &= frame['trial_type'].eq('stim_train')
            f=frame.loc[valid].copy();keep=[]
            for row in f.itertuples():
                path=str((stimuli/row.stim_file).resolve())
                if path not in source_map:
                    digest,ph=signature(path)
                    distance=int(np.count_nonzero(testhash!=ph,axis=1).min())
                    if digest in testexact or distance<=8:
                        source_map[path]=-1;excluded.append({'path':path,'test_phash_distance':distance})
                    else:
                        if digest not in exact:exact[digest]=len(paths);paths.append(path)
                        source_map[path]=exact[digest]
                if source_map[path]>=0:keep.append([float(row.onset),source_map[path]])
            if keep:
                suffix='.bdf' if name=='Xu2024Alljoined' else '.edf'
                raw=Path(str(events).replace('_events.tsv','_eeg'+suffix));assert raw.exists()
                jobs.append({'dataset':name,'raw':str(raw),'subject':name+'/'+raw.parts[-4], 'events':keep})
    # No active test image IDs can appear in external training metadata.
    assert not set(testids)&{p[1] for j in jobs for p in j['events']}
    dump(ROOT/'inventory.json',{'paths':paths,'jobs':jobs,'base_images':len(manifest['paths']),'excluded_images':excluded,'test_ids':testids.tolist(),'identity':'exact decoded RGB hash for reuse; conservative 63bit pHash distance<=8 excluded against every current test image'})
    print('INVENTORY',len(jobs),'recordings',len(paths),'images',len(excluded),'excluded',flush=True)


def process_job(item):
    from pydantic import PrivateAttr
    from neuralset.extractors.neuro import EegExtractor
    from neuralset.events.etypes import Eeg
    import mne
    class PreparedExtractor(EegExtractor):
        _prepared:object=PrivateAttr(default=None)
        def _get_data(self,events):
            for _ in events:yield self._prepared
    index,job=item;dest=ROOT/'eeg'/f'{index:05d}'
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    raw=mne.io.read_raw(job['raw'],preload=False,verbose='ERROR')
    if 'Afz' in raw.ch_names:raw.rename_channels({'Afz':'AFz'})
    montage=mne.channels.make_standard_montage('standard_1005')
    raw.set_channel_types({n:('eeg' if n in montage.ch_names else 'misc') for n in raw.ch_names},verbose='ERROR')
    raw.set_montage(montage,on_missing='ignore')
    event=Eeg(filepath=job['raw'],timeline=job['raw'],subject=job['subject'],start=0,frequency=float(raw.info['sfreq']),duration=raw.n_times/raw.info['sfreq'])
    cfg=json.loads((BASE/'data/manifest.json').read_text())['neuro_config'].copy();cfg.pop('name');cfg.pop('infra')
    ex=PreparedExtractor(**cfg,mne_cpus=1)
    ex._prepared=ex._preprocess_raw(raw,event)
    channels=list(ex._prepared.ch_names);ex._update_channels(channels)
    # Reuse official window padding, baseline, rounding and clamp implementation.
    arr=np.lib.format.open_memmap(dest.with_suffix('.npy'),mode='w+',dtype='float32',shape=(len(job['events']),len(channels),120))
    for i,(onset,_) in enumerate(job['events']):
        x=ex._get_timed_array(event,start=onset-.2,duration=1.).data
        assert x.shape==(len(channels),120) and np.isfinite(x).all()
        arr[i]=x
    arr.flush()
    result={**job,'channels':channels,'shape':list(arr.shape),'array':str(dest.with_suffix('.npy'))}
    dump(dest.with_suffix('.json'),result)
    return result


def eeg():
    (ROOT/'eeg').mkdir(exist_ok=True)
    inv=json.loads((ROOT/'inventory.json').read_text())
    with ProcessPoolExecutor(max_workers=6) as pool:
        results=[]
        for i,result in enumerate(pool.map(process_job,enumerate(inv['jobs']))):
            results.append(result)
            if i%50==0:print('EEG',i+1,'/',len(inv['jobs']),flush=True)
    for dataset in ['Xu2024Alljoined','Xu2025Alljoined']:
        names=[r['channels'] for r in results if r['dataset']==dataset]
        assert names and all(n==names[0] for n in names),'Channel mismatch within dataset'
    dump(ROOT/'eeg_complete.json',{'recordings':results,'trials':sum(r['shape'][0] for r in results)})


def features():
    import torch
    from torch.nn import functional as F
    from transformers import AutoModel,AutoImageProcessor
    from torchvision import transforms
    from neuralset.extractors.image import _fix_pixel_values
    from upstream_blur import load_blurred,LEVELS
    rank=int(os.environ.get('FEATURE_RANK','0'));world=int(os.environ.get('FEATURE_WORLD','1'))
    inv=json.loads((ROOT/'inventory.json').read_text());out=ROOT/'features';out.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    hf=Path(os.environ['HF_HOME'])/'hub/models--facebook--dinov2-giant/snapshots/611a9d42f2335e0f921f1e313ad3c1b7178d206d'
    model=AutoModel.from_pretrained(hf,local_files_only=True).cuda().eval().requires_grad_(False)
    processor=AutoImageProcessor.from_pretrained(hf,local_files_only=True,do_rescale=False)
    transform=transforms.Compose([transforms.Resize(518),transforms.ToTensor()])
    def encode(image):
        inputs=processor(images=[transform(image)],return_tensors='pt');_fix_pixel_values(inputs)
        with torch.inference_mode():
            r=model(**inputs.to('cuda'),output_hidden_states=True)
            return r.hidden_states[int(.6667*(len(r.hidden_states)-1))][0].mean(0).float()
    # New images only; base official clean and corrected blur caches stay unchanged.
    for i in list(range(inv['base_images'],len(inv['paths'])))[rank::world]:
        dest=out/f'{i:05d}.npz'
        if dest.exists():continue
        path=inv['paths'][i]
        with Image.open(path) as image:clean=encode(image.convert('RGB')).cpu().numpy()
        blur=torch.stack([F.normalize(encode(load_blurred(path,k)),dim=-1) for k in LEVELS]).mean(0).cpu().numpy()
        assert np.isfinite(clean).all() and np.isfinite(blur).all()
        np.savez(dest,clean=clean,blur=blur,path=path)
        if i%100==0:print('FEATURE',rank,i,'/',len(inv['paths']),flush=True)
    dump(out/f'complete_{rank}.json',{'rank':rank,'world':world})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inventory','eeg','features']);args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    globals()[args.stage]()
