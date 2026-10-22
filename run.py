"""Only v33/dropout -> v39/B; fresh output directories required."""
import argparse, hashlib, json, os, sys
from pathlib import Path
HOME = Path(__file__).resolve().parent
sys.path.insert(0, str(HOME/'src'))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('step',choices=['check','stage1','extract','stage2'])
    p.add_argument('--assets',type=Path,default=HOME/'configs/assets.local.json')
    p.add_argument('--output',type=Path,default=HOME/'runs/reproduction')
    p.add_argument('--checkpoint',type=Path,help='Fresh stage1 checkpoint; default uses the archived epoch7')
    p.add_argument('--cache',type=Path,help='Fresh extracted h cache; default uses archived cache')
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args();assets=json.loads(a.assets.read_text());data=Path(assets['data_root'])
    os.environ.setdefault('HF_HOME',assets['hf_home']);os.environ.setdefault('HF_HUB_OFFLINE','1')
    if a.step=='check':
        import numpy as np
        m=json.loads((data/'data/manifest.json').read_text())
        assert m['preprocessing_contract']=='competition_eeg_image_v1'
        for split in ['train','val','test']:
            x=np.load(data/'data'/f'{split}_eeg.npy',mmap_mode='r');ids=np.load(data/'data'/f'{split}_ids.npy')
            assert x.shape==(len(ids),63,120)
        for stage in ['stage1','stage2']:
            assert sha(assets[stage+'_checkpoint'])==assets[stage+'_checkpoint_sha256']
        assert sha(HOME/'submission/reve_v39b_epoch8_track1.zip')==json.loads((HOME/'evidence/checksums.json').read_text())['submission/reve_v39b_epoch8_track1.zip']
        print('PASS: data shapes, both original checkpoint hashes, submission hash');return
    import torch
    torch.set_num_threads(4)
    if a.step=='stage1':
        import stage1
        stage1.DATA=data
        sys.argv=['stage1.py','dropout','--root',str(a.output/'stage1')]+(['--preflight'] if a.preflight else [])
        stage1.main();return
    checkpoint=a.checkpoint or Path(assets['stage1_checkpoint'])
    cfg=json.loads((HOME/'configs/stage2.json').read_text())
    cfg.update(source_root=str(data),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
    if a.step=='extract':
        import numpy as np
        from torch.utils.data import DataLoader
        from data import DatasetEEG
        from stage1 import AblationModel
        cache=a.output/'cache';cache.mkdir(parents=True,exist_ok=False)
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=AblationModel(ck['manifest'],'dropout');model.load_state_dict(ck['model'],strict=True)
        model=model.cuda().eval().requires_grad_(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with torch.no_grad():
            for split in ['train','val','test']:
                ds=DatasetEEG(data,split,aux=False)
                arr=np.lib.format.open_memmap(cache/f'{split}_h.npy',mode='w+',dtype='float32',shape=(len(ds),1536));start=0
                for b in DataLoader(ds,batch_size=256,num_workers=2):
                    x=model.preprocess(b['eeg'].cuda());pos=model.positions.to(x).unsqueeze(0).expand(len(x),-1,-1)
                    h=model.blur_head(model.encoder(x,pos=pos,return_output=True)[-1].mean(1)).cpu().numpy()
                    assert np.isfinite(h).all();arr[start:start+len(h)]=h;start+=len(h)
                assert start==len(ds);arr.flush()
                np.save(cache/f'{split}_ids.npy',ds.ids);np.save(cache/f'{split}_subjects.npy',ds.subjects)
                print(split,start,flush=True)
        (cache/'complete.json').write_text(json.dumps({'checkpoint_sha256':cfg['checkpoint_sha256'],'dtype':'float32','encoder_mode':'eval'}));return
    import stage2
    dest=a.output/'stage2';dest.mkdir(parents=True,exist_ok=False)
    cfg.update(run_root=str(dest),cache_root=str(a.cache or assets['cached_h']))
    (dest/'config.json').write_text(json.dumps(cfg,indent=2))
    sys.argv=['stage2.py','--root',str(dest)];stage2.main()

if __name__=='__main__':main()
