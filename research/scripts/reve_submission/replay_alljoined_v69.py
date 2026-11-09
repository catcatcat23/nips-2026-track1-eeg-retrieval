"""Replay actual exported ZIP on every public Alljoined test epoch."""
import importlib.util,json,socket,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
OUT=BASE/'submission_v69_alljoined_ep15_20261105'
sys.path[:0]=[str(BASE/'reve_v69_joint_stage1/source'),'/home/Anteng/eeg-retrieval/competition-2026']
from v33_recovery_stage2 import metrics

def main():
    folder=OUT/'real_zip_replay';folder.mkdir(exist_ok=True)
    archive=OUT/'reve_v69_alljoined_ema9998_ep15_track1.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        z.extractall(folder)
    spec=importlib.util.spec_from_file_location('actual_zip_v69',folder/'submission.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cfg=json.loads((BASE/'reve_v67_alljoined2025_v64recipe/config.json').read_text())
    manifest=json.loads(Path(cfg['data_manifest']).read_text())
    meta=dict(n_chans=32,n_times=120,sfreq=120.,ch_names=manifest['channels'],n_outputs=1536,submission_dir=folder,device='cuda')
    original=socket.socket.connect
    def guard(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('Network forbidden')
        return original(sock,address)
    socket.socket.connect=guard
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=mod.Solver().load_model(meta)
    total=sum(len(r['events']) for r in manifest['test'])
    pred=np.lib.format.open_memmap(OUT/'real_predictions.npy',mode='w+',dtype='float32',shape=(total,1536))
    ids=[];subjects=[];offset=0;start=time.time()
    for ri,r in enumerate(manifest['test']):
        assert r['channels']==meta['ch_names']
        x=np.load(r['array'],mmap_mode='r');assert x.shape==(len(r['events']),32,120)
        for j in range(0,len(x),256):
            b=torch.from_numpy(x[j:j+256].copy()).cuda()
            y=model.predict(b).cpu().numpy();assert np.isfinite(y).all()
            pred[offset:offset+len(y)]=y;offset+=len(y)
        ids.extend([e[1] for e in r['events']]);subjects.extend([r['subject']]*len(x))
        if ri%20==0:
            report=dict(state='running',samples=offset,total=total,seconds=time.time()-start)
            (OUT/'replay_status.json').write_text(json.dumps(report));print(report,flush=True)
    pred.flush();assert offset==total
    result=metrics(pred,np.array(ids),np.array(subjects),np.load(cfg['clean_features']))
    expected=json.loads((OUT/'build.json').read_text())['expected_clean'];diff={}
    for group in ['single_trial','subject_aggregated']:
        for k in ['top1','top5']:
            diff[group+'.'+k]=round(abs(result[group][k]-expected[group][k])*result[group]['count'])
            assert diff[group+'.'+k]<=5,(group,k,diff)
    report=dict(verified=True,metrics=result,expected=expected,rank_flip_counts=diff,seconds=time.time()-start,
                samples=total,artifact=str(archive),network='disabled',scope='full public Alljoined test; actual ZIP reload, local environment')
    (OUT/'real_replay.json').write_text(json.dumps(report,indent=2))
    (OUT/'replay_status.json').write_text(json.dumps(dict(state='complete',samples=total,seconds=time.time()-start)))
    print({k:result[k] for k in ['single_trial','subject_aggregated']},flush=True)

if __name__=='__main__':main()
