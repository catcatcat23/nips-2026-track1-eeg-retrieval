"""Replay ZIP; compare unrecovered h vs clean gallery, using all test trials."""
import importlib.util,json,socket,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from build_alljoined_ensemble import BASE,OUT

def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'diagnostic';root.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT/'reve_alljoined_v69_v85_ensemble_track1.zip') as z:
        assert z.testzip() is None;z.extractall(root)
    spec=importlib.util.spec_from_file_location('submission_ensemble',root/'submission.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cfg=json.loads((BASE/'reve_v67_alljoined2025_v64recipe/config.json').read_text());m=json.loads(Path(cfg['data_manifest']).read_text())
    ids=np.array([e[1] for r in m['test'] for e in r['events']]);subjects=np.array([r['subject'] for r in m['test'] for e in r['events']])
    gallery_ids,y=np.unique(ids,return_inverse=True);_,sid=np.unique(subjects,return_inverse=True);group=sid*len(gallery_ids)+y
    count=np.bincount(group);assert len(gallery_ids)==200 and len(count)==4000 and (count>0).all()
    gallery=F.normalize(torch.tensor(np.load(cfg['clean_features'])[gallery_ids],device='cuda'),dim=-1)
    model=mod.Solver().load_model(dict(n_chans=32,n_times=120,sfreq=120.,ch_names=m['channels'],n_outputs=1536,submission_dir=root,device='cuda'))
    def no_network(*a,**kw):raise RuntimeError('network disabled')
    socket.socket.connect=no_network
    names=['v69_blur_to_clean','v85_blur_to_clean','ensemble_blur_to_clean','ensemble_recovered_to_clean']
    sums={k:np.zeros((4000,200),np.float64) for k in names};hits={k:np.zeros(2,np.int64) for k in names}
    captured=[[],[]]
    for i,member in enumerate(model.models):member.head.register_forward_hook(lambda module,args,out,i=i:captured[i].append(out.detach()))
    off=0;start=time.time()
    with torch.inference_mode():
        for ri,r in enumerate(m['test']):
            arr=np.load(r['array'],mmap_mode='r')
            for j in range(0,len(arr),256):
                x=torch.tensor(arr[j:j+256].copy(),device='cuda');captured[0].clear();captured[1].clear()
                clean=model.predict(x);h0=torch.cat(captured[0]);h1=torch.cat(captured[1])
                for name,pred in zip(names,[h0,h1,.75*h0+.25*h1,clean]):
                    scores=(pred@gallery.T).cpu().numpy();assert np.isfinite(scores).all()
                    top=np.argsort(-scores,axis=1)[:,:5];target=y[off:off+len(x)]
                    hits[name]+=[(top[:,0]==target).sum(),(top==target[:,None]).any(1).sum()]
                    np.add.at(sums[name],group[off:off+len(x)],scores)
                off+=len(x)
            if ri%20==0:
                status=dict(samples=off,total=len(ids),seconds=time.time()-start);(root/'status.json').write_text(json.dumps(status));print(status,flush=True)
    assert off==len(ids)
    result={}
    for name in names:
        scores=sums[name]/count[:,None];top=np.argsort(-scores,axis=1)[:,:5];target=np.tile(np.arange(200),20)
        result[name]=dict(single_trial=dict(zip(['top1','top5'],(hits[name]/off).tolist())),subject_aggregated=dict(top1=float((top[:,0]==target).mean()),top5=float((top==target[:,None]).any(1).mean())))
    expected=json.loads((OUT/'build.json').read_text())['expected_clean']
    for group_name,n in [('single_trial',off),('subject_aggregated',4000)]:
        for key in ['top1','top5']:assert abs(result['ensemble_recovered_to_clean'][group_name][key]-expected[group_name][key])*n<=10
    report=dict(results=result,samples=off,candidates=200,groups=4000,seconds=time.time()-start,zip_replay_passed=True,protocol='single trial forward; raw predictions averaged per subject-image; clean gallery; no refit')
    (root/'result.json').write_text(json.dumps(report,indent=2));(root/'status.json').write_text(json.dumps(dict(state='complete',samples=off)))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
