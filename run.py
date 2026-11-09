"""v64 single-stage REVE + Local OT + EMA. Fresh output directories required; no silent overwrite."""
import argparse, hashlib, json, os, runpy, sys
from pathlib import Path
HOME = Path(__file__).resolve().parent
sys.path.insert(0, str(HOME/'src'))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def resolve(name, assets, output):
    cfg=json.loads((HOME/'configs'/name).read_text())
    if name.startswith('things'):
        cfg.update(run_root=str(output),source_root=assets['things_data_root'],test_blur_shards=assets['things_test_blur_shards'])
    else:
        cfg.update(run_root=str(output),data_manifest=assets['alljoined_data_manifest'],test_blur_shards=assets['things_test_blur_shards'],
                   clean_features=str(Path(assets['things_data_root'])/'data/clean.npy'),blur_features=str(Path(assets['things_data_root'])/'features/blur.npy'),
                   local_features=assets['things_local_features'])
    return cfg

def main():
    p=argparse.ArgumentParser()
    p.add_argument('step',choices=['check','prepare','preflight','train','evaluate','alljoined'])
    p.add_argument('--assets',type=Path,default=HOME/'configs/assets.local.json')
    p.add_argument('--output',type=Path,help='Explicit data-disk directory; required except for check')
    p.add_argument('--checkpoint',type=Path,help='evaluate: checkpoint to score (default: submitted EMA0.999 epoch 9)')
    p.add_argument('--reuse-local',action='store_true',help='prepare: link the existing Local OT image cache instead of re-extracting')
    p.add_argument('--preflight',action='store_true',help='alljoined: preflight only')
    a=p.parse_args()
    if a.step != 'check':
        if a.output is None:
            p.error('--output must point to a data-disk directory')
        resolved = a.output.expanduser().resolve()
        if any(resolved.is_relative_to(Path(x)) for x in ('/home', '/tmp')):
            p.error('--output resolves to system storage; use a data disk')
        a.output = resolved
    assets=json.loads(a.assets.read_text())
    os.environ.setdefault('HF_HOME',assets['hf_home']);os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('LOCAL_OT_UPSTREAM_SCRIPT',assets['local_ot_upstream_script'])
    data=Path(assets['things_data_root'])
    if a.step=='check':
        import numpy as np
        m=json.loads((data/'data/manifest.json').read_text())
        for split in ['train','val','test']:
            x=np.load(data/'data'/f'{split}_eeg.npy',mmap_mode='r');ids=np.load(data/'data'/f'{split}_ids.npy')
            assert x.shape==(len(ids),63,120)
        assert len(m['channels'])==63
        assert sha(assets['submission_checkpoint'])==assets['submission_checkpoint_sha256']
        print('PASS: THINGS-EEG2 cache shapes, 63 channels, submitted checkpoint hash');return
    if a.step=='alljoined':
        out=a.output;out.mkdir(parents=True,exist_ok=True)  # preflight and training share the predictability cache
        cfg=resolve('alljoined_v67.json',assets,out);(out/'config.json').write_text(json.dumps(cfg,indent=2))
        sys.path.insert(0,str(HOME/'alljoined'))
        sys.argv=['alljoined_v67.py','--config',str(out/'config.json')]+(['--preflight'] if a.preflight else [])
        runpy.run_path(str(HOME/'alljoined/alljoined_v67.py'),run_name='__main__');return
    out=a.output;cfg=resolve('things_v64.json',assets,out);config=out/'config.json'
    if a.step=='prepare':
        out.mkdir(parents=True,exist_ok=False);config.write_text(json.dumps(cfg,indent=2))
        if a.reuse_local:
            src=Path(assets['things_local_features'])
            assert sha(src)==assets['things_local_features_sha256']
            (out/'local.npy').symlink_to(src);(out/'local_complete.json').symlink_to(src.parent/'local_complete.json')
        else:
            sys.path.insert(0,str(HOME/'cache_reference'));sys.argv=['local_ot_extract.py','--config',str(config)]
            runpy.run_path(str(HOME/'cache_reference/local_ot_extract.py'),run_name='__main__')
        # Per-subject Local OT patch weights from training EEG only (fold-local Ridge predictability).
        sys.argv=['local_ot_weights.py','--config',str(config)];runpy.run_path(str(HOME/'src/local_ot_weights.py'),run_name='__main__');return
    assert config.exists(),'run prepare first'
    import torch
    if a.step=='preflight':
        sys.argv=['preflight_ema.py',str(config)];runpy.run_path(str(HOME/'src/preflight_ema.py'),run_name='__main__');return
    if a.step=='train':
        sys.argv=['local_ot_train_ema.py','train','--config',str(config)]
        import local_ot_train_ema;local_ot_train_ema.main();return
    # evaluate: score one saved checkpoint with the training evaluator (blur + clean, public test).
    import numpy as np
    from local_ot_train import LocalOTModel,evaluate
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=a.checkpoint or Path(assets['submission_checkpoint'])
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=LocalOTModel(ck['manifest']).cuda();model.load_state_dict(ck['model'],strict=True)
    weights=json.loads((out/'predictability/subjects.json').read_text())
    r=evaluate(model,cfg,weights);r={k:{g:r[k][g] for g in ('single_trial','subject_aggregated')} for k in ('blur','clean')}
    path=out/f'evaluate_{checkpoint.stem}.json';path.write_text(json.dumps(dict(checkpoint=str(checkpoint),epoch=ck['epoch'],metrics=r),indent=2))
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
