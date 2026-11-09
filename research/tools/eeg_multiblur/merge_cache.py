import json
from pathlib import Path
import numpy as np
from data import config

def main():
    root=Path(config()['run_root']);src=root/'feature_shards';out=root/'features';out.mkdir(exist_ok=True)
    for r in range(2):assert (src/f'complete_rank{r}.json').is_file()
    n=len(json.loads((root/'data/manifest.json').read_text())['paths'])
    blur=np.lib.format.open_memmap(out/'blur.npy',mode='w+',dtype='float32',shape=(n,1536))
    local=np.lib.format.open_memmap(out/'local.npy',mode='w+',dtype='float32',shape=(n,9,1536))
    seen=np.zeros(n,bool)
    for p in sorted(src.glob('*.npz')):
        z=np.load(p);ids=z['ids'];assert not seen[ids].any()
        blur[ids]=z['blur'];local[ids]=z['local'];seen[ids]=True
    required=np.union1d(np.load(root/'data/train_ids.npy'),np.load(root/'data/val_ids.npy'))
    assert seen[required].all() and seen.sum()==len(required)
    blur.flush();local.flush();np.save(out/'available.npy',seen)
    (out/'complete.json').write_text(json.dumps({'images':int(seen.sum()),'test_auxiliary_features':'not used or extracted'}))
if __name__=='__main__':main()
