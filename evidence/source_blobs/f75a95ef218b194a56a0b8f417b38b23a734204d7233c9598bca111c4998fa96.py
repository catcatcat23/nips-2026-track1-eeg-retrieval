"""Validate filepath/ID mapping and merge a complete reference cache without overwrite."""
import json
from pathlib import Path
import numpy as np
from data import config
from extract_reference import SCHEMA

def main():
    root=Path(config()['run_root']);src=root/'reference_shards';out=root/'features'
    manifest=json.loads((root/'data/manifest.json').read_text());paths=manifest['paths']
    ids=np.union1d(np.load(root/'data/train_ids.npy'),np.load(root/'data/val_ids.npy'))
    first=json.loads((src/'complete_rank0.json').read_text())
    for rank in range(first['world_size']):
        meta=json.loads((src/f'complete_rank{rank}.json').read_text());assert meta['schema']==SCHEMA
    # Validate everything before creating the destination.
    for i in ids:
        with np.load(src/f'{i:05d}.npz') as z:
            assert int(z['id'])==i and str(z['filepath'])==paths[i] and str(z['schema'])==SCHEMA
            assert z['levels'].shape==(12,1536) and np.isfinite(z['levels']).all()
            np.testing.assert_allclose(np.linalg.norm(z['levels'],axis=-1),1,rtol=1e-5,atol=1e-5)
            np.testing.assert_allclose(z['blur'],z['levels'].mean(0),rtol=1e-5,atol=1e-7)
    out.mkdir(exist_ok=False)
    blur=np.lib.format.open_memmap(out/'blur.npy',mode='w+',dtype='float32',shape=(len(paths),1536));blur[:]=np.nan
    for i in ids:
        with np.load(src/f'{i:05d}.npz') as z:blur[i]=z['blur']
    blur.flush();available=np.zeros(len(paths),bool);available[ids]=True;np.save(out/'available.npy',available)
    (out/'complete.json').write_text(json.dumps({'schema':SCHEMA,'images':len(ids),'source_commit':'94b3a31','order':'PIL RGB resize224; OpenCV uint8 blur; Resize518; official processor; layer0.6667 token mean; per-scale L2; mean','clean':'unchanged official cache'},indent=2))
if __name__=='__main__':main()
