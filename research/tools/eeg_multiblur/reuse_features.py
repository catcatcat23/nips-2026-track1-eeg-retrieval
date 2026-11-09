"""Reuse image-only caches after proving identical row identity and clean targets."""
import json
from pathlib import Path
import numpy as np
from data import config

root=Path(config()['run_root'])
old=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/multiblur_v1')
new_manifest=json.loads((root/'data/manifest.json').read_text())
old_manifest=json.loads((old/'data/manifest.json').read_text())
assert new_manifest['paths']==old_manifest['paths']
assert new_manifest['input_shape']==[63,120]
assert new_manifest['val_images_absent_from_train']==0
assert np.array_equal(np.load(root/'data/clean.npy'),np.load(old/'data/clean.npy'))
required=np.union1d(np.load(root/'data/train_ids.npy'),np.load(root/'data/val_ids.npy'))
assert np.load(old/'features/available.npy')[required].all()
dest=root/'features';dest.mkdir(exist_ok=True)
for name in ['blur.npy','local.npy','available.npy']:
    p=dest/name
    if not p.exists():p.symlink_to(old/'features'/name)
# Deliberately do not copy EEG-derived region weights.
p=root/'smoke_features'
if not p.exists():p.symlink_to(old/'smoke_features')
(root/'cache_pairing_check.json').write_text(json.dumps({'passed':True,'input_shape':[63,120],
    'clean_targets':'bitwise identical','image_rows':'identical paths and order',
    'region_weights':'must recompute using new EEG'},indent=2))
print('CACHE PAIRING PASSED',flush=True)
