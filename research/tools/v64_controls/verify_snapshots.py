import ast,hashlib,json
from pathlib import Path
import numpy as np
base=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
for name in ['reve_v77_v64_mean4','reve_v78_v64_identity','reve_v79_v64_loso']:
 root=base/name
 for rel,sha in json.loads((root/'source_sha256.json').read_text()).items():
  p=root/'source'/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
  if p.suffix=='.py':ast.parse(p.read_text())
root=base/'reve_v79_v64_loso';data=base/'reve_v13_reference/data'
ids=np.concatenate([np.load(data/f'{s}_ids.npy') for s in ['train','val']]);subjects=np.concatenate([np.load(data/f'{s}_subjects.npy').astype(str) for s in ['train','val']])
testids=np.load(data/'test_ids.npy');testsubjects=np.load(data/'test_subjects.npy').astype(str)
val=np.array(json.loads((root/'validation_ids.json').read_text()));rows=[]
for fold in sorted(root.glob('fold_*')):
 cfg=json.loads((fold/'config.json').read_text());held=cfg['heldout_subject']
 train=(subjects!=held)&~np.isin(ids,val);valid=(subjects!=held)&np.isin(ids,val);test=testsubjects==held
 assert train.any() and valid.any() and test.any()
 assert not np.intersect1d(ids[train],ids[valid]).size
 assert not np.intersect1d(ids,testids).size
 assert held not in subjects[train] and held not in subjects[valid]
 assert len(np.unique(testids[test]))==200
 rows.append(dict(subject=held,train_count=int(train.sum()),validation_count=int(valid.sum()),test_count=int(test.sum()),train_images=len(np.unique(ids[train])),validation_images=len(np.unique(ids[valid])),test_images=200))
print(json.dumps(dict(passed=True,folds=rows),indent=2))
