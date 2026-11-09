import hashlib,json,shutil
from pathlib import Path
import numpy as np
base=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments');root=base/'reve_v79_v64_loso';v64=base/'reve_v64_stage1_ema'
data=base/'reve_v13_reference/data'
ids=np.unique(np.concatenate([np.load(data/f'{s}_ids.npy') for s in ['train','val']]))
val_ids=np.sort(np.random.default_rng(33).choice(ids,len(ids)//10,replace=False)).tolist()
(root/'validation_ids.json').write_text(json.dumps(val_ids))
for run in [base/'reve_v77_v64_mean4',base/'reve_v78_v64_identity',root]:
 shutil.copyfile(Path(__file__).parent/'extensions.py',run/'source/extensions.py')
for folder in [root/'train_only_weights']+sorted(root.glob('fold_*')):
 p=folder/'config.json';c=json.loads(p.read_text());c['validation_ids_path']=str(root/'validation_ids.json');c['training_splits']=['train','val'];c['split']='combine original train+val; fixed seed33 10% image holdout (1654 images) on seen subjects for validation, remaining14886 images for training; heldout subject excluded throughout; original200 test images of heldout subject for final test only';c['selection']='seen-subject fixed image-disjoint validation blur single Top5; final heldout test once'
 p.write_text(json.dumps(c,indent=2))
s=(v64/'source/local_ot_weights.py').read_text()
s=s.replace("subjects=np.unique", "validation_ids=np.array(json.loads(Path(cfg['validation_ids_path']).read_text()));subjects=np.unique")
s=s.replace(";assert not np.intersect1d(targetids,testids).size", ";targetids=targetids[~np.isin(targetids,validation_ids)];assert not np.intersect1d(targetids,testids).size")
s=s.replace("indices=np.flatnonzero(subs==subject)","indices=np.flatnonzero((subs==subject)&np.isin(ids,targetids))")
s=s.replace("test_used=False", "test_used=False,validation_ids_excluded=validation_ids.tolist()")
(root/'source/local_ot_weights.py').write_text(s)
p=root/'protocol.json';v=json.loads(p.read_text());v.update(train='9 subjects, 14886 training images from original train+val',validation='same9 subjects,1654 disjoint training images fixed seed33',weights='per-subject fitted on14886 training images only; heldout subject weights never used',foundation_note='Downstream LOSO; does not establish exclusion from REVE foundation pretraining');p.write_text(json.dumps(v,indent=2))
for run in [base/'reve_v77_v64_mean4',base/'reve_v78_v64_identity',root]:
 (run/'source_sha256.json').write_text(json.dumps({str(p.relative_to(run/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'source').rglob('*') if p.is_file()},indent=2))
print('Fixed LOSO image split before launching queues')
