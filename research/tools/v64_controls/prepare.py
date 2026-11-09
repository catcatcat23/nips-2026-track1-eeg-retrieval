"""Build new immutable run snapshots; never alter the v64 source or caches."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
V64=BASE/'reve_v64_stage1_ema'
HERE=Path(__file__).resolve().parent
cfg0=json.loads((V64/'config.json').read_text())

def source(root):
    shutil.copytree(V64/'source',root/'source')
    for p in (root/'source').rglob('*'):
        if p.is_file():p.chmod(0o644)
    for name in ['extensions.py','run_control.py','check_cpu.py','summarize_loso.py']:shutil.copyfile(HERE/name,root/'source'/name)
    p=root/'source/local_ot_train_ema.py';s=p.read_text()
    s=s.replace('total=np.zeros(4)','total=np.zeros(5)').replace("['clean','blur','local']","['clean','blur','local','identity']").replace("['total','clean','blur','local']","['total','clean','blur','local','identity']")
    s=s.replace('public test development, not independent',"see config selection_split; held-out subject never used for selection")
    p.write_text(s)
    hashes={str(p.relative_to(root/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').rglob('*') if p.is_file()}
    (root/'source_sha256.json').write_text(json.dumps(hashes,indent=2))

def config(root,**changes):
    cfg=dict(cfg0);cfg.update(run_root=str(root),baseline='v64 actual immutable source',only_change=changes.get('experiment'),**changes)
    (root/'config.json').write_text(json.dumps(cfg,indent=2))
    for name in ['local.npy','local_complete.json']:(root/name).symlink_to(V64/name)
    return cfg

def launch(root,gpu,dependency,extra=''):
    text=f'''#!/bin/bash
set -euo pipefail
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES={gpu} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH={root}/source:{BASE}/upstream_dinov2_mean_sub01_seed21/deps
cd {root}
trap 'echo failed > queue_status.txt' ERR
while true; do
 if [ "$(cat {dependency}/queue_status.txt 2>/dev/null || true)" = complete ]; then
  used=$(nvidia-smi -i {gpu} --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$used" -lt 1000 ]; then break; fi
 fi
 sleep 30
done
{extra}
echo preflight > queue_status.txt
python -u source/run_control.py --config config.json --preflight > preflight.log 2>&1
echo training > queue_status.txt
python -u source/run_control.py --config config.json > train.log 2>&1
echo complete > queue_status.txt
'''
    (root/'launch.sh').write_text(text);(root/'queue_status.txt').write_text('queued_waiting_for_dependency_and_gpu\n')

for version,experiment,opts,gpu,dep in [
 (77,'v64 + 100% same-subject same-image four distinct trials; original channel mask retained',dict(fixed_repeats=4),3,'reve_v75_cross_subject_four_equal'),
 (78,'v64 + subject supervised contrastive auxiliary loss on pooled REVE via independent projection; original augmentation unchanged',dict(identity_weight=.1,identity_temperature=.1,identity_head='LayerNorm512-Linear256-GELU-Linear128',identity_positives='same subject excluding self; all other subjects negative'),2,'reve_v76_cross_subject_four_weighted')]:
    root=BASE/('reve_v77_v64_mean4' if version==77 else 'reve_v78_v64_identity')
    root.mkdir();source(root);config(root,experiment=experiment,**opts)
    (root/'predictability').symlink_to(V64/'predictability',target_is_directory=True)
    launch(root,gpu,BASE/dep)

root=BASE/'reve_v79_v64_loso';root.mkdir();source(root)
shared=root/'train_only_weights';shared.mkdir()
config(shared,experiment='training-only regional predictability cache for LOSO',training_splits=['train'])
p=root/'source/local_ot_weights.py';s=p.read_text().replace("for s in ['train','val']","for s in ['train']").replace("training_splits=['train','val']","training_splits=['train']")
p.write_text(s)
# Recompute manifest after changing weight-estimation scope.
(root/'source_sha256.json').write_text(json.dumps({str(p.relative_to(root/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').rglob('*') if p.is_file()},indent=2))
folds=[]
for subject in range(1,11):
    fold=root/f'fold_{subject:02d}';fold.mkdir()
    (fold/'source').symlink_to(root/'source',target_is_directory=True)
    config(fold,experiment=f'v64 LOSO THINGS fold {subject}; fresh pretrained, nine downstream training subjects',heldout_subject=f'Gifford2022Large/{subject}',training_splits=['train'],selection_split='val_seen_subjects',split='train: original train images of nine subjects; val: original val images of same nine; test: original 200 test images of held-out subject',selection='seen-subject original validation blur single Top5; best across Raw and EMA variants; heldout test once',early_stop='seen-subject validation only; patience5',pretraining_caveat='LOSO excludes subject from downstream supervised training, not a claim about foundation pretraining provenance')
    (fold/'predictability').symlink_to(shared/'predictability',target_is_directory=True)
    (fold/'queue_status.txt').write_text('queued\n');folds.append(str(fold))
(root/'folds.json').write_text(json.dumps(folds,indent=2))
# Queue whole 10-fold experiment after v77 on GPU3; CPU weights fitted once.
launch(root,3,BASE/'reve_v77_v64_mean4')
p=root/'launch.sh';s=p.read_text();start=s.index('echo preflight > queue_status.txt')
s=s[:start]+f'''echo preparing_train_only_weights > queue_status.txt
python -u source/local_ot_weights.py --config train_only_weights/config.json > weights.log 2>&1
for fold in fold_{{01..10}}; do
 echo "$fold preflight" > queue_status.txt
 python -u source/run_control.py --config "$fold/config.json" --preflight > "$fold/preflight.log" 2>&1
 echo "$fold training" > queue_status.txt
 python -u source/run_control.py --config "$fold/config.json" > "$fold/train.log" 2>&1
done
python -u source/summarize_loso.py --root . > summary.log 2>&1
echo complete > queue_status.txt
'''
p.write_text(s)
# Status/protocol for the LOSO parent is explicit; there is no single train config.
(root/'protocol.json').write_text(json.dumps(dict(folds=10,dataset='THINGS-EEG2',baseline='v64',train='9 subjects, original train images',validation='same 9 subjects, original val images',test='1 heldout subject, 200 original test images, all valid repeats',selection='validation blur single Top5 across raw and EMA; final heldout test once per fold',weights='per-subject predictability fitted on original train only, no heldout subject weights used'),indent=2))
print('Created v77/v78 and v79 ten-fold queue snapshots')
