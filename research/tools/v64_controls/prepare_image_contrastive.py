"""Prepare v92 from verified v82 sources; never launch or overwrite a run."""
import ast
import difflib
import hashlib
import json
import shutil
import time
from pathlib import Path

BASE = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
PARENT = BASE / 'reve_v82_grouped_identity_weight1'
OUT = BASE / 'reve_v92_image_contrastive'
LOCAL = Path(__file__).resolve().parent
PYTHON = '/mnt/DATA-4/anteng/eeg-retrieval/env/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    assert not OUT.exists(), f'Refusing to overwrite {OUT}'
    parent_source = PARENT / 'source_gradient_monitor'
    parent_manifest = PARENT / 'source_gradient_monitor_sha256.json'
    expected = json.loads(parent_manifest.read_text())
    actual = {str(p.relative_to(parent_source)): digest(p)
              for p in parent_source.rglob('*.py')}
    assert actual == expected, 'Parent source hashes or file inventory changed'
    for name in ['image_contrastive.py', 'run_image_contrastive.py']:
        ast.parse((LOCAL / name).read_text())
    original = json.loads((PARENT / 'config.json').read_text())
    dependencies = {name: (PARENT / name).resolve(strict=True)
                    for name in ['local.npy', 'local_complete.json', 'predictability']}
    source_root = Path(original['source_root'])
    cache_paths = [source_root / 'data' / f'{s}_{k}.npy'
                   for s in ['train', 'val', 'test'] for k in ['eeg', 'ids', 'subjects']]
    cache_paths += [source_root / 'data/clean.npy', source_root / 'data/manifest.json',
                    source_root / 'features/blur.npy', source_root / 'features/complete.json']
    hf_root = Path('/mnt/DATA-4/anteng/eeg-retrieval/huggingface')
    pretrained = hf_root / 'hub/models--brain-bzh--reve-base/snapshots' / original['pretrained_snapshot']
    assert pretrained.is_dir()
    for path in cache_paths:
        assert path.exists(), path
    disk = shutil.disk_usage(BASE)
    assert disk.free > 20 * 1024**3, 'Less than 20 GiB free on data disk'

    OUT.mkdir()
    source = OUT / 'source'
    shutil.copytree(parent_source, source, ignore=shutil.ignore_patterns('__pycache__'))
    for name, path in dependencies.items():
        (OUT / name).symlink_to(path, target_is_directory=path.is_dir())
    for name in ['image_contrastive.py', 'run_image_contrastive.py']:
        shutil.copyfile(LOCAL / name, source / name)
    cfg = {k: v for k, v in original.items() if not k.startswith('identity_')}
    cfg.update(run_root=str(OUT), parent=str(PARENT), identity_weight=0,
               experiment='v92 EEG-EEG image SupCon, same-image positives across subjects',
               image_contrastive_weight=1.0, image_contrastive_temperature=0.1,
               auxiliary_seed=810033, image_contrastive_location='pooled512',
               auxiliary_positives='same image across all subjects, self excluded; 7 per anchor',
               auxiliary_negatives='different images across all subjects; 248 per anchor',
               only_change='Replace auxiliary image-conditioned subject SupCon with image SupCon; same auxiliary sampling and budget',
               initialization='fresh pretrained REVE, no trained retrieval checkpoint loaded',
               selection='public test development: blur single-trial Top5 per raw/EMA variant; not independent heldout evaluation')
    write(OUT / 'config.json', cfg)
    write(OUT / 'config_diff.json', {
        k: dict(before=original.get(k), after=cfg.get(k),
                before_present=k in original, after_present=k in cfg)
        for k in sorted(set(original) | set(cfg)) if original.get(k) != cfg.get(k)})

    loop = source / 'local_ot_train_ema.py'
    text = loop.read_text()
    assert text.count('from grouped_identity import auxiliary_backward') == 1
    text = text.replace('from grouped_identity import auxiliary_backward',
                        'from image_contrastive import auxiliary_backward')
    text = text.replace('identity', 'image_contrastive')
    init = " torch.manual_seed(33);model=LocalOTModel(manifest).cuda();model.perturbation_kind=cfg.get('augmentation','none')\n"
    assert text.count(init) == 1
    text = text.replace(init, init + " from image_contrastive import state_sha256\n initial_hash=state_sha256(model)\n assert initial_hash==json.loads((out/'image_contrastive_gpu_check.json').read_text())['initial_state_sha256'], 'Formal model initialization differs from verified v82 initialization'\n write(out/'initialization.json',dict(state_sha256=initial_hash,matches_v82=True,trained_retrieval_checkpoint_loaded=False))\n")
    old = 'see config selection_split; held-out subject never used for selection'
    assert old in text
    text = text.replace(old, 'public test development selection; not independent heldout testing')
    loop.chmod(0o644)
    loop.write_text(text)
    monitor = source / 'gradient_balance.py'
    text = monitor.read_text().replace('identity', 'image_contrastive')
    monitor.chmod(0o644)
    monitor.write_text(text)
    diff = []
    for p in sorted(source.rglob('*.py')):
        ast.parse(p.read_text())
        rel = p.relative_to(source)
        old = parent_source / rel
        before = old.read_text().splitlines(True) if old.exists() else []
        diff.extend(difflib.unified_diff(before, p.read_text().splitlines(True),
                                         fromfile=f'v82/{rel}', tofile=f'v92/{rel}'))
    (OUT / 'source_diff.patch').write_text(''.join(diff))
    hashes = {str(p.relative_to(source)): digest(p) for p in source.rglob('*.py')}
    write(OUT / 'source_sha256.json', hashes)
    write(OUT / 'provenance.json', dict(
        created_unix=time.time(), parent=str(PARENT), parent_source=str(parent_source),
        parent_manifest_sha256=digest(parent_manifest), parent_source_verified=True,
        parent_actual_sha256=actual, source_sha256=hashes,
        config_sha256=digest(OUT / 'config.json'), preparer_sha256=digest(Path(__file__)),
        git='Workspace has no usable git repository; actual file SHA256 manifest used',
        disk_free_bytes=disk.free, pretrained_snapshot=str(pretrained),
        dependencies={k: str(v) for k, v in dependencies.items()},
        caches=[dict(path=str(p.resolve()), bytes=p.stat().st_size,
                     mtime_ns=p.stat().st_mtime_ns) for p in cache_paths]))
    write(OUT / 'protocol.json', dict(
        dataset='THINGS-EEG2; original train+val combined',
        retrieval='unchanged clean+blur+0.1 Local OT, original random batch256',
        auxiliary='32 images x4 distinct subjects x2 distinct trials=256; no EEG augmentation',
        positive='same image, including same and different subjects; self excluded; 7',
        negative='different image, regardless of subject; 248',
        loss='mean-positive supervised contrastive CE on normalized pooled512, temperature0.1, weight1',
        optimization='one step after retrieval backward and weighted auxiliary backward; shared gradient clipping',
        model='fresh pretrained REVE; original heads; no identity loss or new classifier',
        comparison='v82 matches two-batch sampling and compute. Positive/negative composition and loss semantics both change. Weight1 does not guarantee equal encoder gradient strength; log actual gradient norms and cosine.',
        evaluation='all four clean/blur single/aggregate metrics; raw and EMA; public-test development selection unchanged',
        status='prepared only; CPU/GPU preflight and training not yet run'))
    for p in source.rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    for p in sorted((p for p in source.rglob('*') if p.is_dir()), reverse=True):
        p.chmod(0o555)
    source.chmod(0o555)
    (OUT / 'config.json').chmod(0o444)

    launch = r'''#!/bin/bash
set -euo pipefail
RUN=__RUN__
PY=__PY__
GPU=${1:-0}
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'GPU must be an integer' >&2; exit 2; }
cd "$RUN"
exec 9>"$RUN/run.lock"
flock -n 9 || { echo 'This run already has a launcher' >&2; exit 3; }
[[ ! -d stage1 ]] || { echo 'Refusing to overwrite existing stage1; use an explicit resume task' >&2; exit 4; }
state() {
 "$PY" -B -c 'import json,sys,time,pathlib; p=pathlib.Path(sys.argv[1]); t=p.with_suffix(".tmp"); t.write_text(json.dumps(dict(state=sys.argv[2],exit_code=int(sys.argv[3]),time_unix=time.time(),gpu=int(sys.argv[4]))));t.replace(p)' "$RUN/launch_status.json" "$1" "${2:-0}" "$GPU"
 echo "$1" > "$RUN/queue_status.txt"
}
trap 'rc=$?; if [ "$rc" -ne 0 ]; then state failed "$rc"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/mnt/DATA-4/anteng/eeg-retrieval/huggingface TMPDIR=/mnt/DATA-4/anteng/tmp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$RUN/source:/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_dinov2_mean_sub01_seed21/deps"
state cpu_preflight
"$PY" -B -u source/run_image_contrastive.py --config config.json --preflight cpu > cpu_preflight.log 2>&1
state waiting_for_gpu
while true; do
 read -r used util < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
 apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)
 if [[ "$used" -lt 1000 && "$util" -eq 0 && -z "$apps" ]]; then break; fi
 sleep 30
done
state gpu_preflight
"$PY" -B -u source/run_image_contrastive.py --config config.json --preflight gpu > gpu_preflight.log 2>&1
state training
"$PY" -B -u source/run_image_contrastive.py --config config.json > train.log 2>&1
state complete
'''
    launch = launch.replace('__RUN__', str(OUT)).replace('__PY__', PYTHON)
    (OUT / 'launch.sh').write_text(launch)
    (OUT / 'launch.sh').chmod(0o555)
    (OUT / 'queue_status.txt').write_text('prepared_not_started\n')
    print(OUT)


if __name__ == '__main__':
    main()
