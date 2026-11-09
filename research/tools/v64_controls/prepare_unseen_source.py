"""Prepare paired source-only runs; never launch or touch an existing run."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil

from train_unseen_calibration import plan_records


BASE = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
HERE = Path(__file__).resolve().parent
EXCLUDED = ['Gifford2022Large/1', 'Gifford2022Large/2']
SOURCES = [f'Gifford2022Large/{i}' for i in range(3, 11)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=BASE / 'reve_v93_unseen_calibration')
    ap.add_argument('--reference-report', type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    assert root.is_relative_to(BASE) and root != BASE
    assert not root.exists(), 'Refusing to overwrite existing experiment'
    assert shutil.disk_usage(BASE).free > 150 * 1024 ** 3, 'Need at least 150 GiB free for paired trajectories'
    parent = BASE / 'reve_v79_v64_loso'
    source_parent = BASE / 'reve_v91_identity_loso_calibration/source'
    original = json.loads((parent / 'fold_01/config.json').read_text())
    weights_dir = parent / 'train_only_weights/predictability'
    weights = json.loads((weights_dir / 'subjects.json').read_text())
    provenance = json.loads((weights_dir / 'complete.json').read_text())
    val = json.loads((parent / 'validation_ids.json').read_text())
    assert provenance['validation_ids_excluded'] == val and provenance['test_used'] is False
    assert all(s in weights and weights[s]['images'] == 14886 for s in SOURCES)
    added = ['run_unseen_source.py', 'prepare_unseen_source.py',
             'train_unseen_calibration.py', 'unseen_subject_calibrator.py',
             'queue_unseen_calibration.py', 'worker_unseen_calibration.py',
             'summarize_unseen_calibration.py']
    for name in added:
        assert (HERE / name).is_file(), f'Finish source implementation before preparing: {name}'
        ast.parse((HERE / name).read_text())
    if args.reference_report:
        assert args.reference_report.is_file()
    root.mkdir()
    shutil.copytree(source_parent, root / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for p in (root / 'source').rglob('*'):
        if p.is_file():
            p.chmod(0o644)
    for name in added:
        shutil.copyfile(HERE / name, root / 'source' / name)
    if args.reference_report:
        shutil.copyfile(args.reference_report, root / 'reference.md')
        (root / 'reference.md').chmod(0o444)
    shutil.copyfile(parent / 'validation_ids.json', root / 'validation_ids.json')
    shared = root / 'shared_predictability'
    shared.mkdir()
    write(shared / 'subjects.json', {s: weights[s] for s in SOURCES})
    provenance.update(subjects=SOURCES, excluded_subjects=EXCLUDED,
                      reuse='each subject independently fitted; only S3--S10 records copied',
                      parent_subjects_json=str(weights_dir / 'subjects.json'),
                      parent_subjects_sha256=digest(weights_dir / 'subjects.json'))
    write(shared / 'complete.json', provenance)
    hashes = {}
    for p in (root / 'source').rglob('*'):
        if p.is_file():
            if p.suffix == '.py':
                ast.parse(p.read_text())
            hashes[str(p.relative_to(root / 'source'))] = digest(p)
            p.chmod(0o444)
    write(root / 'source_sha256.json', hashes)
    for arm, weight in [('base', 0.), ('identity', 1.)]:
        out = root / arm
        out.mkdir()
        (out / 'source').symlink_to(root / 'source', target_is_directory=True)
        (out / 'predictability').symlink_to(shared, target_is_directory=True)
        for name in ['local.npy', 'local_complete.json']:
            (out / name).symlink_to((parent / 'fold_01' / name).resolve())
        cfg = dict(original)
        for key in ['heldout_subject', 'initial_reference', 'reference_root', 'test_blur_shards',
                    'warm_start', 'resume', 'checkpoint', 'base_checkpoint', 'head_checkpoint']:
            cfg.pop(key, None)
        cfg.update(run_root=str(out), source_snapshot=str(root / 'source'),
                   source_hashes_path=str(root / 'source_sha256.json'),
                   validation_ids_path=str(root / 'validation_ids.json'),
                   excluded_subjects=EXCLUDED, source_subjects=SOURCES,
                   identity_weight=weight, identity_temperature=.1, identity_location='pooled512',
                   training_splits=['train', 'val'], selection_split='val_source_subjects',
                   experiment=f'v93 {arm}: source-only S3--S10; S1/S2 entirely excluded',
                   only_change='paired source models differ only in direct pooled512 subject contrast weight 0 versus 1',
                   initialization='fresh original pretrained REVE; all retrieval/OT heads randomly initialized, seed33',
                   initialization_kind='original_pretrained_reve_fresh_heads',
                   split='S3--S10 only; train14886 images; source-val1654 disjoint images; no target EEG used',
                   selection='source-val blur single-trial Top5, best Raw/EMA; no S1/S2 evaluation',
                   early_stop='source-val only, patience5',
                   normal_forward_uses_calibrator=False,
                   source_recipe='immutable v79/v91 full-parameter training recipe')
        write(out / 'config.json', cfg)
        (out / 'queue_status.txt').write_text('prepared_not_launched\n')
        # Resource scheduling belongs to the parent dispatcher. An explicit GPU
        # index is required; this launcher never silently picks an occupied GPU.
        q = shlex.quote
        script = f'''#!/bin/bash
set -euo pipefail
gpu="${{1:?Pass an explicitly allocated GPU index}}"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/mnt/DATA-4/anteng/tmp OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH={q(str(root / 'source'))}:{q(str(BASE / 'upstream_dinov2_mean_sub01_seed21/deps'))}
cd {q(str(out))}
trap 'echo failed > queue_status.txt' ERR
echo preflight > queue_status.txt
python -u source/run_unseen_source.py --config config.json --preflight > preflight.log 2>&1
echo training > queue_status.txt
python -u source/run_unseen_source.py --config config.json > train.log 2>&1
echo complete > queue_status.txt
'''
        (out / 'launch.sh').write_text(script)
    write(root / 'protocol.json', dict(status='prepared_not_launched',
          source_subjects=SOURCES, excluded_subjects=EXCLUDED,
          arms=['base', 'identity'], paired_difference='pooled512 identity contrast weight0/1, temperature.1',
          calibration='after source fitting: independent seen, novel01 and novel02 residual calibrators; test EEG never fitted',
          test='no S1/S2 EEG or metrics accessed by source fitting/selection',
          pretrained='original REVE checkpoint; downstream exclusion only',
          source_parent=str(source_parent), source_parent_hashes=str(source_parent.parent / 'source_sha256.json')))
    calibration = dict(
        seeds=list(range(33, 38)), lr=3e-5, weight_decay=.01,
        epochs=20, patience=5, batch_size=256, gradient_clip=1.,
        selection_space='blur', selection_metric='development correct-image mean rank (minimize)',
        selection_ties='earliest epoch; epoch0 identity map included',
        test_blur_shards=str(Path(original['test_blur_shards']).resolve()),
        architecture='e + Linear(64,512)(GELU(Linear(512,64)(LayerNorm(e))))',
        trainable_parameters=67136, initialization='last linear weight and bias zero',
        frozen='REVE, original blur projection, B recovery, image adapter; frozen heads always eval',
        loss='one-direction CE clean + CE native adapter blur, equal weights; image-only L2 normalization; temperature1',
        augmentation='none during calibration; one real trial per image',
        source_route='calibrator bypassed during ordinary source training',
        inference='pointwise; no target support, target identity, target labels or test-time updates',
        fitting_images=14886, development_images=1654, original_test_images=200,
        aggregation='all valid trials of each subject/image; mean raw predictions after inference',
        primary_comparison='identity versus base within each pool, seed and target subject',
        directions=dict(novel01='fit S1, evaluate S2', novel02='fit S2, evaluate S1',
                        seen='fit one source-person trial per image from S3--S10, evaluate S1 and S2'),
        interpretation='bidirectional development pilot; not a full episodic meta-learning reproduction')
    write(root / 'calibration_protocol.json', calibration)
    cfg = json.loads((root / 'base/config.json').read_text())
    records = plan_records(cfg)
    assert records['fit_images'] == 14886 and records['dev_images'] == 1654
    write(root / 'calibration_records.json', records)
    protected = ['protocol.json', 'calibration_protocol.json', 'calibration_records.json',
                 'validation_ids.json', 'source_sha256.json', 'base/config.json', 'identity/config.json',
                 'shared_predictability/subjects.json', 'shared_predictability/complete.json',
                 'base/local_complete.json', 'identity/local_complete.json']
    if args.reference_report:
        protected.append('reference.md')
    write(root / 'input_sha256.json', {name: digest(root / name) for name in protected})
    for name in protected + ['input_sha256.json']:
        path = root / name
        if not path.is_symlink():
            path.chmod(0o444)
    print(root)


if __name__ == '__main__':
    main()
