"""CPU provenance, cache reuse and real frozen-head acceptance checks.

Writes only to a new --output directory on the experiment data disk. Historical
checkpoint/features are read-only and serve solely as pipeline smoke inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.v64_controls import train_unseen_calibration as runner
from tools.v64_controls.unseen_subject_calibrator import CalibratedReadout


def rejected(call, exception=AssertionError):
    try:
        call()
    except exception:
        return True
    raise AssertionError('Unsafe reuse was accepted')


def check_cache_and_trials(root):
    data = root / 'source_data/data'
    data.mkdir(parents=True)
    arm = root / 'arm'
    arm.mkdir()
    source = root / 'source'
    source.mkdir()
    runner_path = Path(runner.__file__).resolve()
    (source / runner_path.name).symlink_to(runner_path)
    runner.write(root / 'source_sha256.json', {runner_path.name: runner.digest(runner_path)})
    runner.write(data / 'manifest.json', {'synthetic_fixture': True})
    runner.write(root / 'validation_ids.json', [1])
    for split in ['train', 'val', 'test']:
        np.save(data / f'{split}_ids.npy', np.array([0, 1]))
        np.save(data / f'{split}_subjects.npy', np.array(['person1', 'person2']))
        np.save(data / f'{split}_eeg.npy', np.ones((2, 2, 3), dtype='float32'))
    cfg = dict(source_root=str(data.parent), source_snapshot=str(source),
               source_hashes_path=str(root / 'source_sha256.json'),
               validation_ids_path=str(root / 'validation_ids.json'))
    runner.write(arm / 'config.json', cfg)
    plan = {'arrays': {'seen_fit': [{'image': 0}, {'image': 1}]}}
    provenance = {'sha256': 'fixture_checkpoint'}
    cache = arm / 'calibration_cache'
    cache.mkdir()
    origin = runner.cache_origin(cfg, arm, plan, provenance)
    fingerprint = runner.object_digest(origin)
    marker = dict(fingerprint=fingerprint, origin=origin)
    runner.write(cache / 'origin.json', marker)
    array_path = cache / 'seen_fit.npy'
    np.save(array_path, np.ones((2, 512), dtype='float32'))
    entry = dict(file=array_path.name, rows=2, sha256=runner.digest(array_path), origin_fingerprint=fingerprint)
    runner.write(cache / 'seen_fit.json', entry)
    complete = dict(**marker, arrays={'seen_fit': entry})
    runner.write(cache / 'complete.json', complete)
    # A complete, matching cache avoids loading a model or touching CUDA.
    assert runner.extract(cfg, arm, root, plan, {}, provenance) == cache
    results = {'matching_complete_cache_reused_without_gpu': True}
    results['changed_checkpoint_rejected'] = rejected(
        lambda: runner.extract(cfg, arm, root, plan, {}, {'sha256': 'different'}))
    runner.write(arm / 'config.json', dict(cfg, edited=True))
    results['changed_source_config_rejected'] = rejected(
        lambda: runner.extract(cfg, arm, root, plan, {}, provenance))
    runner.write(arm / 'config.json', cfg)
    np.save(data / 'train_ids.npy', np.array([3, 4]))
    results['changed_data_metadata_rejected'] = rejected(
        lambda: runner.extract(cfg, arm, root, plan, {}, provenance))
    np.save(data / 'train_ids.npy', np.array([0, 1]))
    bad_entry = dict(entry, origin_fingerprint='different')
    results['partial_array_source_mismatch_rejected'] = rejected(
        lambda: runner.validate_cached_array(cache, 'seen_fit', plan['arrays']['seen_fit'], bad_entry, fingerprint))
    np.save(array_path, np.zeros((2, 512), dtype='float32'))
    results['corrupt_completed_array_rejected'] = rejected(
        lambda: runner.extract(cfg, arm, root, plan, {}, provenance))

    dest = root / 'trial'
    dest.mkdir()
    trial_origin = {'calibration_fingerprint': 'fixture', 'pool': 'seen', 'seed': 33}
    assert runner.trial_reusable(dest, trial_origin) is False
    trial_fp = runner.object_digest(trial_origin)
    runner.write(dest / 'origin.json', dict(fingerprint=trial_fp, origin=trial_origin))
    results['incomplete_trial_refused'] = rejected(
        lambda: runner.trial_reusable(dest, trial_origin), FileExistsError)
    for name in ['best.pt', 'last.pt', 'history.json']:
        (dest / name).write_bytes(b'small synthetic integrity fixture')
    result = dict(state='complete', fingerprint=trial_fp,
                  artifacts_sha256={name: runner.digest(dest / name) for name in ['best.pt', 'last.pt', 'history.json']})
    runner.write(dest / 'result.json', result)
    assert runner.trial_reusable(dest, trial_origin)
    results['matching_complete_trial_reused'] = True
    results['changed_trial_protocol_rejected'] = rejected(
        lambda: runner.trial_reusable(dest, dict(trial_origin, seed=34)))
    (dest / 'best.pt').write_bytes(b'changed')
    results['changed_trial_checkpoint_rejected'] = rejected(
        lambda: runner.trial_reusable(dest, trial_origin))
    return results


def check_real_heads():
    previous = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/identity_reference_loso_pilot/fold_01_baseline')
    protocol = json.loads((previous / 'protocol.json').read_text())
    config = json.loads((previous / 'config.json').read_text())
    features = torch.load(previous / 'features.pt', map_location='cpu', weights_only=False, mmap=True)
    checkpoint = torch.load(protocol['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    model = CalibratedReadout.from_checkpoint_model_state(checkpoint['model'])
    images = features['images'][:8].tolist()
    source = Path(config['source_root'])
    clean = np.load(source / 'data/clean.npy', mmap_mode='r')
    raw_blur = np.load(source / 'features/blur.npy', mmap_mode='r')
    banks = {name: torch.zeros(len(clean), 1536) for name in ['clean', 'blur']}
    banks['clean'][images] = F.normalize(torch.tensor(clean[images]), dim=-1)
    banks['blur'][images] = model.encode_images(torch.tensor(raw_blur[images]))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    result = runner.calibration_preflight(model, features['etrain'][:8], images, banks,
                                          dict(lr=1e-4, weight_decay=.01))
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    result.update(data='Old pilot etrain[:8], real single trials; smoke check only, not v93 scientific results',
                  checkpoint=runner.file_origin(protocol['checkpoint']),
                  features=runner.file_origin(previous / 'features.pt'), images=images,
                  rng_and_model_state_restored=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to(Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments'))
    root.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    result = dict(passed=True, cache_and_results=check_cache_and_trials(root), real_head_smoke=check_real_heads(),
                  runner_sha256=runner.digest(runner.__file__), test_sha256=runner.digest(__file__))
    runner.write(root / 'acceptance.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
