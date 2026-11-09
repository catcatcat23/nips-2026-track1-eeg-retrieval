"""Run the v82-matched image-positive control from an immutable snapshot."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def verify_snapshot(out):
    files = json.loads((out / 'source_sha256.json').read_text())
    for relative, expected in files.items():
        actual = hashlib.sha256((out / 'source' / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Source snapshot hash mismatch: ' + relative)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--preflight', choices=['cpu', 'gpu'])
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['run_root'])
    verify_snapshot(out)
    provenance = json.loads((out / 'provenance.json').read_text())
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == provenance['config_sha256']
    assert not cfg.get('heldout_subject'), 'This sampler is for the v82 full-subject control, not LOSO'
    assert cfg['identity_weight'] == 0 and cfg.get('picture_weight', 0) == 0
    assert cfg['image_contrastive_weight'] == 1.0
    assert cfg['image_contrastive_temperature'] == 0.1
    assert cfg['auxiliary_seed'] == 810033
    assert cfg['batch_size'] == cfg['auxiliary_batch_size'] == 256
    assert cfg['initialization'] == 'fresh pretrained REVE, no trained retrieval checkpoint loaded'
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import image_contrastive as contrast
    if args.preflight == 'cpu':
        contrast.cpu_preflight(cfg, out)
        return
    import extensions as ex
    import local_ot_train_ema as base
    ex.CFG.update(cfg)
    ex.audit(cfg, out)
    # Set the SAME retrieval-only objective for both preflight and formal training.
    ex.loss_fn = ex.original_loss
    if args.preflight == 'gpu':
        assert json.loads((out / 'image_contrastive_cpu_check.json').read_text())['passed']
        ex.preflight(cfg, out, ex.AugmentedModel)
        contrast.gpu_preflight(cfg, out)
        return
    for name in ['preflight.json', 'image_contrastive_cpu_check.json', 'image_contrastive_gpu_check.json']:
        assert json.loads((out / name).read_text())['passed'], name
    # A new model is initialized inside train(); all preflight updates are discarded.
    contrast.reset_sampler()
    base.LocalOTModel = ex.AugmentedModel
    base.MixedTrials = ex.Trials
    base.loss_fn = ex.original_loss
    base.evaluate = ex.evaluate
    base.train(cfg, out)


if __name__ == '__main__':
    main()
