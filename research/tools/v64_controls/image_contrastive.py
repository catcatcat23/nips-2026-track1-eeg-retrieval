"""Image-supervised EEG contrast: same image positive across all subjects.

Reuse v82's independent 32-image x 4-subject x 2-trial auxiliary batch.
Only the REVE pooled512 representation receives this auxiliary gradient.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from grouped_identity import GroupedIdentity, pooled


def masks(images):
    if images.ndim != 1:
        raise ValueError("Image IDs must be a one-dimensional tensor")
    allowed = ~torch.eye(len(images), device=images.device, dtype=torch.bool)
    positive = (images[:, None] == images[None, :]) & allowed
    negative = (images[:, None] != images[None, :]) & allowed
    return allowed, positive, negative


def image_loss(features, images, temperature=0.1):
    """Mean log probability of ALL same-image positives; exclude self."""
    if features.ndim != 2 or len(features) != len(images):
        raise ValueError("Expected one image ID per feature row")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    allowed, positive, negative = masks(images)
    valid = positive.any(1) & negative.any(1)
    if not valid.all():
        raise ValueError("Every anchor needs both same-image positives and negatives")
    f = F.normalize(features.float(), dim=-1)
    logits = f @ f.T / temperature
    logp = logits - logits.masked_fill(~allowed, -torch.inf).logsumexp(1, keepdim=True)
    return -(logp.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()


def check_batch(images, subjects):
    allowed, positive, negative = masks(images)
    same_subject = subjects[:, None] == subjects[None, :]
    assert len(images) == 256 and len(images.unique()) == 32
    assert positive.sum(1).eq(7).all()
    assert negative.sum(1).eq(248).all()
    assert allowed.sum(1).eq(255).all()
    assert (positive & same_subject).sum(1).eq(1).all()
    assert (positive & ~same_subject).sum(1).eq(6).all()
    for image in images.unique():
        _, counts = subjects[images == image].unique(return_counts=True)
        assert len(counts) == 4 and counts.eq(2).all()


SAMPLER = None


def reset_sampler():
    global SAMPLER
    SAMPLER = None


def auxiliary_backward(model, cfg):
    global SAMPLER
    if SAMPLER is None:
        assert cfg['auxiliary_seed'] == 810033, 'Keep v82 auxiliary sampling unchanged'
        SAMPLER = GroupedIdentity(cfg)
    x, images, subjects = SAMPLER.sample()
    check_batch(images, subjects)
    device = next(model.encoder.parameters()).device
    loss = image_loss(pooled(model, x.to(device)), images.to(device),
                      cfg['image_contrastive_temperature'])
    assert torch.isfinite(loss)
    (cfg['image_contrastive_weight'] * loss).backward()
    return loss.detach()


def write_report(out, name, report):
    path = Path(out) / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def cpu_preflight(cfg, out):
    torch.set_num_threads(4)
    torch.manual_seed(33)
    sampler = GroupedIdentity(cfg)
    first = None
    for _ in range(5):
        x, images, subjects = sampler.sample()
        check_batch(images, subjects)
        assert torch.isfinite(x).all()
        if first is None:
            first = (x, images, subjects)
        f = torch.randn(256, 32, requires_grad=True)
        loss = image_loss(f, images, cfg['image_contrastive_temperature'])
        loss.backward()
        assert torch.isfinite(f.grad).all() and f.grad.abs().sum() > 0
        perm = torch.randperm(len(images))
        torch.testing.assert_close(loss.detach(), image_loss(f.detach()[perm], images[perm]))

    # This control uses exactly the original sampler sequence and real EEG rows.
    replay = GroupedIdentity(cfg).sample()
    for actual, expected in zip(replay, first):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    x, images, subjects = first

    # Independent dense-target CE verifies the multi-positive averaging formula.
    f = F.normalize(torch.randn(256, 32), dim=-1)
    allowed, positive, negative = masks(images)
    logits = f @ f.T / 0.1
    probabilities = positive.float() / positive.sum(1, keepdim=True)
    reference = F.cross_entropy(logits.masked_fill(~allowed, -1e9), probabilities)
    torch.testing.assert_close(image_loss(f, images), reference)

    # A representation containing only image identity should beat subject identity.
    _, labels = images.unique(return_inverse=True)
    picture_vectors = F.one_hot(labels, num_classes=32).float()
    subject_vectors = F.one_hot(subjects).float()
    good = image_loss(picture_vectors, images)
    bad = image_loss(subject_vectors, images)
    assert good < bad
    # For seven positives the optimum approaches log(7), not zero.
    expected = math.log(7 + 248 * math.exp(-10))
    assert abs(float(good) - expected) < 1e-5

    # No-positive and no-negative batches fail explicitly instead of silently training.
    for invalid_ids in (torch.arange(256), torch.zeros(256, dtype=torch.long)):
        try:
            image_loss(f, invalid_ids)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid contrastive batch was accepted')

    test_ids = np.load(Path(cfg['source_root']) / 'data/test_ids.npy')
    assert not np.intersect1d(sampler.images, test_ids).size
    report = dict(passed=True, train_images=len(sampler.images), batches=5,
                  batch_size=256, images_per_batch=32, subjects_per_image=4,
                  trials_per_subject=2, positives=7, same_subject_positives=1,
                  cross_subject_positives=6, negatives=248, test_overlap=0,
                  self_excluded=True, sampler_matches_v82=True,
                  image_only_feature_loss=float(good), subject_only_feature_loss=float(bad),
                  loss_lower_bound=math.log(7), dense_target_cross_entropy_parity=True,
                  first_batch_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
    write_report(out, 'image_contrastive_cpu_check.json', report)
    print('IMAGE_CONTRASTIVE_CPU_PASSED', json.dumps(report), flush=True)


def state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gpu_preflight(cfg, out):
    import extensions as ex
    ex.CFG.update(cfg)
    torch.set_num_threads(4)
    torch.manual_seed(33)
    np.random.seed(33)
    manifest = json.loads((Path(cfg['source_root']) / 'data/manifest.json').read_text())
    model = ex.AugmentedModel(manifest).cuda()
    initial_hash = state_sha256(model)
    # v82's Identity module adds no parameters and must not change initialization.
    torch.manual_seed(33)
    ex.CFG['identity_location'] = 'pooled512'
    try:
        reference = ex.IdentityModel(manifest)
    finally:
        ex.CFG.pop('identity_location')
    assert state_sha256(reference) == initial_hash
    del reference

    model.perturbation_kind = cfg['augmentation']
    model.train()
    reset_sampler()
    loss = auxiliary_backward(model, cfg)
    gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    norm = float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt())
    assert norm > 0
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith('encoder.'))
    model.zero_grad(set_to_none=True)
    del gradients

    # Four real images; each has four subjects and two distinct repeats (32 rows).
    x, images, _ = GroupedIdentity(cfg).sample()
    x, images = x[:32].cuda(), images[:32].cuda()
    assert len(images.unique()) == 4  # sampler stores eight trials per image
    model.eval()
    opt = torch.optim.AdamW(model.encoder.parameters(), lr=1e-4)
    with torch.no_grad():
        initial = float(image_loss(pooled(model, x), images))
    # Avoid an impossible 'loss close to zero' criterion with seven positives.
    floor = math.log(7)
    final, hit = initial, 0.0
    for step in range(80):
        opt.zero_grad(set_to_none=True)
        value = image_loss(pooled(model, x), images)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            features = F.normalize(pooled(model, x), dim=-1)
            scores = (features @ features.T).masked_fill(torch.eye(len(x), device=x.device, dtype=torch.bool), -torch.inf)
            hit = float((images[scores.argmax(1)] == images).float().mean())
            final = float(image_loss(features, images))
        if step >= 2 and hit == 1 and final < floor + 0.5 * (initial - floor):
            break
    assert hit == 1 and final < floor + 0.5 * (initial - floor), 'Real EEG auxiliary tiny-fit failed'
    # Serialization parity on the freshly trained disposable smoke model.
    from io import BytesIO
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    with torch.no_grad():
        before = model.encode_eeg(x[:4])[:2]
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, map_location='cuda', weights_only=True), strict=True)
    with torch.no_grad():
        after = model.encode_eeg(x[:4])[:2]
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    reset_sampler()
    report = dict(passed=True, device=torch.cuda.get_device_name(),
                  initialization_matches_v82=True, initial_state_sha256=initial_hash,
                  full_batch_loss=float(loss), encoder_auxiliary_gradient_l2=norm,
                  auxiliary_updates_encoder_only=True, tiny_images=4, tiny_trials=32,
                  initial_loss=initial, final_loss=final, loss_lower_bound=floor,
                  tiny_eeg_same_image_top1=hit, steps=step + 1,
                  save_reload_parity=True, updates_discarded=True)
    write_report(out, 'image_contrastive_gpu_check.json', report)
    print('IMAGE_CONTRASTIVE_GPU_PASSED', json.dumps(report), flush=True)
