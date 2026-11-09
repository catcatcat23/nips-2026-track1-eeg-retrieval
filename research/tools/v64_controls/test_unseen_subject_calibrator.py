"""CPU synthetic acceptance checks; no EEG/checkpoint files or GPU jobs needed.

Run: PYTHONDONTWRITEBYTECODE=1 python -m tools.v64_controls.test_unseen_subject_calibrator
Checkpoint round trips use BytesIO in RAM, never a system-disk artifact.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tools.v64_controls.unseen_subject_calibrator import (
    BProjectedHead, CalibratedReadout, ResidualCalibrator512,
)


def _original_b_class():
    source = Path(__file__).resolve().parents[1] / "eeg_multiblur/recovery_ablation.py"
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
             and n.name in {"ResidualBlock", "ProjectedHead"}]
    assert len(nodes) == 2
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["ProjectedHead"], hashlib.sha256(source.read_bytes()).hexdigest()


def _gradient_norms(module):
    result = {}
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        result[name] = float(parameter.grad.norm())
    return result


def _frozen_snapshot(model):
    return {name: value.clone() for name, value in model.state_dict().items()
            if not name.startswith("calibrator.")}


def _assert_frozen(model, before):
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name]), name
    for module in (model.blur_head, model.recovery, model.image_adapter):
        assert not module.training
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())


def check_real_b_architecture():
    original_class, source_hash = _original_b_class()
    torch.manual_seed(33)
    original = original_class().eval()
    projection = nn.Linear(512, 1536).eval()
    image = nn.Sequential(nn.Linear(1536, 768), nn.ELU(), nn.Dropout(.85), nn.Linear(768, 1536)).eval()
    state = {}
    for prefix, module in (("blur_head", projection), ("recovery", original), ("image_adapter", image)):
        state.update({f"{prefix}.{name}": value for name, value in module.state_dict().items()})
    model = CalibratedReadout.from_checkpoint_model_state(state, image_adapter_dropout=.85)
    assert isinstance(model.recovery, BProjectedHead)
    before = _frozen_snapshot(model)
    x = torch.randn(4, 512)
    with torch.no_grad():
        h0 = projection(x)
        z0 = original(h0)
        result = model(x)
        torch.testing.assert_close(result["pooled"], x, rtol=0, atol=0)
        torch.testing.assert_close(result["blur"], h0, rtol=0, atol=0)
        torch.testing.assert_close(result["clean"], z0, rtol=0, atol=0)
        raw_images = torch.randn(4, 1536)
        torch.testing.assert_close(model.encode_images(raw_images), F.normalize(image(raw_images), dim=-1), rtol=0, atol=0)
    model.train()
    assert model.calibrator.training
    target = F.normalize(torch.randn(4, 1536), dim=-1)
    optimizer = torch.optim.AdamW(model.calibrator.parameters(), lr=1e-3, weight_decay=0)
    gradients = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        # Clean-only loss must propagate through BOTH frozen original heads.
        loss = F.cross_entropy(model(x)["clean"] @ target.T, torch.arange(4))
        loss.backward()
        grads = _gradient_norms(model.calibrator)
        assert grads["net.3.weight"] > 0
        if step == 0:
            assert grads["net.1.weight"] == 0  # Required by zero last-layer init.
        else:
            assert all(value > 0 for value in grads.values()), grads
        optimizer.step()
        gradients.append(grads)
    _assert_frozen(model, before)
    with torch.no_grad():
        bypass = model(x, enabled=False)
        torch.testing.assert_close(bypass["clean"], z0, rtol=0, atol=0)
        torch.testing.assert_close(bypass["blur"], h0, rtol=0, atol=0)
        assert bypass["pooled"] is x
        assert not torch.equal(model(x)["clean"], z0)
        model.eval()
        adapted = model(x)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = model(x[permutation])
        for key in adapted:
            singleton = torch.cat([model(row[None])[key] for row in x])
            torch.testing.assert_close(permuted[key], adapted[key][permutation], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(singleton, adapted[key], rtol=1e-5, atol=1e-5)
    # The native loader must reject a different recovery architecture.
    bad_state = dict(state)
    del bad_state["recovery.net.2.alpha"]
    try:
        CalibratedReadout.from_checkpoint_model_state(bad_state)
    except RuntimeError:
        invalid_recovery_rejected = True
    else:
        raise AssertionError("Incomplete B recovery state was silently accepted")
    return {"repo_b_source_sha256": source_hash, "zero_step_exact": True,
            "clean_only_gradients_step1_step2": gradients,
            "frozen_heads_unchanged": True, "bypass_exact_after_updates": True,
            "batch_and_order_independent_atol": 1e-5,
            "strict_recovery_state": invalid_recovery_rejected}


def _small_readout():
    projection = nn.Linear(512, 32, bias=False)
    recovery = nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        orthogonal, _ = torch.linalg.qr(torch.randn(32, 32))
        recovery.weight.copy_(orthogonal)
    image = nn.Sequential(nn.Linear(32, 32), nn.Dropout(.85))
    return CalibratedReadout(projection, recovery, image)


def check_routing_and_tiny_fit():
    torch.manual_seed(35)
    model = _small_readout().train()
    x = torch.randn(8, 512)
    raw_image = torch.randn(8, 32)
    target = model.encode_images(raw_image)
    with torch.no_grad():
        clean_target = F.normalize(model.recovery(target), dim=-1)
    labels = torch.arange(8)
    before = _frozen_snapshot(model)

    def objective():
        out = model(x)
        return (F.cross_entropy(out["blur"] @ target.T, labels)
                + F.cross_entropy(out["clean"] @ clean_target.T, labels))

    initial = float(objective().detach())
    optimizer = torch.optim.AdamW(model.calibrator.parameters(), lr=1e-2, weight_decay=0)
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.calibrator.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            out = model(x)
            blur_hits = float(((out["blur"] @ target.T).argmax(1) == labels).float().mean())
            clean_hits = float(((out["clean"] @ clean_target.T).argmax(1) == labels).float().mean())
        if step >= 2 and blur_hits == clean_hits == 1 and float(loss.detach()) < initial * .1:
            break
    final = float(objective().detach())
    assert blur_hits == clean_hits == 1 and final < initial * .1, (initial, final, blur_hits, clean_hits)
    _assert_frozen(model, before)

    # Source routing leaves C untouched while preserving gradients into an encoder.
    model.zero_grad(set_to_none=True)
    source_x = x.detach().clone().requires_grad_(True)
    model(source_x, enabled=False)["clean"].square().mean().backward()
    assert source_x.grad is not None and source_x.grad.norm() > 0
    assert all(p.grad is None for p in model.calibrator.parameters())

    model.eval()
    with torch.no_grad():
        expected = model(x)
        permutation = torch.tensor([5, 1, 7, 0, 3, 6, 2, 4])
        permuted = model(x[permutation])
        for key in expected:
            singleton = torch.cat([model(item[None])[key] for item in x])
            torch.testing.assert_close(permuted[key], expected[key][permutation], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(singleton, expected[key], rtol=1e-5, atol=1e-5)
    # No output file: round-trip in memory only.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _small_readout().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    buffer.close()
    with torch.no_grad():
        for enabled in (True, False):
            actual, prior = restored(x, enabled=enabled), model(x, enabled=enabled)
            for key in prior:
                torch.testing.assert_close(actual[key], prior[key], rtol=0, atol=0)
    _assert_frozen(model, before)
    return {"tiny_samples": 8, "tiny_steps": step + 1, "initial_loss": initial,
            "final_loss": final, "blur_top1": blur_hits, "clean_top1": clean_hits,
            "batch_and_order_independent_atol": 1e-5, "source_route_input_gradient": float(source_x.grad.norm()),
            "source_route_calibrator_gradient": None, "save_reload_exact": True,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def self_test():
    torch.set_num_threads(4)
    return {"passed": True, "device": "cpu", "real_b_structure": check_real_b_architecture(),
            "pointwise_routing_and_tiny_fit": check_routing_and_tiny_fit(),
            "real_data_or_pretrained_checkpoint_used": False,
            "scope": "Module acceptance only; not a scientific performance experiment"}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))
