"""A pointwise, zero-initialised calibration route before frozen REVE heads.

Source episodes can bypass this route; the unseen-subject route enables it.
The caller supplies pooled REVE features and owns the EEG encoder/preprocessing.
No labels, subject IDs, gallery statistics or target-side optimisation are needed
by inference. This module implements the route, not an episodic training recipe.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class ResidualCalibrator512(nn.Module):
    """e + Linear64→512(GELU(Linear512→64(LayerNorm(e))))."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 512)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled: torch.Tensor, *, enabled: bool = True) -> torch.Tensor:
        if pooled.ndim != 2 or pooled.shape[-1] != 512:
            raise ValueError("Expected pooled REVE features of shape [batch,512]")
        if not isinstance(enabled, bool):
            raise TypeError("enabled is one explicit bool per route, not a subject-ID mask")
        return pooled + self.net(pooled) if enabled else pooled


class BResidualBlock(nn.Module):
    """Exact state/layout of tools/eeg_multiblur/recovery_ablation.ResidualBlock."""

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.net = nn.Sequential(
            nn.LayerNorm(1536), nn.Linear(1536, 3072), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(3072, 1536),
        )

    def forward(self, features):
        return features + self.alpha * self.net(features)


class BProjectedHead(nn.Module):
    """Exact B recovery: no additional outer ``h + recovery(h)`` residual."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(1536), nn.Linear(1536, 1536),
            BResidualBlock(), BResidualBlock(),
            nn.LayerNorm(1536), nn.Linear(1536, 1536),
        )

    def forward(self, blur):
        return self.net(blur)


def _substate(model_state, prefix):
    result = {name[len(prefix):]: value for name, value in model_state.items()
              if name.startswith(prefix)}
    if not result:
        raise ValueError(f"No weights with prefix {prefix!r} in checkpoint model state")
    return result


class CalibratedReadout(nn.Module):
    """Frozen native heads with a separately controlled trainable calibrator.

    Modules supplied to the constructor are retained and frozen in place. Use
    deepcopy before constructing if the caller needs an independent trainable
    copy. Heads stay in eval mode even when this wrapper is set to train mode,
    so their dropout/BatchNorm cannot depend on route or batch composition.
    Autograd remains enabled through the heads to the calibrator and input.
    """

    def __init__(self, blur_head: nn.Module, recovery: nn.Module,
                 image_adapter: nn.Module, calibrator: nn.Module | None = None):
        super().__init__()
        self.blur_head = blur_head
        self.recovery = recovery
        self.image_adapter = image_adapter
        if calibrator is None:
            calibrator = ResidualCalibrator512()
            reference = next(self.blur_head.parameters(), None)
            if reference is not None:
                calibrator.to(device=reference.device, dtype=reference.dtype)
        self.calibrator = calibrator
        self._freeze_heads()

    def _freeze_heads(self):
        for module in (self.blur_head, self.recovery, self.image_adapter):
            module.eval().requires_grad_(False)
            for parameter in module.parameters():
                parameter.grad = None

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_heads()
        return self

    def forward(self, pooled: torch.Tensor, *, enabled: bool = True):
        adapted = self.calibrator(pooled, enabled=enabled)
        blur = self.blur_head(adapted)
        # Do not use no_grad/detach here: clean/blur losses must reach C.
        clean = self.recovery(blur)
        return {"pooled": adapted, "blur": blur, "clean": clean}

    @torch.no_grad()
    def encode_images(self, raw_blur: torch.Tensor) -> torch.Tensor:
        """Native frozen visual targets, kept outside the trainable EEG route."""
        return F.normalize(self.image_adapter(raw_blur), dim=-1)

    def predict_clean(self, pooled: torch.Tensor, *, enabled: bool = True):
        return self(pooled, enabled=enabled)["clean"]

    @classmethod
    def from_checkpoint_model_state(cls, model_state: Mapping[str, torch.Tensor],
                                    *, image_adapter_dropout: float = 0.0):
        """Load the original blur/B-recovery/image heads strictly from ck['model'].

        Other state entries (REVE/local-OT/identity heads) are intentionally left
        to the caller. Recovery keys must match the actual B ProjectedHead; an
        old residual-MLP/ridge/token head is rejected, never silently substituted.
        Dropout p is not stored in a state_dict. It may be supplied from the run
        config; p=0 is equivalent for these permanently frozen eval-mode heads.
        """
        if not isinstance(model_state, Mapping):
            raise TypeError("Pass checkpoint['model'], not a checkpoint filename")
        projection_state = _substate(model_state, "blur_head.")
        recovery_state = _substate(model_state, "recovery.")
        image_state = _substate(model_state, "image_adapter.")
        weight = projection_state.get("weight")
        if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (1536, 512):
            raise ValueError("Expected original blur_head.weight of shape [1536,512]")
        # Loading existing heads must not perturb the new calibrator's RNG seed.
        with torch.random.fork_rng(devices=[]):
            projection = nn.Linear(512, 1536)
            recovery = BProjectedHead()
            image_adapter = nn.Sequential(
                nn.Linear(1536, 768), nn.ELU(), nn.Dropout(image_adapter_dropout),
                nn.Linear(768, 1536),
            )
        for module, state in ((projection, projection_state), (recovery, recovery_state),
                              (image_adapter, image_state)):
            module.to(device=weight.device, dtype=weight.dtype)
            module.load_state_dict(state, strict=True)
        return cls(projection, recovery, image_adapter)

