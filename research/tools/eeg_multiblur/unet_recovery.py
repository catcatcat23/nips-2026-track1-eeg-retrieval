"""Deterministic vector U-Net; frozen v33 features, unchanged clean loss.

This is an architecture ablation, not a diffusion model or a spatial CNN.
No image features are required by forward().
"""
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import no_ridge_recovery as engine
from neuraltrain.losses.losses import ClipLoss


def transition(input_dim, output_dim, dropout):
    return nn.Sequential(
        nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim),
        nn.GELU(), nn.Dropout(dropout),
    )


class FeatureUNet(nn.Module):
    def __init__(self, widths=(1536, 1024, 512, 256), dropout=0.1):
        super().__init__()
        self.encoder = nn.ModuleList([
            transition(a, b, dropout) for a, b in zip(widths[:-1], widths[1:])
        ])
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(widths[-1]), nn.Linear(widths[-1], widths[-1]),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.decoder = nn.ModuleList([
            transition(b, a, dropout)
            for a, b in reversed(list(zip(widths[:-1], widths[1:])))
        ])
        self.output = nn.Sequential(nn.LayerNorm(widths[0]), nn.Linear(widths[0], 1536))
        # Match v45's initial zero prediction and output scale, not its weights.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, h):
        skips = []
        x = h
        for layer in self.encoder:
            skips.append(x)
            x = layer(x)
        x = x + self.bottleneck(x)
        for layer, skip in zip(self.decoder, reversed(skips)):
            x = layer(x) + skip
        return 0.1 * self.output(x)


def make(cfg):
    torch.manual_seed(cfg['seed'])
    return FeatureUNet(tuple(cfg['widths']), cfg['dropout']).cuda()


def preflight(cfg, out, h, ids, target):
    _, indices = np.unique(ids.cpu().numpy(), return_index=True)
    indices = torch.tensor(indices[:8], device=h.device)
    x, y = h[indices], target[ids[indices]]
    model = make(cfg).eval()
    criterion = ClipLoss(norm_kind='y', temperature=False, symmetric=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    with torch.no_grad():
        assert model(x).shape == (8, 1536)
        assert torch.count_nonzero(model(x)) == 0
        initial = float(criterion(model(x), y))
    for step in range(200):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            q = model(x)
            final = float(criterion(q, y))
            acc = float(((F.normalize(q, dim=-1) @ F.normalize(y, dim=-1).T)
                         .argmax(1) == torch.arange(8, device=x.device)).float().mean())
        if step >= 4 and acc == 1 and final < initial * 0.3:
            break
    assert acc == 1 and final < initial * 0.3
    model.train()
    optimizer.zero_grad()
    criterion(model(x), y).backward()
    gradients = {}
    modules = {f'encoder_{i}': m for i, m in enumerate(model.encoder)}
    modules.update({f'decoder_{i}': m for i, m in enumerate(model.decoder)})
    modules.update(bottleneck=model.bottleneck, output=model.output)
    for name, module in modules.items():
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in module.parameters())
        gradients[name] = sum(float(p.grad.abs().sum()) for p in module.parameters())
        assert gradients[name] > 0, name
    model.eval()
    with torch.no_grad():
        q = model(x)
        torch.testing.assert_close(model(x[:1]), q[:1], rtol=1e-4, atol=1e-4)
        permutation = torch.tensor([3, 0, 7, 2, 4, 1, 6, 5], device=x.device)
        torch.testing.assert_close(model(x[permutation]), q[permutation], rtol=1e-4, atol=1e-4)
    report = dict(passed=True, parameters=sum(p.numel() for p in model.parameters()),
                  initial_loss=initial, final_loss=final, steps=step+1, tiny_top1=acc,
                  gradient_l1=gradients, batch_invariance=True, updates_discarded=True)
    engine.write(out / 'preflight.json', report)
    print('PREFLIGHT PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    engine.make = make
    engine.preflight = preflight
    engine.main()
