"""Pretrained REVE with official probe preprocessing and blur/recovery heads."""
import torch
from torch import nn
from torch.nn import functional as F


class ReveModel(nn.Module):
    def __init__(self, manifest, cfg):
        super().__init__()
        from braindecode.models import REVE
        if manifest['input_shape'][1] != 120:
            raise ValueError('Expected competition 120-point windows')
        self.encoder = REVE.from_pretrained(
            'brain-bzh/reve-base', n_outputs=1,
            n_chans=manifest['input_shape'][0], n_times=200, sfreq=200, chs_info=None)
        names = [str(n).removeprefix('EEG ').split('-', 1)[0] for n in manifest['channels']]
        self.register_buffer('positions', self.encoder.get_positions(names))
        self.finetune = cfg.get("encoder") == "reve_finetune"
        self.encoder.requires_grad_(self.finetune)
        # Original classification head is bypassed by return_features=True.
        self.encoder.final_layer.requires_grad_(False)
        if not self.finetune:
            self.encoder.eval()
        self.microbatch = cfg.get('encoder_microbatch', 64)
        self.global_head = nn.Sequential(nn.Dropout(cfg.get('head_dropout', 0.1)), nn.Linear(512, 1536))
        self.restore = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 768), nn.GELU(), nn.Linear(768, 1536))
        nn.init.zeros_(self.restore[-1].weight)
        nn.init.zeros_(self.restore[-1].bias)
        self.clean_only = cfg.get('clean_only', False)
        if self.clean_only:
            assert not cfg.get('image_adapter', False)
            # Consume identical initialization RNG, but remove restoration parameters.
            del self.restore
        self.use_image_adapter = cfg.get('image_adapter', False)
        if self.use_image_adapter:
            # Preserve baseline EEG/head initialization and subsequent RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.image_adapter = nn.Sequential(
                    nn.Linear(1536, 768), nn.ELU(),
                    nn.Dropout(cfg.get('image_adapter_dropout', 0.0)),
                    nn.Linear(768, 1536))

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:
            self.encoder.eval()
        return self

    def features(self, eeg):
        # Explicit precision makes extraction identical in train and evaluation.
        with torch.autocast(eeg.device.type, enabled=False):
            x = F.interpolate(eeg.float(), size=200, mode='linear', align_corners=False)
            x = ((x-x.mean(-1, keepdim=True))/x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)).clamp(-15,15)
        outputs = []
        for chunk in x.split(self.microbatch):
            pos = self.positions.unsqueeze(0).expand(len(chunk), -1, -1)
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.finetune), torch.autocast(eeg.device.type, dtype=torch.bfloat16, enabled=eeg.is_cuda):
                outputs.append(self.encoder(chunk, pos=pos, return_features=True)['features'].mean((1,2)).float())
        return torch.cat(outputs)

    def recover(self, h):
        return h + self.restore(h)

    def forward(self, eeg, blur=None, local=None):
        h = self.global_head(self.features(eeg))
        if self.clean_only:
            return h
        z = self.recover(h)
        if blur is None:
            return z
        if self.use_image_adapter:
            adapted = self.image_adapter(blur)
            return h, z, self.recover(adapted), adapted
        return h, z, self.recover(blur)


def build_model(manifest, cfg):
    if cfg.get('recipe') == 'reference_94b3a31':
        from reference_model import ReferenceModel
        return ReferenceModel(manifest)
    if cfg.get('encoder') in ('reve_frozen', 'reve_finetune'):
        if cfg['local_weight']:
            raise ValueError('REVE experiment does not implement local OT')
        return ReveModel(manifest, cfg)
    from model import Model
    return Model(*manifest['input_shape'], dropout=cfg.get('dropout', (.25,.65)), use_local=bool(cfg['local_weight']))


def optimizer_groups(model, cfg):
    if cfg.get('encoder') == 'reve_finetune':
        return [
            {'params': [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')], 'lr': cfg['encoder_lr'], 'name': 'encoder'},
            {'params': [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')], 'lr': cfg['lr'], 'name': 'heads'},
        ]
    return [p for p in model.parameters() if p.requires_grad]
