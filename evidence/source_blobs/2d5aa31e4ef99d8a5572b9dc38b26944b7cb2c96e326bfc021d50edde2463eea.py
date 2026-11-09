"""Training-only perturbations, applied after the existing EEG normalization."""
import torch
from torch import nn
from torch.nn import functional as F

class TrainAugmentation(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm

    def signal(self, x):
        if not self.training or self.arm != 'noise':
            return x
        mask = torch.rand(len(x), 1, 1, device=x.device) < .5
        return x + mask * torch.randn_like(x) * .02

    def features(self, x):
        return F.dropout(x, p=.1, training=self.training) if self.arm == 'feature_dropout' else x
