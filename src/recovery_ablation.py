import numpy as np
import torch
from torch import nn
from evaluate import normalize, group_means
from evaluate_parallel import gpu_hits

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.net = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 3072),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(3072, 1536))

    def forward(self, x):
        return x + self.alpha * self.net(x)

class ProjectedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 1536),
                                 ResidualBlock(), ResidualBlock(),
                                 nn.LayerNorm(1536), nn.Linear(1536, 1536))

    def forward(self, h):
        # Raw outputs intentionally retained for official mean-then-normalize scoring.
        return self.net(h)

def metrics(pred, ids, subjects, clean):
    unique, inverse = np.unique(ids, return_inverse=True)
    candidates, image_idx = np.unique(clean[unique], axis=0, return_inverse=True)
    labels = image_idx.reshape(-1)[inverse]
    gallery = normalize(candidates)
    _, sid = np.unique(subjects, return_inverse=True)
    keys, means = group_means(pred, np.stack([sid, labels], axis=1))
    result = {'candidate_images': len(gallery), 'per_subject': {}}
    for name, x, y, subject_indices in [('single_trial', pred, labels, sid), ('subject_aggregated', means, keys[:, 1], keys[:, 0])]:
        hits = gpu_hits(x, gallery, y, torch.device('cuda'))
        result[name] = {'top1': float(hits[:, 0].mean()), 'top5': float(hits[:, 1].mean()), 'count': len(x)}
        for i, subject in enumerate(np.unique(subjects)):
            selected = hits[subject_indices == i]
            result['per_subject'].setdefault(str(subject), {})[name] = {'top1': float(selected[:, 0].mean()), 'top5': float(selected[:, 1].mean()), 'count': len(selected)}
    return result

@torch.no_grad()
def predictions(head, x):
    head.eval()
    return np.concatenate([head(b).cpu().numpy() for b in x.split(1024)])
