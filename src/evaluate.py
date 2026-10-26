"""Official retrieval semantics; chunk queries only to bound score-matrix RAM."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import DatasetEEG
# Vendored copy of neural-interfaces26/2026-competition benchmark_utils/metrics.py.
from official_metrics import group_means, topk_accuracy


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def retrieval_metrics(pred, targets, subjects):
    candidates, idx = np.unique(targets, axis=0, return_inverse=True)
    idx = idx.reshape(-1)
    gallery = normalize(candidates)
    def score(x, labels):
        hits = np.zeros(2, dtype=np.float64)
        for start in range(0, len(x), 256):
            stop = min(start + 256, len(x))
            scores = normalize(x[start:stop]) @ gallery.T
            for j, k in enumerate((1, 5)):
                hits[j] += topk_accuracy(scores, labels[start:stop], k=k) * (stop-start)
        return dict(top1=hits[0]/len(x), top5=hits[1]/len(x), count=len(x))
    _, subject_idx = np.unique(subjects, return_inverse=True)
    keys, means = group_means(pred, np.stack([subject_idx, idx], axis=1))
    return {'single_trial': score(pred, idx),
            'subject_aggregated': score(means, keys[:, 1]),
            'candidate_images': len(candidates)}


@torch.no_grad()
def evaluate(model, root, split, device):
    ds=DatasetEEG(root,split,aux=False);model.eval();pred=[]
    for b in DataLoader(ds,batch_size=256,num_workers=2):
        # Same FP32 forward used for submission inference.
        z=model(b['eeg'].to(device))
        pred.append(z.float().cpu().numpy())
    z=np.concatenate(pred)
    return retrieval_metrics(z, np.asarray(ds.clean[ds.ids]), ds.subjects)
