"""Training-only, same-subject/image averaging with explicit epoch seeds."""
import numpy as np
import torch
from torch.utils.data import Dataset


class AverageTrials(Dataset):
    def __init__(self, base, train, ids, subjects):
        self.base = base
        train = np.asarray(train, dtype=np.int64)
        _, sid = np.unique(subjects[train], return_inverse=True)
        order = np.lexsort((ids[train], sid))
        keys = np.stack((sid[order], ids[train][order]), axis=1)
        edges = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1, len(train)]
        self.groups = np.split(train[order], edges[1:-1])
        self.group_of = np.full(len(base), -1, dtype=np.int32)
        for gi, group in enumerate(self.groups):
            self.group_of[group] = gi
        sizes = np.array([len(g) for g in self.groups])
        self.audit = {'groups':len(sizes), 'training_trials':len(train),
                      'group_size_histogram':{str(k):int((sizes == k).sum()) for k in np.unique(sizes)},
                      'fallback':'min(requested, available); distinct trials, always includes anchor',
                      'averaging_stage':'official cached EEG before REVE interpolation/zscore',
                      'sampling':'per-anchor/epoch deterministic seed; same subject and image, training only'}

    def __len__(self):
        return len(self.base)

    def members(self, index, requested, epoch):
        gi = self.group_of[index]
        assert gi >= 0, 'Non-training anchor'
        group = self.groups[gi]
        count = min(requested, len(group))
        rng = np.random.default_rng(np.random.SeedSequence([33, int(epoch), int(index), 3701]))
        extra = rng.choice(group[group != index], count - 1, replace=False)
        return np.r_[index, extra]

    def __getitem__(self, task):
        index, requested, epoch = task
        members = self.members(index, requested, epoch)
        sample = dict(self.base[int(index)])
        if len(members) > 1:
            sample['eeg'] = torch.stack([sample['eeg']] + [self.base[int(i)]['eeg'] for i in members[1:]]).mean(0)
        sample['average_requested'] = requested
        sample['average_actual'] = len(members)
        return sample


class AverageBatches:
    def __init__(self, base):
        self.base = base

    @property
    def epoch(self):
        return self.base.epoch

    @epoch.setter
    def epoch(self, value):
        self.base.epoch = value

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([33, self.epoch, 3702]))
        for indices in self.base:
            n = len(indices)
            counts = np.array([1] * (n - 2 * (n // 4)) + [2] * (n // 4) + [4] * (n // 4))
            rng.shuffle(counts)
            yield [(int(i), int(k), self.epoch) for i, k in zip(indices, counts)]
