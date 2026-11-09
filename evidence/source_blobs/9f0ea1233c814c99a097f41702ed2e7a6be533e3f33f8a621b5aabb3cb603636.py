"""Same-image averaging with one training trial per distinct subject."""
import numpy as np
import torch
from torch.utils.data import Dataset


class CrossSubjectTrials(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.ids = np.concatenate([d.ids for d in ds.datasets])
        self.subjects = np.concatenate([d.subjects.astype(str) for d in ds.datasets])
        self.groups = {}
        for i, (image, subject) in enumerate(zip(self.ids, self.subjects)):
            self.groups.setdefault(int(image), {}).setdefault(str(subject), []).append(i)

    def __len__(self):
        return len(self.ds)

    def members(self, i, requested):
        group = self.groups[int(self.ids[i])]
        others = [s for s in group if s != self.subjects[i]]
        n = min(requested, len(others)+1)
        selected = np.random.choice(others, n-1, replace=False)
        members = [i] + [int(np.random.choice(group[s])) for s in selected]
        assert len(set(self.subjects[members])) == n
        assert np.all(self.ids[members] == self.ids[i])
        return members

    def __getitem__(self, i):
        r = np.random.random()
        requested = 1 if r < .5 else (2 if r < .75 else 4)
        members = self.members(i, requested)
        b = dict(self.ds[i])
        if len(members) > 1:
            b['eeg'] = torch.stack([self.ds[j]['eeg'] for j in members]).mean(0)
        b['repeat_count'] = len(members)
        return b
