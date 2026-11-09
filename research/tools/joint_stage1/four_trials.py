"""Exactly four distinct, same-person/same-image training trials per input."""
import numpy as np
import torch
from augment_trials import MixedTrials


class FourTrials(MixedTrials):
    def __init__(self, ds):
        super().__init__(ds)
        self.valid = [i for i, key in enumerate(self.keys) if len(self.groups[key]) >= 4]
        self.audit = dict(total_anchors=len(self.keys), eligible_anchors=len(self.valid),
                          excluded_anchors=len(self.keys)-len(self.valid),
                          excluded_groups=sum(len(g)<4 for g in self.groups.values()),
                          repeats=4, replacement=False)
        assert self.valid

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, index):
        i = self.valid[index]
        others = [j for j in self.groups[self.keys[i]] if j != i]
        chosen = [i]+np.random.choice(others, 3, replace=False).tolist()
        assert len(set(chosen)) == 4 and all(self.keys[j] == self.keys[i] for j in chosen)
        b = self.ds[i]
        b['eeg'] = torch.stack([self.ds[j]['eeg'] for j in chosen]).mean(0)
        b['repeat_count'] = 4
        return b
