"""v72 sampling unchanged; retain 70% of the anchor when mixing subjects."""
import numpy as np
import torch
from cross_subject_mix import CrossSubjectTrials


class WeightedCrossSubjectTrials(CrossSubjectTrials):
    def __init__(self, ds):
        super().__init__(ds)
        assert min(len(g) for g in self.groups.values()) >= 4
        self.audit = dict(anchors=len(self), scope='same image, distinct subjects, within dataset',
                          probabilities=[.5, .25, .25], weights={'1':[1.], '2':[.7,.3], '4':[.7,.1,.1,.1]})

    def __getitem__(self, i):
        r = np.random.random()
        requested = 1 if r < .5 else (2 if r < .75 else 4)
        members = self.members(i, requested)
        assert len(members) == requested and members[0] == i
        b = dict(self.ds[i])
        if requested > 1:
            weights = [.7,.3] if requested == 2 else [.7,.1,.1,.1]
            x = torch.stack([self.ds[j]['eeg'] for j in members])
            w = x.new_tensor(weights).reshape((-1,) + (1,)*(x.ndim-1))
            b['eeg'] = (x*w).sum(0)
        b['repeat_count'] = requested
        return b
