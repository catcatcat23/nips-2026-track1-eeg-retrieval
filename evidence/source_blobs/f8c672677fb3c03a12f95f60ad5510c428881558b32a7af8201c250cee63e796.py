"""Always mix four distinct subjects for the same training image."""
import torch
from cross_subject_mix import CrossSubjectTrials


class FixedCrossSubjectFour(CrossSubjectTrials):
    def __init__(self, ds, anchor_weight=.25):
        super().__init__(ds)
        assert anchor_weight in [.25, .7]
        assert min(len(g) for g in self.groups.values()) >= 4
        self.weights = [.25]*4 if anchor_weight == .25 else [.7,.1,.1,.1]
        self.audit = dict(anchors=len(self), probabilities=[0.,0.,1.],
                          weights=self.weights, distinct_subjects=4,
                          scope='same training image, within dataset; anchor first')

    def __getitem__(self, i):
        members = self.members(i, 4)
        assert len(members) == 4 and members[0] == i
        b = dict(self.ds[i])
        x = torch.stack([self.ds[j]['eeg'] for j in members])
        b['eeg'] = (x*x.new_tensor(self.weights).reshape((-1,)+(1,)*(x.ndim-1))).sum(0)
        b['repeat_count'] = 4
        return b
