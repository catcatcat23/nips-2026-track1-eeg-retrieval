"""Image-balanced multi-positive version of the reference one-way ClipLoss."""
import torch
from clip_loss import ClipLoss


class ImageIdClipLoss(ClipLoss):
    """Each unique image has one candidate's mass, independent of repetitions.

    For stochastic adapter views, the image score is log-mean-exp of its
    view scores. All matching views are positives (not EEG-to-EEG pairs).
    With unique IDs this is exactly the reference y-only, temperature=1 loss.
    Call after autograd feature gather AND global ID gather when using DDP.
    """
    def __init__(self):
        super().__init__(norm_kind='y', temperature=False, symmetric=False)

    def forward(self, estimate, candidate, image_ids):
        if estimate.ndim != 2 or candidate.shape != estimate.shape:
            raise ValueError('Expected paired [B,D] estimate and candidate tensors')
        ids = image_ids.to(device=estimate.device)
        if ids.ndim != 1 or len(ids) != len(estimate) or not len(ids):
            raise ValueError('Expected one image ID per paired row, nonempty batch')
        with torch.autocast(device_type=estimate.device.type, enabled=False):
            scores = self.get_scores(estimate.float(), candidate.float())
            positive = ids[:, None] == ids[None, :]
            # Dividing each view's exp(score) by its image's multiplicity makes
            # the denominator a sum over images, not over repeated trial rows.
            scores = scores - positive.sum(0).float().log()[None, :]
            return (torch.logsumexp(scores, dim=1) - torch.logsumexp(
                scores.masked_fill(~positive, -torch.inf), dim=1)).mean()
