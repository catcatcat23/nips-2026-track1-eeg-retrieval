# Copyright Meta Platforms, Inc. and affiliates. See licenses/neuraltrain_LICENSE.
import torch
from torch import nn
from torch.nn import functional as F

class ClipLoss(nn.Module):
    """CLIP constrastive loss.

    Contrastive Language-Image Pretraining (CLIP) loss from [1]_. Default values reflect the
    configuration of the CLIP loss used in [2]_.

    Parameters
    ----------
    norm_kind : {"x", "y", "xy"} or None
        How to normalize the estimates and/or candidates before computing their dot products.
            ``'x'``: normalize estimates only.
            ``'y'``: normalize candidates only (approach originally used in brainmagick).
            ``'xy'``: normalize both estimates and candidates.
            ``None``: do not normalize.
    temperature : bool
        If True, use a learnable temperature parameter.
    symmetric : bool
        If True, compute loss in both retrieval directions, i.e. retrieve candidates given
        estimates and retrieve estimates given candidates (requires estimates and candidates to be
        of the same shape). If False, only do the former.
    reduction : str
        Reduction applied to the per-example cross-entropy loss (forwarded to
        ``F.cross_entropy``).

    References
    ----------
    .. [1] Radford, Alec, et al. "Learning transferable visual models from natural language
        supervision." International conference on machine learning. PMLR, 2021.
    .. [2] Défossez, Alexandre, et al. "Decoding speech perception from non-invasive brain
        recordings." Nature Machine Intelligence (2023): 1-11.
    """

    def __init__(
        self,
        norm_kind: str | None = "y",
        temperature: bool = True,
        symmetric: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.norm_kind = norm_kind
        self.temperature = (
            nn.Parameter(torch.tensor(1 / 0.07).log())
            if temperature
            else nn.Parameter(torch.tensor(0.0), requires_grad=False)
        )
        self.symmetric = symmetric
        self.reduction = reduction

    @staticmethod
    def _compute_similarity(
        x: torch.Tensor, y: torch.Tensor, norm: str | None = None, eps=1e-15
    ) -> torch.Tensor:
        if norm is None:
            eq, inv_norms = "b", torch.ones(x.shape[0])
        elif norm == "x":
            eq, inv_norms = "b", 1 / (eps + x.norm(dim=(1), p=2))
        elif norm == "y":
            eq, inv_norms = "o", 1 / (eps + y.norm(dim=(1), p=2))
        elif norm == "xy":
            eq = "bo"
            inv_norms = 1 / (
                eps + torch.outer(x.norm(dim=(1), p=2), y.norm(dim=(1), p=2))
            )
        else:
            raise ValueError(f"norm must be None, x, y or xy, got {norm}.")

        # Normalize inside einsum to avoid creating a copy of candidates which can be pretty big
        return torch.einsum(f"bc,oc,{eq}->bo", x, y, inv_norms)

    def get_scores(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Given estimates of shape [B, F] and candidates of shape [B', F], return a [B, B'] matrix
        of similarity scores.
        """
        scores = self._compute_similarity(estimate, candidate, norm=self.norm_kind)
        scores = self.temperature.exp() * scores
        return scores

    def get_probabilities(
        self, estimate: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        """Given estimates of shape [B, F] and candidates of shape [B', F], return a [B, B'] matrix
        of matching probability.
        """
        scores = self.get_scores(estimate, candidate)
        return F.softmax(scores, dim=1)

    def forward(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Warning: estimate and candidate are not necessarily symmetrical.

        If estimate of shape [B, C] and candidate of shape [B', C] with B'>=B, the first B samples
        of candidate are targets, while the remaining B'-B samples of candidate are only used as
        negatives.
        """
        n_est = estimate.size(0)
        n_cand = candidate.size(0)
        if n_est > n_cand:
            raise ValueError(f"need candidates >= estimates, got {n_cand} vs {n_est}")
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)

        loss_e = F.cross_entropy(scores, target, reduction=self.reduction)
        if self.symmetric:
            if scores.shape[0] != scores.shape[1]:
                raise ValueError(f"need square scores, got {scores.shape}")
            loss_c = F.cross_entropy(
                scores.transpose(1, 0), target, reduction=self.reduction
            )
            loss = (loss_e + loss_c) / 2
        else:
            loss = loss_e
        return loss
