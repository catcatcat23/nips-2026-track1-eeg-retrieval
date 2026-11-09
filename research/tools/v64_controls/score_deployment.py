"""Label-free SCORE deployment, ported from arXiv:2608.19134v1, Eq. 3--9.

No labels, image IDs, subject IDs, model updates, or file IO enter this module.
This is a formula port of the deployment operator, not the paper's complete
recovery-aware source-training system. See SCORE_DEPLOYMENT.md for provenance
and the explicitly chosen details not specified by the public paper.
"""
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F


PAPER_URL = "https://arxiv.org/html/2608.19134v1"


@dataclass(frozen=True)
class ScoreConfig:
    # Explicit in the paper's experimental setup.
    neighbors: int = 10
    rho: float = 0.1
    min_landmarks: int = 12
    max_landmarks: int = 160
    # Numerical values absent from the public v1: fixed local conventions.
    epsilon: float = 1e-6
    margin_floor: float = 1e-6


def _validate(query: torch.Tensor, gallery: torch.Tensor, config: ScoreConfig) -> None:
    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise ValueError("Expected query [N,D] and gallery [M,D] with equal D")
    if query.shape[0] < 2 or gallery.shape[0] < 2 or query.shape[1] < 1:
        raise ValueError("At least two queries and two gallery candidates are required")
    if query.device != gallery.device:
        raise ValueError("Query and gallery must be on the same device")
    if not query.is_floating_point() or not gallery.is_floating_point():
        raise ValueError("Features must be floating-point tensors")
    if not torch.isfinite(query).all() or not torch.isfinite(gallery).all():
        raise ValueError("Features contain nonfinite values")
    if config.neighbors < 1 or not 2 <= config.min_landmarks <= config.max_landmarks:
        raise ValueError("Invalid neighbor or landmark counts")
    if config.rho < 0 or config.epsilon <= 0 or config.margin_floor <= 0:
        raise ValueError("rho must be nonnegative and numerical floors positive")


def cosine_scores(query: torch.Tensor, gallery: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Cosine similarities; normalization is local to scoring, not to Eq. 3."""
    return F.normalize(query, dim=1, eps=epsilon) @ F.normalize(gallery, dim=1, eps=epsilon).T


def csls_scores(cosine: torch.Tensor, neighbors: int = 10) -> tuple[torch.Tensor, dict]:
    """Eq. 4 on rectangular N x M scores, with independent cross-space k-NNs."""
    nq, ng = cosine.shape
    kg, kq = min(neighbors, ng), min(neighbors, nq)
    r_gallery = cosine.topk(kg, dim=1).values.mean(dim=1)
    r_query = cosine.topk(kq, dim=0).values.mean(dim=0)
    scores = 2 * cosine - r_gallery[:, None] - r_query[None, :]
    counts = torch.bincount(scores.argmax(dim=1), minlength=ng)
    return scores, {
        "k_gallery": kg,
        "k_query": kq,
        "k_clipped_for_small_input": kg != neighbors or kq != neighbors,
        "query_neighborhood_mean": float(r_gallery.mean()),
        "gallery_neighborhood_mean": float(r_query.mean()),
        "top1_gallery_coverage": int((counts > 0).sum()),
        "largest_top1_hub_count": int(counts.max()),
    }


def select_landmarks(scores: torch.Tensor, config: ScoreConfig) -> dict[str, torch.Tensor]:
    """Mutual top-1 pairs, then largest margins if more than the fixed cap.

    Mutuality is computed from the same CSLS matrix in both directions. At most
    one query can be paired to each gallery row, even when N is much larger than
    M. Equal scores are resolved by first index (torch.argmax), never labels.
    """
    query_to_gallery = scores.argmax(dim=1)
    gallery_to_query = scores.argmax(dim=0)
    rows = torch.arange(len(scores), device=scores.device)
    mutual = gallery_to_query[query_to_gallery] == rows
    query_indices = rows[mutual]
    gallery_indices = query_to_gallery[mutual]
    top = scores[query_indices].topk(2, dim=1).values
    margins = top[:, 0] - top[:, 1]
    all_query_indices = query_indices.clone()
    all_gallery_indices = gallery_indices.clone()
    all_margins = margins.clone()
    # The public paper gives the cap but not its selection rule. This fixed
    # descending-margin convention is documented, not claimed as author code.
    keep = torch.argsort(margins, descending=True, stable=True)[:config.max_landmarks]
    query_indices, gallery_indices, margins = query_indices[keep], gallery_indices[keep], margins[keep]
    clipped = margins.clamp_min(config.margin_floor)
    weights = len(clipped) * clipped / clipped.sum() if len(clipped) else clipped
    return {
        "query_indices": query_indices,
        "gallery_indices": gallery_indices,
        "margins": margins,
        "weights": weights,
        "all_mutual_query_indices": all_query_indices,
        "all_mutual_gallery_indices": all_gallery_indices,
        "all_mutual_margins": all_margins,
    }


def fit_orthogonal_map(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor,
                       rho: float = 0.1) -> dict[str, Any]:
    """Weighted centered row-vector Procrustes, exactly Eq. 6--8.

    Computes M = Xc.T W Yc + lambda I and R = U Vh. No determinant correction:
    the paper permits the full orthogonal group, including reflections.
    """
    x, y, weights = x.double(), y.double(), weights.double()
    if x.shape != y.shape or weights.shape != (len(x),) or len(x) < 2:
        raise ValueError("Invalid paired landmark dimensions")
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Landmark weights must be finite and positive")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or rho < 0:
        raise ValueError("Invalid landmark values or regularization")
    mu_x = (weights[:, None] * x).sum(0) / weights.sum()
    mu_y = (weights[:, None] * y).sum(0) / weights.sum()
    xc, yc = x - mu_x, y - mu_y
    cross = xc.T @ (weights[:, None] * yc)
    singular = torch.linalg.svdvals(cross)
    spectral = singular[0]
    identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    lam = rho * spectral
    if float(spectral) == 0.0:
        # No supported orientation; report degeneracy instead of an arbitrary
        # SVD choice. The caller also skips landmark-center translation.
        rotation = identity
        status = "degenerate_cross_covariance"
        conditioned_singular = singular
    else:
        u, conditioned_singular, vh = torch.linalg.svd(cross + lam * identity, full_matrices=False)
        rotation = u @ vh
        status = "complete"
    before = (weights[:, None] * (xc - yc).square()).sum()
    after = (weights[:, None] * (xc @ rotation - yc).square()).sum()
    regularizer = lam * (rotation - identity).square().sum()
    orthogonality_error = torch.linalg.vector_norm(rotation.T @ rotation - identity)
    rank_tol = max(cross.shape) * torch.finfo(cross.dtype).eps * spectral
    diag = {
        "status": status,
        "spectral_norm": float(spectral),
        "lambda": float(lam),
        "rho": rho,
        "cross_covariance_rank": int((singular > rank_tol).sum()),
        "weighted_squared_error_identity": float(before),
        "weighted_squared_error_recovered": float(after),
        "regularization_penalty": float(regularizer),
        "regularized_objective": float(after + regularizer),
        "orthogonality_fro_error": float(orthogonality_error),
        "rotation_from_identity_fro": float(torch.linalg.vector_norm(rotation - identity)),
        "regularized_smallest_singular_value": float(conditioned_singular[-1]),
        "weighted_center_gap": float(torch.linalg.vector_norm(mu_y - mu_x)),
    }
    if not torch.isfinite(rotation).all() or float(orthogonality_error) > 1e-7 * x.shape[1]:
        raise RuntimeError("SCORE orthogonal map failed finite/orthogonality checks")
    return {"rotation": rotation, "source_center": mu_x, "target_center": mu_y, "diagnostics": diag}


def _summary(x: torch.Tensor) -> dict[str, float]:
    return {"min": float(x.min()), "mean": float(x.mean()), "max": float(x.max())}


@torch.no_grad()
def run_score_deployment(query: torch.Tensor, gallery: torch.Tensor,
                         config: ScoreConfig | None = None) -> dict[str, Any]:
    """Pure, deterministic, label-free deployment for one target subject.

    Input features retain the caller's norm convention. Eq. 3 uses these actual
    values; cosine normalization happens only when constructing score matrices.
    Computation is float64 on the input device. The function changes neither
    inputs, random state, global torch settings, nor network parameters.

    Returns all score ablations, intermediate features, the affine recovery,
    pseudo-pairs, and JSON-compatible diagnostics. Insufficient landmarks leave
    recovered features equal to moment-matched features and explicitly mark
    recovery incomplete. No non-mutual pairs are invented to meet the minimum.
    """
    config = config or ScoreConfig()
    _validate(query, gallery, config)
    q, g = query.detach().double(), gallery.detach().double()
    mu_q, mu_g = q.mean(0), g.mean(0)
    # Population std is a declared local convention; v1 does not specify ddof.
    sd_q, sd_g = q.std(0, unbiased=False), g.std(0, unbiased=False)
    scale = (sd_g + config.epsilon) / (sd_q + config.epsilon)
    moment = (q - mu_q) * scale + mu_g
    raw_cosine = cosine_scores(q, g, config.epsilon)
    raw_csls, raw_diag = csls_scores(raw_cosine, config.neighbors)
    moment_cosine = cosine_scores(moment, g, config.epsilon)
    moment_csls, moment_diag = csls_scores(moment_cosine, config.neighbors)
    landmarks = select_landmarks(moment_csls, config)
    n_mutual = len(landmarks["all_mutual_query_indices"])
    n_selected = len(landmarks["query_indices"])
    fit = None
    recovered = moment
    status = "insufficient_landmarks"
    if n_selected >= config.min_landmarks:
        fit = fit_orthogonal_map(moment[landmarks["query_indices"]],
                                 g[landmarks["gallery_indices"]], landmarks["weights"], config.rho)
        status = fit["diagnostics"]["status"]
        if status == "complete":
            recovered = (moment - fit["source_center"]) @ fit["rotation"] + fit["target_center"]
    recovered_cosine = cosine_scores(recovered, g, config.epsilon)
    recovered_csls, recovered_diag = csls_scores(recovered_cosine, config.neighbors)
    scores = {
        "raw_cosine": raw_cosine,
        "raw_csls": raw_csls,
        "moment_cosine": moment_cosine,
        "moment_csls": moment_csls,
        "recovered_cosine": recovered_cosine,
        "recovered_csls": recovered_csls,
    }
    if not torch.isfinite(moment).all() or not torch.isfinite(recovered).all():
        raise RuntimeError("Nonfinite recovered SCORE features")
    if not all(bool(torch.isfinite(v).all()) for v in scores.values()):
        raise RuntimeError("Nonfinite SCORE retrieval scores")
    diag = {
        "paper_url": PAPER_URL,
        "implementation": "public-v1-formula-port; no author code or public supplement located",
        "status": status,
        "recovery_complete": status == "complete",
        "config": asdict(config),
        "query_count": len(q), "gallery_count": len(g), "feature_dimension": q.shape[1],
        "rectangular_input": len(q) != len(g),
        "query_gallery_ratio": len(q) / len(g),
        "query_norm": _summary(q.norm(dim=1)), "gallery_norm": _summary(g.norm(dim=1)),
        "input_normalization": "none; scoring alone uses cosine normalization",
        "moment_matching": {
            "std_convention": "population (ddof=0)",
            "scale": _summary(scale),
            "query_constant_dimensions": int((sd_q <= config.epsilon).sum()),
            "gallery_constant_dimensions": int((sd_g <= config.epsilon).sum()),
            "matched_mean_max_abs_error": float((moment.mean(0) - mu_g).abs().max()),
            "relative_displacement": _summary((moment - q).norm(dim=1) / q.norm(dim=1).clamp_min(config.epsilon)),
        },
        "landmarks": {
            "mutual_pairs": n_mutual, "selected_pairs": n_selected,
            "min_required": config.min_landmarks, "max_retained": config.max_landmarks,
            "capped": n_mutual > n_selected,
            "query_fraction_selected": n_selected / len(q),
            "gallery_fraction_selected": n_selected / len(g),
            "margins": _summary(landmarks["margins"]) if n_selected else None,
            "weights": _summary(landmarks["weights"]) if n_selected else None,
            "margin_floor_active_count": int((landmarks["margins"] < config.margin_floor).sum()),
        },
        "recovery": fit["diagnostics"] if fit else {"status": status},
        "csls": {"raw": raw_diag, "moment": moment_diag, "recovered": recovered_diag},
        "protocol_notes": [
            "Uses this whole unlabeled query batch and candidate gallery: transductive deployment.",
            "Mutuality permits at most one landmark query per gallery; no ground-truth grouping.",
            "Single-trial Nx200 input differs from the paper's repetition-averaged EEG protocol.",
            "Unequal repeated-image frequencies alter query moments and gallery-side CSLS neighborhoods.",
            "eps/eta/ddof/cap policy/insufficient policy are fixed implementation conventions, not reported hyperparameters.",
        ],
    }
    return {
        "scores": scores,
        "features": {"moment": moment, "recovered": recovered},
        "landmarks": landmarks,
        "mapping": {"query_mean": mu_q, "gallery_mean": mu_g, "moment_scale": scale,
                    "recovery_applied": status == "complete",
                    "rotation": fit["rotation"] if status == "complete" else None,
                    "source_center": fit["source_center"] if status == "complete" else None,
                    "target_center": fit["target_center"] if status == "complete" else None},
        "diagnostics": diag,
    }


def self_test() -> dict[str, Any]:
    """Synthetic checks only: no EEG, image identities, or test-set tuning."""
    generator = torch.Generator().manual_seed(20261108)
    x = torch.randn(40, 12, generator=generator, dtype=torch.float64)
    y = torch.randn(18, 12, generator=generator, dtype=torch.float64)
    c = cosine_scores(x, y)
    actual, _ = csls_scores(c, 10)
    expected = 2*c - c.sort(dim=1, descending=True).values[:, :10].mean(1, keepdim=True) - c.sort(dim=0, descending=True).values[:10].mean(0, keepdim=True)
    torch.testing.assert_close(actual, expected)
    w = torch.rand(40, generator=generator, dtype=torch.float64) + .2
    rotation, _ = torch.linalg.qr(torch.randn(12, 12, generator=generator, dtype=torch.float64))
    target = x @ rotation + .3
    fit = fit_orthogonal_map(x, target, w, rho=0)
    torch.testing.assert_close(fit["rotation"], rotation, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close((x-fit["source_center"]) @ fit["rotation"]+fit["target_center"], target)
    regularized = fit_orthogonal_map(x, target, w, rho=.1)
    rd = regularized["diagnostics"]
    assert rd["regularized_objective"] <= rd["weighted_squared_error_identity"] + 1e-9
    identity = fit_orthogonal_map(x, x, w, rho=.1)
    torch.testing.assert_close(identity["rotation"], torch.eye(12, dtype=torch.float64), atol=1e-10, rtol=1e-10)
    gallery = F.normalize(torch.randn(24, 16, generator=generator, dtype=torch.float64), dim=1)
    query = gallery.repeat_interleave(3, dim=0) + .001 * torch.randn(72, 16, generator=generator, dtype=torch.float64)
    q0, g0, rng0 = query.clone(), gallery.clone(), torch.random.get_rng_state().clone()
    result = run_score_deployment(query, gallery, ScoreConfig(max_landmarks=15))
    assert result["diagnostics"]["recovery_complete"]
    assert result["scores"]["recovered_csls"].shape == (72, 24)
    mapping = result["mapping"]
    reconstructed = ((query.double()-mapping["query_mean"]) * mapping["moment_scale"]
                     + mapping["gallery_mean"]-mapping["source_center"]) @ mapping["rotation"] + mapping["target_center"]
    torch.testing.assert_close(reconstructed, result["features"]["recovered"])
    selected = result["landmarks"]
    assert len(selected["query_indices"]) == 15
    assert len(selected["gallery_indices"].unique()) == 15
    assert abs(float(selected["weights"].mean()) - 1) < 1e-12
    torch.testing.assert_close(query, q0, atol=0, rtol=0)
    torch.testing.assert_close(gallery, g0, atol=0, rtol=0)
    assert torch.equal(torch.random.get_rng_state(), rng0)
    # A tied/constant query pool has at most one mutual pair; no forced padding.
    skipped = run_score_deployment(torch.ones(18, 16), gallery)
    assert skipped["diagnostics"]["status"] == "insufficient_landmarks"
    assert skipped["mapping"]["rotation"] is None and not skipped["mapping"]["recovery_applied"]
    torch.testing.assert_close(skipped["features"]["recovered"], skipped["features"]["moment"])
    # Query/gallery ordering must not change scores when nearest matches are unique.
    p = torch.randperm(len(query), generator=generator)
    gp = torch.randperm(len(gallery), generator=generator)
    shuffled = run_score_deployment(query[p], gallery[gp], ScoreConfig(max_landmarks=15))
    torch.testing.assert_close(shuffled["scores"]["recovered_csls"], result["scores"]["recovered_csls"][p][:, gp], atol=1e-9, rtol=1e-9)
    return {"status": "passed", "checks": ["rectangular CSLS formula", "row-vector rotation orientation",
            "weighted centering", "regularized objective", "identity map", "mutual-pair uniqueness and cap",
            "mean-one margin weights", "rectangular deployment", "returned mapping replay", "no input/RNG mutation",
            "insufficient-landmark fallback", "permutation equivariance"], "synthetic_only": True}


if __name__ == "__main__":
    import json
    torch.set_num_threads(4)
    print(json.dumps(self_test(), indent=2))
