"""Label-free SATTC deployment operators from a pinned author snapshot.

This ports the deployment stage to frozen REVE embeddings; it does not reproduce
SATTC's encoder training or its EEG-before-inference averaging protocol. Author
functions are AST-extracted without importing or running their training script.
See sattc_deployment.md for the exact scope and source/default provenance.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

AUTHOR_COMMIT = "617cba60db52fa2e51e52f05b8f75bc13259f8ef"
DEFAULT_SOURCE = Path(
    "/mnt/DATA-4/anteng/eeg-retrieval/experiments/"
    "sattc_score_deployment_20261108/sources/SATTC-CVPR2026-617cba60db52"
)
SOURCE_HASHES = {
    "Retrieval/run_sattc_loso.py": "a6765860b298b6ff7c5a3afcbb1c2022b069223681f49c46146ece74fdf8c2c0",
    "Retrieval/utils/cli_args.py": "93aff0b7fd22a7ed001ad785dda2fb4d398ca1009c0c49f05e8b76612bb3e7a0",
}
_FUNCTIONS = {
    "_eigh_inv_sqrt", "estimate_mu_cov", "subject_adaptive_whiten",
    "_whiten_features", "_build_whitening_state_from_features",
    "csls_scores", "csls_adaptive", "build_structural_expert", "fuse_poe_scores",
}
_MAIN_DEFAULTS = {
    "use_saw": True, "saw_shrink": 0.6, "saw_diag": False,
    "global_auto_whiten": True, "global_whiten_auto_shrink": 0.5,
    "global_whiten_auto_diag": False,
    "logits_auto_center": True, "logits_auto_scale": True,
}


class _FailClosed(ast.NodeTransformer):
    """Keep successful author arithmetic, but never silently skip a component."""

    def visit_Try(self, node):
        self.generic_visit(node)
        for handler in node.handlers:
            handler.body = [ast.Raise(exc=None, cause=None)]
        return node


def _function_ast(name, arguments, body):
    node = ast.parse(f"def {name}({arguments}):\n    pass\n").body[0]
    node.body = body
    return ast.fix_missing_locations(node)


@lru_cache(maxsize=2)
def _load_author(source_dir: str):
    source = Path(source_dir)
    for relative, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SATTC source hash mismatch: {relative}: {actual}")
    tree = ast.parse((source / "Retrieval/run_sattc_loso.py").read_text())
    cli_tree = ast.parse((source / "Retrieval/utils/cli_args.py").read_text())
    defaults = dict(_MAIN_DEFAULTS)
    for node in cli_tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"_CANONICAL_DEFAULTS", "_STRUCTURAL_DEFAULTS"}:
                defaults.update(ast.literal_eval(node.value))
    # Verify the actual CLI overrides rather than trusting helper defaults.
    parsed_cli = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        flag = node.args[0]
        if not isinstance(flag, ast.Constant) or not isinstance(flag.value, str):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    parsed_cli[flag.value.lstrip("-")] = ast.literal_eval(keyword.value)
                except ValueError:
                    pass
    for key, expected in _MAIN_DEFAULTS.items():
        if parsed_cli.get(key) != expected:
            raise ValueError(f"Unexpected author CLI default for {key}")

    namespace = {"torch": torch, "F": F, "np": np, "math": math,
                 "Dict": Dict, "Any": Any, "Optional": Optional}
    author_nodes = [copy.deepcopy(n) for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name in _FUNCTIONS]
    if {n.name for n in author_nodes} != _FUNCTIONS:
        raise ValueError("Missing required SATTC source functions")
    evaluate = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "evaluate_model")
    start = next(i for i, n in enumerate(evaluate.body)
                 if isinstance(n, ast.Assign) and any(
                     isinstance(t, ast.Tuple) and any(
                         isinstance(e, ast.Name) and e.id == "q_dim" for e in t.elts
                     ) for t in n.targets))
    evidence_end = next(i for i in range(start, len(evaluate.body))
                        if isinstance(evaluate.body[i], ast.Try))
    evidence_body = copy.deepcopy(evaluate.body[start:evidence_end + 1])
    # Author case6 diagnostic variables are uninitialised when that case is empty.
    # Initialising from the same config changes no scores in the normal path.
    guards = ast.parse(
        "case6_pen_high = max(0., float(getattr(config, 'pre_csls_case6_penalty_high', .5) or 0.))\n"
        "case6_pen_mid = max(0., float(getattr(config, 'pre_csls_case6_penalty_mid', .25) or 0.))\n"
    ).body
    evidence_body = guards + evidence_body + [
        ast.Return(value=ast.Name(id="pre_csls_evidence", ctx=ast.Load()))]
    author_nodes.append(_function_ast("_author_evidence", "base_logits, config", evidence_body))
    compiled = _FailClosed().visit(ast.Module(body=author_nodes, type_ignores=[]))
    exec(compile(ast.fix_missing_locations(compiled), str(source), "exec"), namespace)
    return namespace, defaults, tree


def author_defaults(source_dir=None):
    """Return an independent dict of actual pinned command-line defaults."""
    _, defaults, _ = _load_author(str(Path(source_dir or DEFAULT_SOURCE)))
    return dict(defaults)


def _input(value, name):
    value = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if value.ndim != 2 or min(value.shape) < 2:
        raise ValueError(f"{name} must have shape [N,D], with N,D >= 2")
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite values in {name}")
    return value


def _summary(value):
    v = value.detach().float()
    return {"min": float(v.min()), "mean": float(v.mean()),
            "max": float(v.max()), "std": float(v.std(unbiased=False))}


@torch.no_grad()
def calibrate_sattc(query, gallery, *, config=None, gallery_reference=None, source_dir=None):
    """Calibrate one subject's query set without EEG labels or image IDs.

    Args:
        query: Raw projected EEG features [N,D], not unit-normalised beforehand.
        gallery: Frozen native image-adapter candidate features [M,D].
        gallery_reference: Unique source-training image features [R,D] used for
            author-style candidate-whitening statistics. If omitted, statistics
            use the gallery; this is explicitly marked as a protocol transfer.
        config: Optional overrides of documented author CLI defaults. No search.
        source_dir: The pinned author source snapshot directory.

    Returns:
        scores: CPU float32 matrices [N,M] for raw_dot, raw_cosine,
            whiten_cosine, whiten (global z-score), geometry (Ada-CSLS), full (PoE).
        diagnostics: JSON-serialisable provenance, parameters and numerical data.

    Query statistics use the entire unlabeled query set: this is transductive.
    Batch splitting changes the method. Do not pass different subjects together.
    Labels may only be used by the caller's evaluation after this function returns.
    """
    namespace, defaults, _ = _load_author(str(Path(source_dir or DEFAULT_SOURCE)))
    settings = dict(defaults)
    overrides = dict(config or {})
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown SATTC config keys: {sorted(unknown)}")
    settings.update(overrides)
    for flag in ("use_saw", "global_auto_whiten", "use_csls", "use_ada_csls", "enable_poe"):
        if not settings[flag]:
            raise ValueError(f"Use returned stage ablations instead of disabling {flag}")
    for name in ("saw_shrink", "global_whiten_auto_shrink"):
        if not 0 <= float(settings[name]) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    query, gallery = _input(query, "query"), _input(gallery, "gallery")
    reference_kind = "source_training_unique_images" if gallery_reference is not None else "gallery_only_transfer"
    reference = _input(gallery_reference, "gallery_reference") if gallery_reference is not None else gallery
    if query.shape[1] != gallery.shape[1] or reference.shape[1] != gallery.shape[1]:
        raise ValueError("Query, gallery and reference feature dimensions differ")
    cfg = SimpleNamespace(**settings)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        q_white = namespace["subject_adaptive_whiten"](
            query, shrink=cfg.saw_shrink, diag=cfg.saw_diag)
        state = namespace["_build_whitening_state_from_features"](
            reference, shrink=cfg.global_whiten_auto_shrink, diag=cfg.global_whiten_auto_diag)
        if state is None:
            raise RuntimeError("Author candidate whitening returned no state")
        g_white = F.normalize(namespace["_whiten_features"](gallery, state), dim=1, eps=1e-12)
        whiten_cosine = q_white @ g_white.T
        logits = whiten_cosine.clone()
        logit_mean, logit_std = logits.mean(), logits.std(unbiased=False)
        if cfg.logits_auto_center:
            logits = logits - logit_mean
        if cfg.logits_auto_scale:
            logits = logits / max(float(logit_std), 1e-6)
        evidence = namespace["_author_evidence"](logits, cfg)
        # Mirror evaluate_model's gallery-axis clamps before csls_adaptive.
        nc = gallery.shape[0]
        kmin = max(1, min(int(cfg.csls_kmin), nc))
        kmax = max(kmin, min(int(cfg.csls_kmax), nc))
        col_kmin = max(1, min(int(cfg.csls_col_kmin), nc))
        geometry, aux = namespace["csls_adaptive"](
            logits, k0=max(kmin, min(int(cfg.csls_k), kmax)), kmin=kmin, kmax=kmax,
            alpha=cfg.csls_alpha, m=max(1, min(int(cfg.csls_m), nc)),
            k_side=max(1, min(int(cfg.csls_k_side), nc)), col_alpha=cfg.csls_col_alpha,
            col_m=max(1, min(int(cfg.csls_col_m), nc)), col_kmin=col_kmin,
            col_kmax=max(col_kmin, min(int(cfg.csls_col_kmax), nc)), return_details=True)
        structural = namespace["build_structural_expert"](
            evidence, geometry, device=torch.device("cpu"),
            lambda_pen=cfg.poe_lambda_pen, lambda_bonus=cfg.poe_lambda_bonus)
        full = namespace["fuse_poe_scores"](geometry, structural, beta=cfg.poe_beta)
    scores = {
        "raw_dot": query @ gallery.T,
        "raw_cosine": F.normalize(query, dim=1, eps=1e-12) @ F.normalize(gallery, dim=1, eps=1e-12).T,
        "whiten_cosine": whiten_cosine, "whiten": logits,
        "geometry": geometry, "full": full,
    }
    for name, score in scores.items():
        if score.shape != (query.shape[0], gallery.shape[0]) or not torch.isfinite(score).all():
            raise RuntimeError(f"Invalid SATTC {name} scores")
    geom_std = max(float(geometry.std()), 1e-6)  # Author operator uses unbiased=True.
    diagnostics = {
        "method": "SATTC author deployment operators on frozen REVE",
        "source_commit": AUTHOR_COMMIT, "source_dir": str(Path(source_dir or DEFAULT_SOURCE)),
        "source_sha256": dict(SOURCE_HASHES), "author_defaults": dict(defaults),
        "config": settings, "overrides": overrides,
        "query_shape": list(query.shape), "gallery_shape": list(gallery.shape),
        "gallery_reference_shape": list(reference.shape), "gallery_reference_kind": reference_kind,
        "labels_used": False, "transductive_query_statistics": True,
        "aggregation": "caller averages per-trial calibrated score rows within subject/image",
        "author_aggregation": "test EEG repeats averaged before encoder; one query per image",
        "logit_scale": 1.0, "logit_scale_note": "REVE has no author logit_scale; global z-score cancels positive scale",
        "pre_csls_mean": float(logit_mean), "pre_csls_std": float(logit_std),
        "poe_lambda_pen_effective": cfg.poe_lambda_pen if cfg.poe_lambda_pen is not None else .5 * geom_std,
        "poe_lambda_bonus_effective": cfg.poe_lambda_bonus if cfg.poe_lambda_bonus is not None else .25 * geom_std,
        "case_counts": evidence["case_counts"], "penalty_stats": evidence["penalty_stats"],
        "hub_quantiles": evidence["hub_quantiles"],
        "ada_csls": {name: _summary(value) for name, value in aux.items()},
        "score_statistics": {name: _summary(value) for name, value in scores.items()},
        "fallback_policy": "fail closed; author silent fallback handlers re-raise",
        "edge_fix": "initialise author case6 diagnostic constants even when case6 is empty",
        "author_stdout": stdout.getvalue().splitlines(),
    }
    return {"scores": scores, "diagnostics": diagnostics}


run_sattc_deployment = calibrate_sattc


def verify_author_parity(source_dir=None):
    """Small CPU test against the unmodified author's evaluation score block.

    No model checkpoints or real EEG are read. The reference runs the entire
    official SAW-to-PoE block, independently of this wrapper's stage orchestration.
    """
    source_dir = str(Path(source_dir or DEFAULT_SOURCE))
    _, defaults, tree = _load_author(source_dir)
    helpers = _FUNCTIONS | {
        "_ada_required", "_ada_required_int", "_ada_required_float",
        "_ada_optional_int", "_ada_optional_float", "_assess_alignment_strict",
        "_summarise_logits_alignment",
    }
    namespace = {"torch": torch, "F": F, "np": np, "math": math,
                 "Dict": Dict, "Any": Any, "Optional": Optional,
                 "ADA_CSLS_DEFAULTS": defaults,
                 "_resolve_logits_zscore": lambda config: (None, None)}
    functions = [copy.deepcopy(n) for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in helpers]
    exec(compile(ast.Module(body=functions, type_ignores=[]), source_dir, "exec"), namespace)
    evaluate = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "evaluate_model")

    def assignment_index(name):
        return next(i for i, n in enumerate(evaluate.body)
                    if isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == name for t in n.targets))

    block = ast.Module(body=copy.deepcopy(evaluate.body[
        assignment_index("features_all_device"):assignment_index("labels_cpu_all")
    ]), type_ignores=[])
    reference_code = compile(block, source_dir, "exec")
    cases = []
    for seed, nq, nc, dim in ((33, 48, 23, 16), (34, 200, 200, 32), (35, 160, 17, 12)):
        generator = torch.Generator().manual_seed(seed)
        query = torch.randn(nq, dim, generator=generator) * torch.arange(1, dim + 1)
        gallery = F.normalize(torch.randn(nc, dim, generator=generator), dim=1)
        reference = F.normalize(torch.randn(250, dim, generator=generator), dim=1)
        query_copy, gallery_copy = query.clone(), gallery.clone()
        actual = calibrate_sattc(query, gallery, gallery_reference=reference, source_dir=source_dir)
        state = namespace["_build_whitening_state_from_features"](
            reference, shrink=defaults["global_whiten_auto_shrink"], diag=False)
        reference_namespace = dict(namespace)
        reference_namespace.update(
            features_all=query, device=torch.device("cpu"),
            img_features_all=F.normalize(namespace["_whiten_features"](gallery, state), dim=1, eps=1e-12),
            config=SimpleNamespace(**defaults), eeg_model=SimpleNamespace(logit_scale=torch.tensor(0.)),
            subject_indices_all=None, fallback_subject_numeric=1, use_cos=True, eval_debug_info={},
        )
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(reference_code, reference_namespace)
        for marker in ("falling back", "fallback", "skipping", "failed"):
            if marker in capture.getvalue().lower():
                raise AssertionError(f"Author reference used fallback: {capture.getvalue()}")
        errors = {}
        for stage, reference_name in (("whiten", "base_logits"), ("geometry", "S_geom"), ("full", "logits_all_cpu")):
            expected = reference_namespace[reference_name]
            torch.testing.assert_close(actual["scores"][stage], expected, rtol=0, atol=0)
            errors[stage] = float((actual["scores"][stage] - expected).abs().max())
        torch.testing.assert_close(actual["scores"]["raw_dot"], query @ gallery.T, rtol=0, atol=0)
        torch.testing.assert_close(actual["scores"]["raw_cosine"], F.normalize(query, dim=1) @ F.normalize(gallery, dim=1).T, rtol=0, atol=0)
        assert torch.equal(query, query_copy) and torch.equal(gallery, gallery_copy)
        assert not torch.allclose(actual["scores"]["raw_dot"], actual["scores"]["raw_cosine"])
        json.dumps(actual["diagnostics"])
        cases.append({"seed": seed, "query_shape": list(query.shape),
                      "gallery_shape": list(gallery.shape), "max_abs_error": errors})
    return {"passed": True, "source_commit": AUTHOR_COMMIT, "source_sha256": SOURCE_HASHES,
            "reference": "Unmodified official evaluate_model block, features_all_device through PoE",
            "cases": cases, "raw_baselines_exact": True, "inputs_unchanged": True,
            "diagnostics_json_serialisable": True, "author_fallbacks_triggered": False}

