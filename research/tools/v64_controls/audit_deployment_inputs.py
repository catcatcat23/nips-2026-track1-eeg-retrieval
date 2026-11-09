"""Read-only audit of cached native blur queries and deployment score artifacts.

No feature extraction, training, or artifact writing. Prints JSON to stdout.
An optional run/audit_index.json contains {"records": [{"fold": 1,
"arm": "baseline", "metadata": {"candidate_ids": [...],
"test_order_sha256": "...", "feature_sha256": "...",
"checkpoint_sha256": "..."}, "labels_file": "labels.npy", "methods": {
"raw_dot": {"scores_file": "relative.npy", "metrics": {...}}}}]}.
Labels enter evaluation only; this script does not fit a deployment method.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common_target_readout import ROOT, digest, load
from pooled_residual_calibration import native


def array_hash(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def score_metrics(scores, labels):
    """Group final candidate scores, without returning labels to any transform."""
    scores = torch.as_tensor(np.asarray(scores)).double()
    labels = torch.as_tensor(labels).long()
    assert scores.ndim == 2 and scores.shape == (len(labels), 200)
    assert torch.isfinite(scores).all()
    counts = torch.bincount(labels, minlength=200)
    assert bool((counts > 0).all())
    grouped = torch.stack([scores[labels == i].mean(0) for i in range(200)])

    def hit(matrix, target):
        indices = matrix.topk(5, dim=1).indices
        return dict(top1=float((indices[:, 0] == target).double().mean()),
                    top5=float((indices == target[:, None]).any(1).double().mean()),
                    count=len(target))

    return dict(single_trial=hit(scores, labels),
                subject_aggregated=hit(grouped, torch.arange(200)),
                repeats_min=int(counts.min()), repeats_max=int(counts.max()))


def assert_metrics(actual, expected, tolerance=2e-6):
    for group in ("single_trial", "subject_aggregated"):
        if "count" in expected[group]:
            assert actual[group]["count"] == expected[group]["count"]
        for metric in ("top1", "top5"):
            assert abs(actual[group][metric] - expected[group][metric]) < tolerance, (
                group, metric, actual[group][metric], expected[group][metric])


@torch.inference_mode()
def audit_inputs(fold, arm):
    cache, raw_targets, raw_gallery, labels, meta = load(fold, arm)
    projection, _, gallery, expected = native(fold, arm, cache, meta)
    folder = ROOT / f"fold_{fold:02d}_{arm}"
    cfg = json.loads((folder / "config.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    data = Path(cfg["source_root"]) / "data"
    image_ids = np.load(data / "test_ids.npy")
    subjects = np.load(data / "test_subjects.npy").astype(str)
    rows = np.flatnonzero(subjects == cfg["heldout_subject"])
    ordered_images = image_ids[rows]
    candidates = np.asarray(meta["candidate_ids"])
    assert len(rows) == len(labels) == len(cache["etest"])
    assert np.array_equal(candidates[np.asarray(labels)], ordered_images)
    assert cfg["heldout_subject"] not in protocol["subjects"]
    assert len(set(protocol["subjects"])) == 9
    assert all(not p.requires_grad for p in projection.parameters())
    assert not projection.training
    query = projection(cache["etest"].float())
    assert query.shape == (len(rows), 1536) and gallery.shape == (200, 1536)
    assert torch.isfinite(query).all() and torch.isfinite(gallery).all()
    assert bool((query.norm(dim=1) > 0).all())
    torch.testing.assert_close(gallery.norm(dim=1), torch.ones(200))
    raw_scores = query @ gallery.T
    cosine_scores = F.normalize(query, dim=1) @ gallery.T
    raw = score_metrics(raw_scores.numpy(), labels)
    cosine = score_metrics(cosine_scores.numpy(), labels)
    assert_metrics(raw, expected)
    for metric in ("top1", "top5"):
        assert abs(raw["single_trial"][metric] - cosine["single_trial"][metric]) < 2e-6
    # Independent grouping verifies raw score averaging equals feature averaging.
    feature_means = torch.stack([query[labels == i].double().mean(0) for i in range(200)])
    score_means = torch.stack([raw_scores[labels == i].double().mean(0) for i in range(200)])
    linear_error = float((feature_means @ gallery.double().T - score_means).abs().max())
    assert linear_error < 1e-4
    test_order = np.stack([rows, ordered_images], axis=1).astype(np.int64)
    metadata = dict(fold=fold, arm=arm, heldout=cfg["heldout_subject"],
                    feature_cache=meta["feature_cache"], feature_sha256=meta["feature_sha256"],
                    checkpoint=meta["checkpoint"], checkpoint_sha256=meta["checkpoint_sha256"],
                    protocol_sha256=digest(folder / "protocol.json"),
                    config_sha256=digest(folder / "config.json"),
                    candidate_ids=candidates.tolist(), test_order_sha256=array_hash(test_order),
                    test_order_bytes_sha256=hashlib.sha256(test_order.tobytes()).hexdigest(),
                    test_indices_sha256=array_hash(rows.astype(np.int64)),
                    labels_sha256=array_hash(labels.numpy().astype(np.int64)),
                    labels_bytes_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
                    query_sha256=array_hash(query.numpy()), gallery_sha256=array_hash(gallery.numpy()),
                    source_subjects=protocol["subjects"],
                    query_shape=list(query.shape), gallery_shape=list(gallery.shape),
                    test_count=len(rows), raw_dot_metrics=raw, norm_only_metrics=cosine,
                    linear_score_mean_max_error=linear_error,
                    provenance_scope="current file hashes and extraction protocol plus reproduced original zero metrics; cache contains no extraction-time checkpoint hash")
    # Retain matching arrays in memory only for cross-arm and saved-score checks.
    private = dict(labels=labels, rows=rows, ordered_images=ordered_images,
                   raw_gallery=raw_gallery, raw_targets=raw_targets,
                   support_images=cache["images"], support_records=cache["records"],
                   raw_scores=raw_scores, cosine_scores=cosine_scores)
    return metadata, private


def audit_run(run, records, inputs):
    index = json.loads((run / "audit_index.json").read_text())
    records_by_key = {(r["fold"], r["arm"]): r for r in records}
    outcomes = []
    for record in index["records"]:
        key = (int(record["fold"]), record["arm"])
        meta, arrays = records_by_key[key], inputs[key]
        saved_meta = record["metadata"]
        for field in ("candidate_ids", "feature_sha256", "checkpoint_sha256"):
            assert saved_meta[field] == meta[field], (key, field)
        allowed_order_hashes = {meta[field] for field in (
            "test_order_sha256", "test_order_bytes_sha256", "labels_bytes_sha256")}
        assert saved_meta["test_order_sha256"] in allowed_order_hashes, (key, "test_order_sha256")
        label_path = (run / record["labels_file"]).resolve()
        assert label_path.is_relative_to(run.resolve()), label_path
        assert np.array_equal(np.load(label_path, allow_pickle=False), arrays["labels"].numpy())
        for name, method in record["methods"].items():
            path = (run / method["scores_file"]).resolve()
            assert path.is_relative_to(run.resolve()), path
            scores = np.load(path, mmap_mode="r", allow_pickle=False)
            assert scores.shape == (meta["test_count"], 200)
            metrics = score_metrics(scores, arrays["labels"])
            assert_metrics(metrics, method["metrics"])
            if name in ("raw_dot", "norm_only", "cosine", "raw_cosine"):
                original = arrays["raw_scores"] if name == "raw_dot" else arrays["cosine_scores"]
                torch.testing.assert_close(torch.from_numpy(np.array(scores)), original,
                                           rtol=1e-5, atol=1e-5)
            outcomes.append(dict(fold=key[0], arm=key[1], method=name,
                                 scores_sha256=digest(path), metrics=metrics, state="passed"))
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    torch.set_num_threads(2)
    records, inputs = [], {}
    for fold in args.folds:
        for arm in ("baseline", "identity"):
            meta, private = audit_inputs(fold, arm)
            records.append(meta)
            inputs[(fold, arm)] = private
        baseline, identity = [inputs[(fold, arm)] for arm in ("baseline", "identity")]
        for field in ("rows", "ordered_images", "support_images"):
            assert np.array_equal(baseline[field], identity[field]), (fold, field)
        assert baseline["support_records"] == identity["support_records"]
        for field in ("labels", "raw_gallery", "raw_targets"):
            assert torch.equal(baseline[field], identity[field]), (fold, field)
    result = dict(state="passed", records=records,
                  paired_samples="same EEG rows, labels, support records and raw visual candidates across arms",
                  aggregation="mean final candidate scores for each subject/image; labels used only for scoring",
                  transductive_scope="all single-trial queries may enter unlabeled adaptation; no image-group aggregation before adaptation")
    if args.run:
        result["run_score_audit"] = audit_run(args.run, records, inputs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
