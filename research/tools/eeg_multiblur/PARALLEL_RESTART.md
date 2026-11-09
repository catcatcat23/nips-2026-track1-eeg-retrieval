# v8: v5 weights-only restart with parallel validation

User chose best-weight continuation with an explicitly reset AdamW optimizer.
V7 REVE on GPUs 2,3 is untouched. V8 uses GPUs 0,1 and the v5 MLP configuration.
The source checkpoint is copied into v8 before launch; old results are retained.
Epoch numbering starts after the source checkpoint; total cap remains 200.
Optimizer and two-epoch warmup restart; cosine runs over the remaining epochs.
Early stopping patience remains 5; its counter restarts at zero; the old best
score/checkpoint remain the initial incumbent. This is not an exact resume.

Each validation rank handles indices rank::world_size, with no sampler padding.
Predictions are gathered and reconstructed in original order. Exact image-feature
deduplication operates on unique image IDs first (same candidate values).
Subject aggregation keeps the official CPU float64 group_means function.
Queries are sharded across GPUs for retrieval. GPU matrix multiplication uses
FP32 for single trials and FP64 for subject means, with TF32 disabled. Tied
scores use official average rank; NaN target scores miss. Rows close enough to
a Top-1/5 boundary to risk numerical disagreement are recomputed on CPU.
CPU grouping and communication still create some idle periods; this does not
promise continuous 100% GPU utilization.

New last.pt records optimizer, scheduler, epoch and early-stop state each epoch,
in addition to best.pt. Per-rank RNG state is not yet saved, so it is not a
bitwise-reproducible restart checkpoint.
