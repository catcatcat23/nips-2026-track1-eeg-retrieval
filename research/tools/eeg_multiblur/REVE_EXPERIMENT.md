# Frozen REVE + MultiBlur recovery (v6)

Independent comparison to v5; existing v5 snapshot/processes remain unchanged.
Pretrained brain-bzh/reve-base is frozen and permanently in evaluation mode.
Competition 63-channel, 120 Hz, 120-point windows are reused from v4.
Following competition-2026/tracks/image_decoding/solvers/reve_probe.py:
linear interpolation to 200 points, per-channel population-standard-deviation
normalization (epsilon 1e-6), clipping to [-15,15], standard channel positions,
mean pooling across channel/time tokens to 512 dimensions. Frozen CUDA
extraction uses BF16 consistently in training and validation, microbatch 64.
This precision differs from the official FP32 probe. No EEG trial averaging.

512 -> dropout 0.1 -> Linear(512,1536) -> residual recovery head -> clean output.
Only projection and recovery parameters train. REVE dropout is disabled by eval.
No image adapter, image dropout, or OT. Same immutable clean/blur caches as v5.
Loss = clean contrastive + 0.3 blur contrastive + 0.1 cosine restoration.
DDP global negatives and repeated-image positives remain unchanged.

2 A100 80GB GPUs (2,3), batch 512 each, 1024 global, seed 21,
AdamW 1e-4 / weight decay .01, 200 epochs maximum, patience 5,
2-epoch linear warmup, cosine to 1e-6, gradient clipping 1.
Best checkpoint uses unchanged full-validation subject-aggregated Top-5.
Final test evaluated only for the selected checkpoint.

Gates: eight real distinct training images overfit (dropout disabled for this
sanity check only), then five real DDP full-batch steps with finite gradients.
Frozen features are computed online, not cached; epoch cost includes REVE.
Output checkpoint is an internal inference artifact, not an official submission ZIP.

## v7: user-requested full backbone finetuning

Supersedes frozen v6, whose artifacts are retained. Uses GPUs 2,3; v5 untouched.
Starts afresh from the pretrained REVE weights, not from v6's trained heads.
All REVE feature backbone parameters now receive gradients and optimizer updates.
Only the unused original REVE classification head remains frozen, since
return_features bypasses it. Encoder train/eval mode now follows the outer model.
No detach or no_grad is used on the training feature path. 72,341,760 parameters
are trainable including projection and restoration heads.

AdamW groups: backbone lr=1e-5, heads lr=1e-4. Both use the same two-epoch
warmup and cosine multiplier, ending at 1% of their respective peak rates
(encoder 1e-7, heads 1e-6). Other configuration remains as recorded in config.json:
batch512 per GPU/global1024, 200 epoch cap, patience5, seed21, dropout0.1,
clean+0.3blur+0.1restore, no OT, same data and image feature caches.
Microbatch64 divides forward operations but retains every training chunk's
backward graph: it is not gradient accumulation and does not reduce negatives.

The tiny preflight now trains end-to-end on eight real distinct images; checks
all trainable gradients, nonzero patch-embedding gradients and actual encoder
weight changes. DDP preflight checks finite gradients on every trainable parameter
on both ranks before full training. These checks start from fresh models and do
not transfer their weights into the full experiment.
