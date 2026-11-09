REVE v10 controlled rerun of v7 from pretrained weights, seed 21, no warm start.
GPU 2,3; batch 512/rank (1024 global); encoder LR 1e-5, heads 1e-4;
200 epochs maximum; 2 epoch step warmup, then ReduceLROnPlateau mode=max,
factor=.5 after exactly 5 bad validations (PyTorch patience=4), absolute
threshold=0, minimum LR encoder 1e-7/head 1e-6. Early stopping patience25.
All selection uses validation subject_aggregated.top5, strict improvement.
Same data, features, objective and model as v7; no extraction or image adapter.
Epoch logs include unweighted loss components, current LR, full validation
and per-subject metrics. Fixed training diagnostic samples 256 trials/subject
with seed21, saved indices; gallery consists of unique targets in this subset.
Its retrieval values are not directly comparable to full validation gallery.
Fresh optimizer; no inherited early-stop counter. Both GPUs evaluate without
sampler padding. Latest checkpoints include optimizer/scheduler states.
Gates: scheduler CPU boundary check, eight-real-example overfit, five full-DDP
training steps, sampled CPU/GPU retrieval and per-subject consistency checks.
Final test only after best validation checkpoint selection.
