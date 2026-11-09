# v12 clean-only controlled baseline

Same competition120 input cache and REVE interpolation/window normalization as
v10/v11, same seed21 pretrained REVE and Dropout0.1/Linear512->1536 head.
No restoration parameters, image adapter, or blur data reads. Only clean
bidirectional L2-normalized multipos contrast at temperature0.07, full DDP
batch1024 (512/rank). Encoder LR1e-5, head LR1e-4, AdamW wd0.01, BF16,
200 epochs, warmup2, plateau5 factor0.5, early stopping25. Selection remains
validation subject_aggregated.top5. Same fixed training diagnostics.
Restore is constructed then removed at initialization only to preserve the
reference RNG stream; no restoration parameters remain in model/optimizer.
Run on GPU0,1 independently from v11 on GPU2,3. Fresh pretrained initialization;
no warm-start. Reuse clean data only, no features symlink is required.
Real8 overfit, full-batch five-step DDP gradient gate, sampled CPU/GPU metrics
checks precede full training. Final test only after best checkpoint selection.
This is a clean-only baseline for our new training recipe, not an exact
reproduction of historical NeuralBench's batch64 single-direction objective.
