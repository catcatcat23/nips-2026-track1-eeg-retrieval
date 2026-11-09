# Clean-only EMA and augmentation controls

Launched 2026-10-31. These are matched new controls, not continuations of trained retrieval checkpoints.

| Run | GPU | Training change | Evaluated weights |
| --- | --- | --- | --- |
| v70_clean_ema_control | 2 | Pretrained REVE + fresh linear clean projection; no augmentation | Raw, EMA 0.9998 |
| v71_clean_ema_control | 3 | Same model + 50/25/25% single/mean2/mean4 same-subject/image EEG and 20% chance of masking one channel | Raw, EMA 0.9998 |
| v72_cross_subject_mix | 2 after v70 | Same as v71, but each averaged trial comes from a distinct subject viewing the same image, within the same dataset | Raw, EMA 0.9998 |

v72 is queued, not immediately running: its launcher waits for v70 completion and GPU2 memory below 1GB, then performs its GPU preflight before training. CPU checks verify real-array means, matching image IDs and distinct subjects in each dataset. No cross-dataset EEG averaging or test-time augmentation. v72 versus v71 isolates mixing scope; it does not alone distinguish noise cancellation from learned subject invariance.

Both jointly train THINGS train+val and original Alljoined2025 train, with native 63/32 channel positions. Equal domain gradient weights, batch256/domain, effective512, seed33, FP32 AdamW1e-4 wd0.05, clip1, OneCycle40 epochs pct_start0.1, patience5. No blur loss, image adapter, recovery head, or local OT. Official EEG caches and 120-to-200 REVE input adaptation are unchanged.

Each run contains a read-only source snapshot and source_hashes.json under `/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v70_clean_ema_control` or `reve_v71_clean_ema_control`. launch.sh gates training on real-data tiny-overfit and gradient/EMA checks. Initial model SHA is recorded to verify identical initialization across arms.

Record clean single-trial and subject-aggregated Top1/5 for both datasets and both weight variants each epoch. Aggregate raw predictions within subject/image before cosine normalization, matching official evaluation. There is no learned blur branch in these clean-only controls, so blur retrieval is not a primary metric.

EMA is passive and never changes optimizer updates. This yields four comparisons from two training trajectories. At a common epoch, augmentation effect without EMA = v71 raw - v70 raw; with EMA = v71 EMA - v70 EMA. Interaction = (v71 EMA - v71 raw) - (v70 EMA - v70 raw). A higher combined score alone does not establish a positive interaction. Compare common epochs, not only independently selected peaks. Early stopping can yield different horizons; compare the shared horizon and disclose stopping epochs.

Public test clean single-trial Top5 (mean across datasets) selects checkpoints/early stopping, consistent with current development protocol. These scores are not independent test evidence. One seed is exploratory, not statistical proof or a paper-ready causal result. Augmentation is bundled, so this pair cannot separately attribute gains to trial mixing versus channel masking.
