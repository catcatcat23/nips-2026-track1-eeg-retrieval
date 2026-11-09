# Reference implementation correction

Reference: /home/Anteng/eeg-retrieval/reve-multiblur, branch feature/reve-multiblur, commit 94b3a31a3f3bf5837bad2df40cb89fc4a0cf0e32.
This is a separate recipe. Existing v11/v12 configs, snapshots, clean targets and caches are not rewritten.
Select it with EEG_CONFIG=/absolute/path/reference_config.json. The pre-existing modified config.json is intentionally untouched.

## Corrected image protocol

extract_reference.py uses the verbatim MIT upstream_blur.py function:
PIL RGB -> PIL resize224x224 -> OpenCV uint8 GaussianBlur(sigma=0) -> PIL -> torchvision Resize518/ToTensor -> official HF processor with do_rescale=False and pixel NaN handling -> hidden layer int(.6667*(len(hidden_states)-1)) -> all-token mean -> L2 EACH scale -> mean12.
It preserves per-scale vectors, filepath and global clean-cache image ID. Kernel1 is NOT the official clean target; clean.npy remains unchanged. This differs from the older extract.py Resize518/float blur/raw mean protocol. The old extractor remains only for reproducing historical experiments.

Use a NEW run directory with a data symlink to the competition120 cache, then:

    EEG_CONFIG=/absolute/path/reference_config.json torchrun --standalone --nproc_per_node=2 extract_reference.py
    EEG_CONFIG=/absolute/path/reference_config.json python merge_reference.py

OpenCV is available in the existing isolated deps directory:
/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_dinov2_mean_sub01_seed21/deps
Add it to PYTHONPATH; do not silently replace shared dependencies.

The merger checks every train+validation image ID/filepath, 12 unit-norm vectors and their mean; it refuses an existing destination. Missing test auxiliary features are NaN and are never used. The trainer rejects a cache without schema upstream_per_scale_l2_dino_v1.

## Corrected model/training recipe

reference_model.py copies the reference REVE preprocessing and encoder API (return_output=True, last output token mean). Direct Linear512->1536, image adapter dropout.85, random-initialized residual recovery; no projection dropout, zero-init recovery, or image-side recovery loss. DDP forward includes the adapter.
losses.py calls the actual neuraltrain ClipLoss(norm_kind='y',temperature=False,symmetric=False), two equal losses. Diagonal positives including repeated IDs are deliberately preserved for reference fidelity; this is not the historical multi-positive objective.
reference_config.json: seed33, FP32, AdamW1e-4 all groups, wd.05, OneCycle pct_start.1 cosine, batch256, max40, patience5, single-trial validation top1.
Use nproc_per_node=1 to preserve reference batch256. DDP >1 multiplies effective batch and is NOT an exact training-protocol match. Sampling uses the local DistributedSampler, so update order is not bitwise identical to upstream DataLoader(generator).

After committing and creating an immutable source/config snapshot, run preflight first:

    EEG_CONFIG=/snapshot/reference_config.json torchrun --standalone --nproc_per_node=1 train.py --preflight
    EEG_CONFIG=/snapshot/reference_config.json torchrun --standalone --nproc_per_node=1 train.py

No training or full-cache job was started by this correction.

## Validation and limits

check_reference.py compares all12 uint8 blur outputs exactly, and checks model initialization, forward loss, gradients and inference against checked-out upstream code using a stub backbone. This is CPU code equivalence, not real REVE training validation.
GPU smoke extraction uses 8 actual training/validation-union images and frozen DINO, stored separately in /tmp/eeg_reference_check/reference_smoke. A successful extraction alone does not demonstrate accuracy improvement.
The official clean cache and EEG loader contents on the remote server are unavailable locally. Dataset/split hashes and validation-gallery membership across servers still need comparison: reference validation builds its gallery from all unique seg_ds.triggers filepaths; local validation builds it from actual validation targets. These may differ even when both are called full-gallery top1. Do not claim exact cross-server numerical reproduction until checked.

Completed checks (2026-10-18): CPU stub equivalence PASS; check_reference.py --real PASS with actual pretrained REVE and four cached EEG trials (exact initialization/loss/gradient/inference equality); frozen DINO smoke PASS for eight actual images and all12 scales. Full cache generation and full-batch training preflight have NOT run.
