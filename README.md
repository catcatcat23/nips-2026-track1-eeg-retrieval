# REVE two-stage EEG retrieval: v39/B epoch 8

This repository reproduces **reve_v39b_epoch8_track1.zip**, not the later Local OT, diffusion, Alljoined, or augmentation experiments.

## Recorded result (200 candidates)

| Output | Single Top-1 | Single Top-5 | Subject-aggregated Top-1 | Subject-aggregated Top-5 |
|---|---:|---:|---:|---:|
| Final clean embedding | 17.72929% | 42.49753% | 66.35% | **93.05%** |
| Frozen stage-1 blur retrieval (separate gallery) | 19.7923% | 46.1939% | 74.30% | 95.65% |

These are local public-test **selection-set** results, not independent test estimates or confirmed platform scores. Both stages selected checkpoints by **clean single-trial Top-5**, not aggregate Top-5. Train and validation trials were combined. Do not substitute a newer checkpoint policy when reproducing this run.

## Exact chain

1. v33/dropout epoch 7, source `7b3a0a28b72a`: pretrained REVE, full-parameter training; fresh heads; image adapter dropout **0.2**, no EEG augmentation, no local OT.
2. Freeze REVE and its 512→1536 projection in eval mode and cache their `h` outputs.
3. v39/B epoch 8, source `f361302dfd765ffa2b9be7386e91c477561202ca`: fit fixed Ridge on train+val only; train a fresh residual correction with the original clean contrastive loss. No extra cosine loss.

EEG input `[B,63,120]` → linear interpolation to 200 points → per-channel temporal z-score (population std, floor 1e-6) → clamp ±15 → REVE → token mean → Linear(512,1536) → h.

Stage 1: image adapter Linear(1536,768) → ELU → Dropout(0.2) → Linear(768,1536), normalized output. Loss = clean ClipLoss + blur ClipLoss, equal weight, one direction, image-side normalization only, no learned temperature. Stage-1 clean head is `ProjectedHead`, without an outer identity skip.

Stage 2: `q=((h-mean)/scale) @ W + bias + 0.1*delta(h)`. Ridge alpha=0.01 on mean-squared raw clean error; statistics and solve use training trials only. Delta: LayerNorm → two residual blocks (1536→3072→1536, GELU, dropout .1, learned block scale initialized .1) → LayerNorm → zero-initialized Linear(1536,1536). Ridge buffers are frozen.

Both stages: seed33, batch256, max40 epochs, early-stop patience5, AdamW max LR1e-4, weight decay .05, OneCycle pct_start .1, gradient clip1, FP32, TF32 off. Stage1 backbone LR is **1e-4**, not the later experiments' 1e-5. Original JSON contains inactive legacy parameters; runtime code and this description define the experiment.

## Data and pretrained assets

Only THINGS-EEG2 is used. Required prepared directory:

```
data/manifest.json
data/{train,val,test}_eeg.npy
data/{train,val,test}_ids.npy
data/{train,val,test}_subjects.npy
data/clean.npy
features/blur.npy
```

Manifest must declare `preprocessing_contract=competition_eeg_image_v1`. EEG preprocessing follows the archived official 120 Hz cache; train+val=668400 trials, test=162240 trials. Preserve IDs, channel order, and image pairing. The model's interpolation to 200 samples does not change the official input contract.

Clean targets are frozen official DINOv2-giant features. Blur cache schema `upstream_per_scale_l2_dino_v1`: original reference RGB resize to 224×224, OpenCV Gaussian kernels `[1,3,9,15,21,27,33,39,45,51,57,63]`, followed by the reference DINO preprocessing, layer fraction .6667, token mean; normalize each scale separately then average. Do not replace clean targets with kernel=1. `cache_reference/` preserves extraction references, not a standalone raw-data download pipeline.

REVE weights `brain-bzh/reve-base` snapshot `dc2a075c287bb2f6c04ee5875bd79535a0f7dba6` and its position bank must be available. `configs/assets.local.json` lists this server's exact artifact locations and checkpoint hashes. On another server, transfer licensed data/weights separately and change only asset paths. Large files are intentionally not committed. This handoff supports reproduction **from prepared caches**, not unattended regeneration from raw EDF and images.

## Commands

Use the recorded Python environment (versions in `evidence/environment.versions.json`), or install `requirements.txt`. CUDA training is required; use one available GPU and do not change batch semantics for an exact comparison.

```bash
python run.py check --assets configs/assets.local.json
CUDA_VISIBLE_DEVICES=0 python run.py stage1 --preflight --output runs/reproduction
CUDA_VISIBLE_DEVICES=0 python run.py stage1 --output runs/reproduction
CUDA_VISIBLE_DEVICES=0 python run.py extract --output runs/reproduction --checkpoint runs/reproduction/stage1/dropout/best.pt
CUDA_VISIBLE_DEVICES=0 python run.py stage2 --output runs/reproduction --checkpoint runs/reproduction/stage1/dropout/best.pt --cache runs/reproduction/cache
```

For stage2-only reproduction using the original epoch7 and cached h: `python run.py stage2 --output runs/stage2_only`. The hash in the h cache must match the supplied checkpoint. Never reuse historical h after retraining stage1. Use a fresh output directory; no silent overwrite or resume.

The CLI exposes only the selected dropout arm and stage2 B arm. Some historical helper class definitions remain to preserve the exact initialization sequence; they are not additional experiment configurations. A fixed-Ridge diagnostic and a discarded tiny overfit check precede stage2 B training.

## Inference and scoring

`submission/submission.py` is the exact packaged inference implementation. Original ZIP SHA is recorded in `evidence/checksums.json`; `evidence/submission_real_replay.json` is the historical real-data replay. It receives EEG only and returns raw `[B,1536]`; no images, adapter, gallery, or OT is used in inference.

For aggregation: group by subject and image, average **raw predictions**, then normalize and rank against all 200 clean candidates. Do not normalize individual predictions before averaging. Blur retrieval is a diagnostic in a separate adapted blur gallery, not the competition output.

## Validation and provenance

`evidence/provenance.json` records copied source origins. `configs/*.original.json` preserve original runtime records; `configs/stage1.json` corrects the stale dropout metadata to .2 without changing actual model behavior. `evidence/validation.json` records the checks performed on this handoff. A complete retraining has not been rerun merely to package the project; numerical reproduction across environments is not guaranteed.

See `HANDOFF_PROMPT.md` for the receiving agent's task. Original experiments and active jobs were not changed by this packaging operation.
