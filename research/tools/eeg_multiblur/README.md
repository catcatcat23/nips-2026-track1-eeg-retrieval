# DINOv2 MultiBlur + recovery + auxiliary local OT

This is a **THINGS-EEG2 local experiment**, not a certified competition submission.
The competition's input channel montage/preprocessing contract is still required.
Reference architecture: YanhaoHuang23/EEG-Image-IntraSubject-Retrieval commit
0300630d9ec645e9fbec50d86206f89ac1535f7f (MIT, see UPSTREAM_LICENSE).

## Data and targets

`prepare.py` exports the official seed33 REVE data pipeline without changing
its clean DINOv2 targets. Training seed21 affects optimization, not the fixed
split. EEG repetitions remain individual trials. Manifest and CSV files record
image paths, row order, subjects, channels, sampling and preprocessing settings.
Train/validation use the official timeline split; they may share image identities.
Test image identities must be disjoint. This validation is therefore a matched
baseline protocol, not an unseen-image or unseen-subject validation estimate.

The image manifest uses exact paths rather than assuming directory sort order.
Auxiliary caches cover the union of train and validation images; no test auxiliary
features are needed. Every unblurred extraction batch is compared with the original
clean cache. The original clean cache is never rewritten.

Blur: Resize(518) -> ToTensor -> GaussianBlur -> official HF processor with
`do_rescale=False`. Sigma follows OpenCV's sigma=0 rule; torchvision reflect padding
is used. The official processor may resize/crop again (currently 256 -> 224).
The selected hidden state is `int(0.6667 * (len(hidden_states)-1))`; global mean
includes all tokens exactly like the official extractor. Local features remove
CLS/register tokens and pool the verified square spatial grid to 3x3. Raw features
are averaged across the 12 scales; normalization is performed in the loss.

## Optimization and evaluation

Nonnegative spatial weights are `abs(parameter)` (no `.data` mutation).
Residual recovery is initialized to identity. Global and local temperatures are
0.07. All duplicate image IDs are positive in both directions. Local negatives
are at most 31 different-ID examples, selected by detached clean global similarity.
All same-ID examples remain positive. Local patch selection retains 5 of 9 patches
according to a shared train-only weight vector, and 3 of 6 EEG queries per pair.
Sinkhorn epsilon=0.1, 6 log-space iterations, FP32.

Two A100 GPUs: per-rank batch128, SyncBatchNorm, BF16 forward, FP32 losses,
AdamW 1e-4/0.01, gradient clip1, 2-epoch warmup and cosine to1e-6 over40 epochs.
Full global loss is evaluated on each rank. Autograd gather sums embedding
gradients; DDP averages model gradients, yielding the global-batch reference
without another world-size multiplier. `check_ddp.py` tests this numerically,
including IDs repeated across ranks. DistributedSampler uses set_epoch; the last
incomplete global batch is dropped. Validation is unpadded, complete, rank0-only.

Region weights use one deterministic randomly selected TRAIN trial per image.
Five-fold OOF variance-weighted R² is computed after fold-local EEG scaling/PCA
and image PCA. Negative predictability is clamped; all-unpredictable falls back
to uniform weights. Selected row IDs and fold scores are saved.

Best checkpoint: complete validation candidate-library SINGLE-TRIAL Top1.
Patience5, max40 epochs. Final test runs once for the chosen checkpoint. Metrics
include single-trial and mean prediction per (subject,image) Top1/Top5. OT and
image adapters are never used in validation/test scoring. `inference.py` exports
an EEG-only TorchScript artifact and checks equivalence to the training model;
actual competition-input replay remains pending its input specification.

## Execution

Use a committed, read-only source snapshot for `pipeline.sh`. The pipeline gates
full extraction/training on CPU export completion, real-image cache agreement,
two-rank gradient equivalence, and an 8-real-trial overfit check. Failures stop the
pipeline. `status.txt` distinguishes preparation from actual training.
Runtime root: `/mnt/DATA-4/anteng/eeg-retrieval/experiments/multiblur_v1`.
Python/environment: source `/home/Anteng/eeg-retrieval/scripts/env.sh`.
HF offline mode is mandatory. No Slurm is installed on this host; the launcher
uses `torchrun --standalone --nproc_per_node=2`.

A clean-only same-backbone ablation is recommended before attributing any gain
to MultiBlur, recovery, or OT. This run implements the requested combined model.

## Second run: larger batch, longer optimization

The current config restarts from random initialization (seed21), per-rank batch512,
global batch1024, 200 epochs, **no early stopping** (`patience: null`). AdamW LR
stays1e-4, weight decay0.01, 2-epoch warmup; cosine now spans200 epochs. Both clean
and blur contrastive losses use the global1024-example pool, with all same-image
IDs positive. Local hard negatives remain capped at31.

Use `restart_training.sh` from a new immutable snapshot. It gates training on five
real full-size DDP optimization steps, then starts a fresh process so the preflight
updates do not affect the actual initialization or optimizer. Data/features are
symlinked read-only by convention to v1; no extraction or Ridge fitting repeats.
The previous uniform region-weight fallback remains in effect. Best selection is
still validation-only; the held-out test is evaluated once at the end of this run.
Run root: `/mnt/DATA-4/anteng/eeg-retrieval/experiments/multiblur_v2_b512_e200`.

## Corrected early stopping

The current run is v3_b512_e200_p5: per-rank batch512, maximum200 epochs,
patience5 on validation single-trial Top1. Early stopping is enabled as requested.
The v2 no-early-stop run was stopped near its first epoch; its artifacts are retained.
V3 starts fresh with seed21 and reuses the same data/features.
