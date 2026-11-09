# v64 matched controls and downstream LOSO

| Run | Purpose | Only experimental change | Protocol |
|---|---|---|---|
| v77_v64_mean4 | Test 100% within-subject four-trial averaging | Replace 50/25/25 repeat mix with four distinct same-subject/same-image trials for every input | Original v64 train+val and public-test selection |
| v78_v64_identity | Test whether subject auxiliary supervision helps retrieval | Add pooled512 -> LN -> Linear256 -> GELU -> Linear128 and supervised subject contrastive loss, weight0.1 temperature0.1 | Original v64 augmentation and public-test selection |
| v79_v64_loso | Measure generalization to unseen downstream subjects | Ten independent THINGS leave-one-subject-out folds | Heldout subject excluded from training/selection; fixed image-disjoint validation |

All use v64 pretrained initialization, full-parameter REVE tuning, clean+blur+0.1 Local OT, image adapter dropout0.2, 20% one-channel masking, batch256, AdamW1e-4 weight_decay0.05, OneCycle40 epochs pct_start0.1, early stopping5, FP32 TF32off, raw plus EMA0.999/0.9998. v77/v78 retain every epoch and select blur single-trial Top5 exactly as v64. These are new stage1 training runs, not v64-weight continuation. Checkpoints/logs/cache links live exclusively on DATA-4.

Identity positives are other rows belonging to the same subject, negatives other subjects, self excluded. The loss averages log probability over all same-subject positives. It reaches the shared REVE through a separate 165248-parameter projection; it does not require subject identity during inference. The new head initialization preserves the base RNG state and all v64 initial parameters. Improving identity separation is a hypothesis, not a guaranteed retrieval improvement.

LOSO original local train/val overlap in image identity (16500 images), so they are combined and redivided by image: seed33 selects1654 of16540 images for validation;14886 remain for training. Each fold uses only nine seen subjects for both subsets. The heldout subject's original200 test images are evaluated once after selecting variant and epoch on seen-subject validation blur single Top5. Regional predictability weights are reestimated independently per subject on the14886 training images only; heldout-subject weights are never accessed by training or test scoring. The shared weight-preparation job computes independent per-subject entries; there is no EEG pooling across subjects. This is downstream LOSO, not a statement that REVE foundation pretraining excluded these people. Report each fold and subject-macro mean, not a single fold as final LOSO performance.

Queues: GPU3 v75 -> v77 -> v79 folds01..10; GPU2 v76 -> v78. Existing jobs are not interrupted. Each new training run first executes real-data forward/backward and tiny-set overfit preflight; failure stops that queue. Waiting does not mean training has begun.

Build: run prepare.py then fix_loso_split.py BEFORE launching (these intentionally fail rather than overwrite existing run roots). verify_snapshots.py verifies hashes and ten fold splits. run_control.py wraps the frozen v64 training loop; extensions.py holds new sampling, loss and evaluation logic. prepare.py extends training loss logging with the identity term, including zero for other arms. summarize_loso.py creates the ten-fold result only after all heldout tests exist.

Validation performed: syntax, real-cache minimum repeats and train/val/test identity audit, four-trial exact pairing, subject loss finite gradients/permutation/no-positive behavior, base-weight and RNG initialization parity. GPU preflight remains pending until assigned GPU is idle. Scores are not yet available.
