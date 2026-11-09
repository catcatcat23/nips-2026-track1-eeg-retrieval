# v9 independent-subject MultiBlur diagnostic

Ten independent MLP models trained from scratch, one per subject. No pooled
weights, no shared optimizer, no gradients across subjects. REVE v7 is untouched.
GPU0 queue: 1,2,4,6,8. GPU1 queue: 3,5,7,9,10. Single GPU per model.
Single-trial EEG retained; no repeat averaging before training. Official v4
120Hz preprocessing and unchanged clean/blur caches reused. No OT/image adapter.
Batch1024 matches the previous global batch; local BatchNorm replaces SyncBN.
Dropout .1/.2; lr1e-4; wd.01; seed21; 2-epoch warmup/cosine to1e-6; max200epochs.
Patience15, counted after warmup; select subject-aggregated clean Top5.

Seven subjects retain existing train/validation splits and the full official
16500-image validation gallery. Subjects3,8,9 lack original validation trials:
20% unique training image IDs are held out with seed21; all repeats of an image
stay in one partition. Their gallery consists of held-out images only; their
validation scores are not comparable to those of the other seven subjects.
No test EEG or labels are used for split construction, optimization, or selection.
Every subject uses its original test EEG with the full200-image test gallery,
only after selecting its checkpoint. Save split indices, policy and counts.

This diagnoses within-subject learning; not a competition-ready submission:
selecting a subject-specific model requires an identity/routing rule unavailable
in an EEG-only interface. Do not report it as unknown-subject generalization.

Preflight: five real single-subject batch1024 forward/backward steps, all trainable
gradients finite, full validation GPU metric agreement with official CPU semantics
for both an existing-split subject and a held-out-image subject. MLP tiny-overfit
was already checked in the preceding experiments; architecture/loss unchanged.
