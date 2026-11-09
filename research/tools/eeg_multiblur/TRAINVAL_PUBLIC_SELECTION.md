# v14: combined training data with public-test model selection

User explicitly confirmed this protocol. Training uses ConcatDataset(original train, original val): 534720+133680=668400 trials. Original test EEG never enters the optimizer; its 200-image gallery is used every epoch for checkpoint selection and patience5 early stopping on single_trial.top5. It is therefore NOT an independent test score. Final scores are written to public_test_selection_metrics.json with an evaluation_role label; training history uses selection rather than validation.

Fresh pretrained REVE initialization, seed33; all other recipe settings from v13: FP32, batch256 on one GPU, AdamW peak1e-4/wd.05, OneCycle pct_start.1, max40epochs. Image caches and clean/EEG arrays are reused read-only; concatenation does not copy them. Training diagnostic subset remains sampled from original train only.

CPU checks passed: total count668400, concatenation boundary EEG-target pairs, auxiliary-cache coverage and no test image overlap. Launch requires five real full-batch GPU preflight steps before training. Historical v13 snapshots and results are unchanged.
