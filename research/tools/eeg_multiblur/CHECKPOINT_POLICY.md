# Checkpoint policy (2026-10-22)

User requested retaining the top 3 models by blur Top-5 for future experiments.
Until clarified, retain BOTH subject-aggregated and single-trial blur Top-5 rankings.
Use pure global retrieval scores, excluding OT reranking. Record the metric group.
Keep requested epoch checkpoints even when outside the Top-3. Keep a resumable
latest checkpoint with optimizer, scheduler and RNG states. Never infer retained
weights from history metrics alone.

v53 repeats v43 from original pretrained REVE and reuses feature caches. It saves
all epoch inference weights and dual top3.json rankings, monitors aggregate blur
Top-5 with patience=5, and runs at least 9 epochs. Its second stage explicitly uses
new epoch 9, not the highest-ranked epoch. Stage2 only updates the B recovery head;
blur is fixed, so stage2 selection remains clean single-trial Top-5.
