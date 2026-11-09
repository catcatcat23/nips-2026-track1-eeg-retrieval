# Experiment storage policy

User instruction (2026-10-23): do not put experiment artifacts on the system disk.

- New experiment directories, EEG/image feature caches, checkpoints, predictions, and large logs must live under `/mnt/DATA-4/anteng/eeg-retrieval/experiments/` or another explicitly approved data disk.
- `/home/Anteng/eeg-retrieval/runs/` may contain symlinks to data-disk runs, not new artifact directories.
- Check available disk space before extracting features or launching training.
- Keep source code and small configuration/documentation files in the workspace.
- Do not use `/tmp` for large artifacts; it shares the system disk.
