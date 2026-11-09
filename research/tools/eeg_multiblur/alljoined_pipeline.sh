#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/mpl-alljoined
export PYTHONPATH="/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_dinov2_mean_sub01_seed21/deps:${PYTHONPATH:-}"
r=/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v18_alljoined
trap 'echo FAILED > "$r/status.txt"' ERR
while [ ! -f /mnt/DATA-4/anteng/eeg-retrieval/data/Xu2025Alljoined.unpacked.json ]; do sleep 10; done
echo IMAGE_IDENTITY_AUDIT > "$r/status.txt"
if [ ! -f "$r/inventory.json" ]; then python alljoined_prepare.py inventory > "$r/inventory.log" 2>&1; fi
echo DATA_FEATURE_PREPARATION > "$r/status.txt"
python alljoined_prepare.py eeg > "$r/eeg.log" 2>&1 &
e=$!
CUDA_VISIBLE_DEVICES=2 FEATURE_RANK=0 FEATURE_WORLD=2 python alljoined_prepare.py features > "$r/features0.log" 2>&1 &
f=$!
CUDA_VISIBLE_DEVICES=3 FEATURE_RANK=1 FEATURE_WORLD=2 python alljoined_prepare.py features > "$r/features1.log" 2>&1 &
g=$!
failed=0
for p in "$e" "$f" "$g"; do wait "$p" || failed=1; done
if [ "$failed" -ne 0 ]; then echo FAILED > "$r/status.txt"; exit 1; fi
echo GPU_PREFLIGHT > "$r/status.txt"
CUDA_VISIBLE_DEVICES=1 python alljoined_train.py --preflight > "$r/preflight.log" 2>&1
echo TRAINING > "$r/status.txt"
CUDA_VISIBLE_DEVICES=1 python alljoined_train.py > "$r/train.log" 2>&1
echo COMPLETED > "$r/status.txt"
