#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl-things-adapt
r=/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v19_things_adapt
stage1=/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v18_alljoined/status.txt
trap 'echo FAILED > "$r/status.txt"' ERR
echo PREFLIGHT > "$r/status.txt"
python things_adapt.py --preflight > "$r/preflight.log" 2>&1
echo WAITING_FOR_STAGE1 > "$r/status.txt"
while true; do
 state=$(cat "$stage1")
 if [ "$state" = COMPLETED ]; then break; fi
 if [ "$state" = FAILED ]; then echo BLOCKED_STAGE1_FAILED > "$r/status.txt"; exit 1; fi
 sleep 30
done
echo TRAINING > "$r/status.txt"
python things_adapt.py > "$r/train.log" 2>&1
echo COMPLETED > "$r/status.txt"
