#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
ROOT=/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_dinov2_noadapter_sub01_seed21
stage=preflight
trap 'echo "FAILED stage=$stage" > "$ROOT/status.txt"' ERR
echo "RUNNING stage=$stage" > "$ROOT/status.txt"
python no_image_adapter.py preflight --root "$ROOT" > "$ROOT/preflight.log" 2>&1
stage=training
echo "RUNNING stage=$stage" > "$ROOT/status.txt"
python no_image_adapter.py train --root "$ROOT" > "$ROOT/train.log" 2>&1
echo COMPLETED > "$ROOT/status.txt"
