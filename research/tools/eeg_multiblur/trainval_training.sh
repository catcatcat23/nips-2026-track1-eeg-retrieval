#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export EEG_CONFIG="$PWD/trainval_config.json"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl-eeg-reference
RUN_ROOT=$(python -c 'from data import config;print(config()["run_root"])')
stage=batch_preflight
trap 'echo "FAILED stage=$stage" > "$RUN_ROOT/status.txt"' ERR
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=1 train.py --preflight > "$RUN_ROOT/batch_preflight.log" 2>&1
stage=training_public_test_selection
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=1 train.py > "$RUN_ROOT/train.log" 2>&1
echo COMPLETED > "$RUN_ROOT/status.txt"
