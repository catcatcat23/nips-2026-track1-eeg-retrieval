#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN_ROOT=$(python -c 'from data import config;print(config()["run_root"])')
stage=tiny_preflight
trap 'echo "FAILED stage=$stage" > "$RUN_ROOT/status.txt"' ERR
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
python tiny_reve.py > "$RUN_ROOT/tiny_preflight.log" 2>&1
stage=batch_preflight
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=2 train.py --preflight > "$RUN_ROOT/batch_preflight.log" 2>&1
stage=evaluation_preflight
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=2 plateau_eval_check.py > "$RUN_ROOT/evaluation_preflight.log" 2>&1
stage=training
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=2 train.py > "$RUN_ROOT/train.log" 2>&1
echo COMPLETED > "$RUN_ROOT/status.txt"
