#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export EEG_CONFIG="$PWD/reference_config.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/mpl-eeg-reference
export PYTHONPATH="/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_dinov2_mean_sub01_seed21/deps:${PYTHONPATH:-}"
RUN_ROOT=$(python -c 'from data import config;print(config()["run_root"])')
stage=smoke_extraction
trap 'echo "FAILED stage=$stage" > "$RUN_ROOT/status.txt"' ERR
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=4 extract_reference.py --smoke > "$RUN_ROOT/smoke_extraction.log" 2>&1
stage=tiny_preflight
export CUDA_VISIBLE_DEVICES=0
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
python reference_preflight.py > "$RUN_ROOT/tiny_preflight.log" 2>&1
stage=full_extraction
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=4 extract_reference.py > "$RUN_ROOT/extraction.log" 2>&1
stage=cache_merge
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
python merge_reference.py > "$RUN_ROOT/cache_merge.log" 2>&1
export CUDA_VISIBLE_DEVICES=0
stage=batch_preflight
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=1 train.py --preflight > "$RUN_ROOT/batch_preflight.log" 2>&1
stage=evaluation_preflight
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=1 plateau_eval_check.py > "$RUN_ROOT/evaluation_preflight.log" 2>&1
stage=training
echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"
torchrun --standalone --nproc_per_node=1 train.py > "$RUN_ROOT/train.log" 2>&1
echo COMPLETED > "$RUN_ROOT/status.txt"
