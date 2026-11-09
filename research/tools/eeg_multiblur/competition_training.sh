#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN_ROOT=$(python -c 'from data import config;print(config()["run_root"])')
exec 9>"$RUN_ROOT/controller.lock"
flock -n 9 || exit 1
stage=waiting_120hz_cache
trap 'echo "FAILED stage=$stage" > "$RUN_ROOT/status.txt"' ERR
state() { stage=$1; echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"; }
state waiting_120hz_cache
while [[ ! -f "$RUN_ROOT/data/complete.json" ]]; do
    kill -0 "$(cat "$RUN_ROOT/prepare.pid")" || exit 1
    sleep 15
done
state cache_pairing
python reuse_features.py > "$RUN_ROOT/cache_pairing.log" 2>&1
state gpu_preflight
torchrun --standalone --nproc_per_node=2 preflight.py > "$RUN_ROOT/preflight.log" 2>&1
state region_weights
python estimate_weights.py > "$RUN_ROOT/weights.log" 2>&1
state batch_preflight
torchrun --standalone --nproc_per_node=2 train.py --preflight > "$RUN_ROOT/batch_preflight.log" 2>&1
state training
torchrun --standalone --nproc_per_node=2 train.py > "$RUN_ROOT/train.log" 2>&1
state export
python -c "from inference import export;export('$RUN_ROOT/training/best.pt','$RUN_ROOT/inference')"
echo COMPLETED > "$RUN_ROOT/status.txt"
