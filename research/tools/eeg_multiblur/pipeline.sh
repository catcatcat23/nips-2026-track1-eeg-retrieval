#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES=${MULTIBLUR_GPUS:-0,1}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN_ROOT=$(python -c 'from data import config;print(config()["run_root"])')
stage=waiting_official_export
trap 'echo "FAILED stage=$stage" > "$RUN_ROOT/status.txt"' ERR
state() { stage=$1; echo "RUNNING stage=$stage" > "$RUN_ROOT/status.txt"; }
state waiting_official_export
while [[ ! -f "$RUN_ROOT/data/complete.json" ]]; do
    if ! kill -0 "$(cat "$RUN_ROOT/prepare.pid")" 2>/dev/null; then
        echo 'Official data export process exited without completion marker'; exit 1
    fi
    sleep 15
done
state smoke_features
torchrun --standalone --nproc_per_node=2 extract.py --smoke > "$RUN_ROOT/smoke_extract.log" 2>&1
state gpu_preflight
torchrun --standalone --nproc_per_node=2 preflight.py > "$RUN_ROOT/preflight.log" 2>&1
state full_features
torchrun --standalone --nproc_per_node=2 extract.py > "$RUN_ROOT/extract.log" 2>&1
python merge_cache.py > "$RUN_ROOT/merge.log" 2>&1
state region_weights
python estimate_weights.py > "$RUN_ROOT/weights.log" 2>&1
state training
torchrun --standalone --nproc_per_node=2 train.py > "$RUN_ROOT/train.log" 2>&1
state export
python -c "from inference import export; export('$RUN_ROOT/training/best.pt','$RUN_ROOT/inference')"
echo COMPLETED > "$RUN_ROOT/status.txt"
