#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/mpl-eeg-recovery
RUN_ROOT=$(python -c 'import json;print(json.load(open("recovery_ablation_config.json"))["run_root"])')
trap 'echo FAILED > "$RUN_ROOT/status.txt"' ERR
echo GPU_PREFLIGHT > "$RUN_ROOT/status.txt"
CUDA_VISIBLE_DEVICES=1 python recovery_ablation.py preflight > "$RUN_ROOT/preflight.log" 2>&1
echo FEATURE_EXTRACTION > "$RUN_ROOT/status.txt"
CUDA_VISIBLE_DEVICES=1 python recovery_ablation.py extract > "$RUN_ROOT/extraction.log" 2>&1
echo HEAD_TRAINING > "$RUN_ROOT/status.txt"
CUDA_VISIBLE_DEVICES=1 python recovery_ablation.py train --arm A > "$RUN_ROOT/A.log" 2>&1 &
a=$!
CUDA_VISIBLE_DEVICES=2 python recovery_ablation.py train --arm B > "$RUN_ROOT/B.log" 2>&1 &
b=$!
CUDA_VISIBLE_DEVICES=3 python recovery_ablation.py train --arm D > "$RUN_ROOT/D.log" 2>&1 &
d=$!
printf '{"A":%s,"B":%s,"D":%s}\n' "$a" "$b" "$d" > "$RUN_ROOT/pids.json"
failed=0
for p in "$a" "$b" "$d"; do wait "$p" || failed=1; done
if [ "$failed" -ne 0 ]; then echo FAILED > "$RUN_ROOT/status.txt"; exit 1; fi
echo COMPLETED > "$RUN_ROOT/status.txt"
