#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
ROOT=/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_dinov2_mean_sub01_seed21
export PYTHONPATH="$ROOT/deps${PYTHONPATH:+:$PYTHONPATH}"
stage=prepare
trap 'echo "FAILED stage=$stage" > "$ROOT/status.txt"' ERR
setstage() { stage=$1; echo "RUNNING stage=$stage" > "$ROOT/status.txt"; }
setstage prepare
python dino_image_ablation.py prepare --root "$ROOT" > "$ROOT/prepare.log" 2>&1
setstage gpu_smoke
CUDA_VISIBLE_DEVICES=0 python dino_image_ablation.py extract --root "$ROOT" --rank 0 --smoke > "$ROOT/smoke0.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python dino_image_ablation.py extract --root "$ROOT" --rank 1 --smoke > "$ROOT/smoke1.log" 2>&1 &
p1=$!
wait "$p0"
wait "$p1"
setstage feature_extraction
CUDA_VISIBLE_DEVICES=0 python dino_image_ablation.py extract --root "$ROOT" --rank 0 > "$ROOT/extract0.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python dino_image_ablation.py extract --root "$ROOT" --rank 1 > "$ROOT/extract1.log" 2>&1 &
p1=$!
wait "$p0"
wait "$p1"
setstage merge
python dino_image_ablation.py merge --root "$ROOT" > "$ROOT/merge.log" 2>&1
setstage model_preflight
CUDA_VISIBLE_DEVICES=0 python dino_image_preflight.py "$ROOT" > "$ROOT/model_preflight.log" 2>&1
setstage training
CUDA_VISIBLE_DEVICES=0 python dino_image_ablation.py train --root "$ROOT" > "$ROOT/train.log" 2>&1
echo COMPLETED > "$ROOT/status.txt"
