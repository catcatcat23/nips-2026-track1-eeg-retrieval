#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/Anteng/eeg-retrieval/scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN_ROOT=$(python -c 'import json;print(json.load(open("subject_config.json"))["run_root"])')
queue() {
    local gpu=$1
    shift
    export CUDA_VISIBLE_DEVICES=$gpu
    for subject in "$@"; do
        mkdir -p "$RUN_ROOT/subject_$subject"
        if ! python subject_train.py --subject "$subject" > "$RUN_ROOT/subject_$subject/train.log" 2>&1; then
            echo "FAILED subject=$subject GPU=$gpu" > "$RUN_ROOT/queue_$gpu.status"
            return 1
        fi
    done
    echo COMPLETED > "$RUN_ROOT/queue_$gpu.status"
}
echo RUNNING > "$RUN_ROOT/status.txt"
queue 0 1 2 4 6 8 &
first=$!
queue 1 3 5 7 9 10 &
second=$!
failed=0
wait "$first" || failed=1
wait "$second" || failed=1
if ((failed)); then echo FAILED > "$RUN_ROOT/status.txt"; exit 1; fi
python - <<'PY'
import json
from pathlib import Path
root=Path(json.load(open('subject_config.json'))['run_root'])
rows={str(i):json.loads((root/f'subject_{i}/test_metrics.json').read_text()) for i in range(1,11)}
mean={group:{metric:sum(r[group][metric] for r in rows.values())/10 for metric in ['top1','top5']} for group in ['single_trial','subject_aggregated']}
(root/'summary.json').write_text(json.dumps({'per_subject':rows,'macro_test':mean,'test_candidates':200,'note':'Validation galleries differ for subjects 3,8,9; do not macro-average validation scores.'},indent=2))
PY
echo COMPLETED > "$RUN_ROOT/status.txt"
