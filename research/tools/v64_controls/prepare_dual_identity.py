"""Create a new immutable v81 snapshot without touching existing runs."""
import hashlib,json,shutil
from pathlib import Path
root=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
parent=root/'reve_v80_v64_identity_direct';out=root/'reve_v81_grouped_identity_dualbatch'
assert not (out/'stage1').exists(), out
if not out.exists():
 out.mkdir();shutil.copytree(parent/'source',out/'source')
for name in ['local.npy','local_complete.json','predictability']:
 if not (out/name).exists(): (out/name).symlink_to((parent/name).resolve())
c=json.loads((parent/'config.json').read_text());c.update(run_root=str(out),experiment='v81: random retrieval batch256 + grouped identity-only batch256',only_change='separate grouped image-conditioned subject loss; no identity loss on retrieval batch',identity_positives='same image AND same subject; negatives same image different subjects; other images excluded',auxiliary_batch_size=256,auxiliary_images=32,auxiliary_subjects=4,auxiliary_trials=2,auxiliary_augmentation='none; original single trials',optimizer_update='one step after both backward passes; summed losses, no division by two')
(out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
shutil.copy2('tools/v64_controls/grouped_identity.py',out/'source/grouped_identity.py')
p=out/'source/local_ot_train_ema.py';p.chmod(0o644);s=p.read_text();s=s.replace('import local_ot_train\n','import local_ot_train\nfrom grouped_identity import auxiliary_backward\n')
needle='assert torch.isfinite(loss);loss.backward()'
assert s.count(needle)==1
s=s.replace(needle,needle+"\n   identity=auxiliary_backward(model,cfg);parts['identity']=identity;loss=loss.detach()+cfg['identity_weight']*identity")
p.write_text(s)
p=out/'source/run_control.py';p.chmod(0o644);s=p.read_text();needle='    base.LocalOTModel=cls;'
assert needle in s
s=s.replace(needle,"    def retrieval_loss(model,batch,weights):\n        value,parts=ex.original_loss(model,batch,weights)\n        parts['identity']=value.new_zeros(())\n        model.pooled_for_identity=None\n        return value,parts\n    ex.loss_fn=retrieval_loss\n"+needle)
p.write_text(s)
sha={str(p.relative_to(out/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
(out/'source_sha256.json').write_text(json.dumps(sha,indent=2))
launch='''#!/bin/bash
set -euo pipefail
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=RUNROOT/source:/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_dinov2_mean_sub01_seed21/deps
cd RUNROOT
trap 'echo failed > queue_status.txt' ERR
echo waiting_for_gpu2 > queue_status.txt
while true; do
 used=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
 if [ "$used" -lt 1000 ]; then break; fi
 sleep 60
done
echo preflight > queue_status.txt
python -u source/run_control.py --config config.json --preflight > preflight.log 2>&1
python -u source/grouped_identity.py --config config.json > grouped_preflight.log 2>&1
echo training > queue_status.txt
python -u source/run_control.py --config config.json > train.log 2>&1
echo complete > queue_status.txt
'''.replace('RUNROOT',str(out))
(out/'launch.sh').write_text(launch)
print(out)
