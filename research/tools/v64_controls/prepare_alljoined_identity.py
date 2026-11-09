"""Create immutable v85-v87 Alljoined controls and a two-GPU sequential queue."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
PARENT=BASE/'reve_v67_alljoined2025_v64recipe'
roots=[]
for version, scope in [(85,'global'),(86,'image'),(87,'concept')]:
    root=BASE/f'reve_v{version}_alljoined_identity_{scope}'
    assert not root.exists(),root
    root.mkdir();roots.append(root)
    shutil.copytree(PARENT/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(Path(__file__).with_name('alljoined_identity.py'),root/'source/alljoined_identity.py')
    trainer=root/'source/alljoined_v67.py'
    trainer.chmod(0o644)
    s=trainer.read_text()
    assert 'total=np.zeros(4)' in s
    s=s.replace('total=np.zeros(4)','total=np.zeros(5)')
    s=s.replace("['clean','blur','local']", "['clean','blur','local','identity']")
    s=s.replace("['total','clean','blur','local']", "['total','clean','blur','local','identity']")
    # Original preflight intentionally tests the unchanged retrieval objective only.
    s=s.replace("parts[k].detach()", "parts[k].detach()")
    ast.parse(s);trainer.write_text(s)
    (root/'predictability').symlink_to(PARENT/'predictability',target_is_directory=True)
    cfg=json.loads((PARENT/'config.json').read_text())
    cfg.update(run_root=str(root),identity_scope=scope,identity_weight=1.,identity_temperature=.1,
               identity_location='REVE pooled512; no new head',parent=str(PARENT),
               experiment=f'v{version}: Alljoined v67 + {scope} identity SupCon',
               only_change=f'{scope} identity loss weight1 temperature0.1; independent aux batch256 for image/concept, original batch for global',
               initialization='fresh pretrained REVE, seed33; no retrieval checkpoint resume')
    (root/'config.json').write_text(json.dumps(cfg,indent=2))
    (root/'protocol.json').write_text(json.dumps(dict(baseline=str(PARENT),scope=scope,
       original_batch=256,auxiliary_batch=0 if scope=='global' else 256,
       auxiliary_layout={'global':'original random retrieval batch','image':'32 images x4 subjects x2 trials','concept':'16 concepts x4 subjects x2 images x2 trials'}[scope],
       identity_weight=1.,temperature=.1,projection='none; normalized pooled512',
       selection='unchanged v67: public-test blur single-trial Top5 early stopping; retain all epochs for aggregate Top5 comparison',
       limitation='public test used for model selection; not independent test. Grouped controls add encoder compute and change positive/negative composition.'),indent=2))
    hashes={str(p.relative_to(root/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').rglob('*.py')}
    (root/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    for name in hashes:(root/'source'/name).chmod(0o444)
    launch='''#!/bin/bash
set -euo pipefail
cd ROOT
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES=$1
export TMPDIR=/mnt/DATA-4/anteng/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/source:${PYTHONPATH:-}"
echo waiting_for_gpu > queue_status.txt
while [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1000 ]; do sleep 30; done
echo gpu_preflight > queue_status.txt
python source/alljoined_identity.py --config config.json --check gpu > identity_gpu.log 2>&1
python source/alljoined_v67.py --config config.json --preflight > preflight.log 2>&1
echo training > queue_status.txt
python source/alljoined_identity.py --config config.json > train.log 2>&1
echo complete > queue_status.txt
'''.replace('ROOT',str(root))
    (root/'launch.sh').write_text(launch)
queue=BASE/'alljoined_identity_queue_20261105'
queue.mkdir(exist_ok=False)
(queue/'gpu2.sh').write_text(f'#!/bin/bash\nbash {roots[0]}/launch.sh 2\na=$?\necho "v85_exit=$a" >> {queue}/status.txt\nbash {roots[2]}/launch.sh 2\necho "v87_exit=$?" >> {queue}/status.txt\n')
print('\n'.join(map(str,roots)))
