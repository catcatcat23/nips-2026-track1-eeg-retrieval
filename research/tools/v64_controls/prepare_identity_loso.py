from pathlib import Path
import shutil,json,hashlib,ast
base=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments');old=base/'reve_v79_v64_loso';out=base/'reve_v91_identity_loso_calibration'
assert not out.exists();out.mkdir();shutil.copytree(old/'source',out/'source')
p=out/'source/extensions.py';p.chmod(0o644);s=p.read_text();s=s.replace('self.identity_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512,256), nn.GELU(), nn.Linear(256,128))','self.identity_head = nn.Identity()');s=s.replace("+(['identity_head'] if cfg.get('identity_weight') else [])",'+[]');ast.parse(s);p.write_text(s)
shutil.copyfile('tools/v64_controls/loso_calibration.py',out/'source/loso_calibration.py')
for fold in [1,2]:
 name=f'fold_{fold:02d}';dst=out/name;dst.mkdir();parent=old/name
 for n in ['local.npy','local_complete.json','predictability']:(dst/n).symlink_to((parent/n).resolve())
 cfg=json.loads((parent/'config.json').read_text());cfg.update(run_root=str(dst),identity_weight=1.,identity_temperature=.1,identity_location='pooled512',experiment='v91 THINGS LOSO direct pooled identity weight1',only_change='add global subject contrast on same random retrieval batch; no new head; heldout excluded')
 (dst/'config.json').write_text(json.dumps(cfg,indent=2))
 common=f'''#!/bin/bash
set -euo pipefail
source /home/Anteng/eeg-retrieval/scripts/env.sh
export CUDA_VISIBLE_DEVICES={fold-1} PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/mnt/DATA-4/anteng/tmp OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH={out}/source:/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_dinov2_mean_sub01_seed21/deps
trap 'echo failed > {dst}/queue_status.txt' ERR
echo preflight > {dst}/queue_status.txt
python -u {out}/source/run_control.py --config {dst}/config.json --preflight > {dst}/preflight.log 2>&1
echo training > {dst}/queue_status.txt
python -u {out}/source/run_control.py --config {dst}/config.json > {dst}/train.log 2>&1
echo calibrating > {dst}/queue_status.txt
python -u {out}/source/loso_calibration.py --config {dst}/config.json --out {dst}/calibration_identity > {dst}/calibration_identity.log 2>&1
echo complete > {dst}/queue_status.txt
'''
 (dst/'launch.sh').write_text(common)
 baseline=common[:common.index("trap '")].replace(f'CUDA_VISIBLE_DEVICES={fold-1}',f'CUDA_VISIBLE_DEVICES={fold+1}')+f'python -u {out}/source/loso_calibration.py --config {parent}/config.json --out {dst}/calibration_baseline > {dst}/calibration_baseline.log 2>&1\n'
 (dst/'baseline.sh').write_text(baseline)
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').glob('*.py')}
for p in (out/'source').glob('*.py'):ast.parse(p.read_text());p.chmod(0o444)
(out/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
(out/'protocol.json').write_text(json.dumps(dict(baseline=str(old),folds=[1,2],identity='global subject supervised contrast directly pooled512, weight1 temperature.1; same batch256',selection='source-subject image-disjoint validation only; best blur single Top5 across raw/EMA',calibration='50/200/1000 unique target training images, one trial per image; frozen encoder and heads, rank64 residual on h; 200 AdamW updates, no test selection',status='prepared'),indent=2))
print(out)
