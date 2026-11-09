from pathlib import Path
import json,hashlib,subprocess
import torch
from braindecode.models.reve import RevePositionBank
import braindecode.models.reve as reve

project=Path('/home/Anteng/eeg-retrieval');out=project/'submissions/reve_seed33'
out.mkdir(parents=True, exist_ok=True)
source=Path(reve.__file__).read_text()
classes=source[source.index('class GEGLU('):source.index('class RevePositionBank(')]
header='''# REVE architecture adapted from Braindecode (BSD-3-Clause), Jonathan Lys.
# See LICENSE_BRAINDЕCODE.txt. All inference code is contained in this file.
import json
import math
import logging
from pathlib import Path
from typing import Union
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange
from benchmark_utils.base_solver import CompetSolver
HAS_SDPA = True
logger = logging.getLogger(__name__)

'''.replace('BRAINDЕCODE','BRAINDECODE')
(out/'submission.py').write_text(header+classes+(project/'scripts/reve_submission/wrapper_tail.py').read_text())
checkpoints=[p for p in Path('/mnt/DATA-4/anteng/eeg-retrieval/results').rglob('best.ckpt') if '942-a28caefc' in str(p)]
assert len(checkpoints)==1
cp=checkpoints[0];c=torch.load(cp,map_location='cpu',weights_only=True);s=c['state_dict'];weights={}
for k,v in s.items():
 prefix='model.wrapped_model.model.'
 if k.startswith(prefix) and not k.startswith(prefix+'final_layer.'):
  weights[k[len(prefix):]]=v
 elif k.startswith('model.probe.'):
  weights['head.'+k[len('model.probe.'):]]=v
assert len(weights)>100
torch.save(weights,out/'weights.pt')
bank=RevePositionBank();positions={k:bank.embedding[i].tolist() for k,i in bank.mapping.items()}
(out/'positions.json').write_text(json.dumps(positions))
manifest=json.loads(Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/multiblur_v1/data/manifest.json').read_text())
actual=torch.tensor([positions[k] for k in manifest['channels']]);assert torch.equal(actual,s['model.wrapped_model.bank_positions'])
metadata={'seed':33,'checkpoint':str(cp),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'best_epoch_zero_based':c['epoch'],'selection':'first predeclared seed33; checkpoint selected by validation, not test','competition_git_sha':subprocess.check_output(['git','-C',str(project/'competition-2026'),'rev-parse','HEAD'],text=True).strip(),'target_dim':1536,'training_sampling_rate':200,'training_window':[-.2,.8],'competition_sampling_rate':120,'bridge':'official REVE-probe linear interpolation to200Hz, window/channel standardization and clamp15','limitation':'Platform preprocessing differs from NeuralBench REVE training; local baseline scores do not certify ZIP scores. No hidden-cohort evaluation.'}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
licenses=list(Path(reve.__file__).parents[2].glob('braindecode-*.dist-info/licenses/LICENSE*'))
assert licenses, 'Locate upstream license before redistribution'
(out/'LICENSE_BRAINDECODE.txt').write_bytes(licenses[0].read_bytes())
print('Built',out,'parameters',sum(x.numel() for x in weights.values()))
