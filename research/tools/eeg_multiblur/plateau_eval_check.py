"""Real-data distributed diagnostics checked against CPU retrieval."""
import json,os
from pathlib import Path
import torch
import torch.distributed as dist
from data import config
from reve_model import build_model
from evaluate_parallel import evaluate_parallel
cfg=config();root=Path(cfg['run_root']);local=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
torch.manual_seed(cfg['seed'])
model=build_model(json.loads((root/'data/manifest.json').read_text()),cfg).to(device)
(root/'training').mkdir(exist_ok=True)
for split in ('train','val'):
    result=evaluate_parallel(model,root,split,device,check_reference=True,max_per_subject=8,seed=cfg['seed'])
    for group in ('single_trial','subject_aggregated'):
        assert sum(v[group]['count'] for v in result['per_subject'].values())==result[group]['count']
        for metric in ('top1','top5'):
            avg=sum(v[group][metric]*v[group]['count'] for v in result['per_subject'].values())/result[group]['count']
            assert abs(avg-result[group][metric])<1e-12
if dist.get_rank()==0:
    (root/'evaluation_preflight.json').write_text(json.dumps({'passed':True,'cpu_gpu_match':True,'per_subject_aggregation_match':True}))
dist.barrier();dist.destroy_process_group()
