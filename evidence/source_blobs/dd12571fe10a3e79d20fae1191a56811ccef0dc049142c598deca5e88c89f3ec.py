import os,json,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from reve_model import build_model
from evaluate_parallel import evaluate_parallel,gpu_hits
from evaluate import normalize,topk_accuracy
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl');device=torch.device('cuda',rank)
# Boundary cases: duplicated candidates, all tied, zero vectors, NaNs, uneven shards.
rng=np.random.default_rng(21)
g=rng.normal(size=(11,32)).astype('float32');g[1]=g[0]
p=rng.normal(size=(17,32)).astype('float32');p[:2]=g[0];p[2]=0;p[3]=np.nan
labels=np.arange(17)%11
for dtype in (np.float32,np.float64):
    x=p.astype(dtype);gallery=normalize(g)
    h=gpu_hits(x,gallery,labels,device)
    scores=normalize(x)@gallery.T
    for j,k in enumerate([1,5]):assert h[:,j].mean()==topk_accuracy(scores,labels,k)
root=Path(os.environ.get('EVAL_ROOT','/mnt/DATA-4/anteng/eeg-retrieval/experiments/multiblur_v5_lowdrop_noot'))
c=torch.load(Path(os.environ.get('EVAL_CHECKPOINT',str(root/'training/best.pt'))),map_location='cpu',weights_only=False)
m=build_model(c['manifest'],c['config']).to(device);m.load_state_dict(c['model'])
tick=time.time();result=evaluate_parallel(m,root,'val',device,check_reference=os.environ.get("CPU_REFERENCE","1")=="1")
for group in ('single_trial','subject_aggregated'):
    for key in ('top1','top5','count'):
        assert abs(result[group][key]-c['validation'][group][key])<1e-12,(result,c['validation'])
dist.barrier()
if rank==0:
    report={'passed':True,'checkpoint_epoch':c['epoch'],'full_validation':result,'seconds_including_cpu_reference':time.time()-tick,'edge_cases':'ties, NaN, zero vectors, FP32 and FP64, uneven query counts'}
    Path(os.environ['EVAL_CHECK_REPORT']).write_text(json.dumps(report,indent=2))
    print(report,flush=True)
dist.destroy_process_group()
