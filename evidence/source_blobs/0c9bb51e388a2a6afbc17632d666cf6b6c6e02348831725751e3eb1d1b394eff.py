import argparse,json,os,random,time,math
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,ConcatDataset
from reve_model import build_model,optimizer_groups
from losses import objective
from data import config,DatasetEEG
from evaluate import evaluate
from evaluate_parallel import evaluate_parallel
from plateau_schedule import WarmupPlateau

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true',help='Run 5 real full-batch steps, without validation or checkpoints')
    args=parser.parse_args()
    cfg=config();root=Path(cfg['run_root']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if cfg.get('recipe')=='reference_94b3a31':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    manifest=json.loads((root/'data/manifest.json').read_text())
    model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(build_model(manifest,cfg)).to(device)
    initial=None
    if cfg.get('warm_start'):
        initial=torch.load(cfg['warm_start'],map_location='cpu',weights_only=False)
        assert initial['manifest']['channels']==manifest['channels']
        assert initial['manifest']['input_shape']==manifest['input_shape']
        model.load_state_dict(initial['model'])
    model=DDP(model,device_ids=[local])
    if cfg.get('required_feature_schema'):
        cache_meta=json.loads((root/'features/complete.json').read_text())
        assert cache_meta.get('schema')==cfg['required_feature_schema'], 'Incompatible blur cache; regenerate with extract_reference.py'
    training_splits=cfg.get('training_splits',['train'])
    assert training_splits and set(training_splits)<= {'train','val'} and len(set(training_splits))==len(training_splits)
    datasets=[DatasetEEG(root,s,aux=not cfg.get('clean_only',False),local_aux=bool(cfg['local_weight'])) for s in training_splits]
    ds=datasets[0] if len(datasets)==1 else ConcatDataset(datasets)
    selection_split=cfg.get('selection_split','val')
    assert selection_split in ('val','test')
    if 'val' in training_splits:assert selection_split=='test'
    selection_key='selection' if selection_split=='test' else 'validation'
    sampler=DistributedSampler(ds,shuffle=True,seed=cfg['seed'],drop_last=True)
    loader=DataLoader(ds,batch_size=cfg['batch_size'],sampler=sampler,num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True)
    optim=torch.optim.AdamW(optimizer_groups(model.module,cfg),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    total=(cfg['epochs']-(initial['epoch'] if initial is not None else 0))*len(loader);warm=cfg['warmup_epochs']*len(loader)
    def factor(step):
        if step<warm:return (step+1)/max(1,warm)
        t=(step-warm)/max(1,total-warm);minimum=cfg['min_lr']/cfg['lr']
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*t))
    plateau=cfg.get('scheduler')=='warmup_plateau'
    if cfg.get('scheduler')=='onecycle':
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optim,max_lr=[cfg['encoder_lr'],cfg['lr']] if cfg.get('encoder')=='reve_finetune' else cfg['lr'],total_steps=total,pct_start=.1,anneal_strategy='cos')
    else:
        scheduler=WarmupPlateau(optim,warm,cfg) if plateau else torch.optim.lr_scheduler.LambdaLR(optim,factor)
    weights=torch.from_numpy(np.load(root/'features/weights.npy')).to(device) if cfg['local_weight'] else None
    best=-1.;bad=0;out=root/'training';out.mkdir(exist_ok=True)
    if rank==0 and not args.preflight:
        (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'status.json').write_text(json.dumps({'state':'training_running','world_size':dist.get_world_size()}))
    start_epoch=initial['epoch'] if initial is not None else 0
    if initial is not None and not args.preflight:
        best=initial['validation'][cfg['selection_group']][cfg['selection_metric']]
        if rank==0:
            torch.save(initial,out/'best.pt')
            (out/'restart.json').write_text(json.dumps({'type':'weights_only_optimizer_reset','source':cfg['warm_start'],'source_epoch':start_epoch,'bad_epochs_reset':True}))
    started=time.time()
    for epoch in range(start_epoch,cfg['epochs']):
        sampler.set_epoch(epoch);model.train();loss_sum=0.;part_sum={k:0. for k in ('clean','blur','restore','local')};tick=time.time()
        for step,b in enumerate(loader):
            eeg=b['eeg'].to(device,non_blocking=True);blur=b['blur'].to(device) if 'blur' in b else None;local_v=b['local'].to(device) if cfg['local_weight'] else None
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg.get('recipe')!='reference_94b3a31'):outputs=model(eeg,blur,local_v)
            with torch.autocast('cuda',enabled=False):loss,parts=objective(outputs,b['clean'].to(device),blur,b['id'].to(device),weights,cfg)
            assert torch.isfinite(loss)
            loss.backward()
            if args.preflight:
                for name,p in model.named_parameters():
                    if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),name
                if cfg.get('encoder') == 'reve_finetune':
                    assert sum(p.grad.float().abs().sum().item() for p in model.module.encoder.parameters() if p.requires_grad) > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optim.step();scheduler.step();loss_sum+=loss.item()
            for k,v in parts.items():part_sum[k]+=v
            if rank==0 and step%100==0:print(json.dumps({'epoch':epoch+1,'step':step,'steps':len(loader),'loss':loss.item(),'parts':parts,'lr':scheduler.get_last_lr()}),flush=True)
            if args.preflight and step==4:
                torch.cuda.synchronize()
                memory=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=device)
                dist.all_reduce(memory,op=dist.ReduceOp.MAX)
                if rank==0:
                    report={'passed':True,'steps':5,'training_splits':training_splits,'training_samples':len(ds),'selection_split':selection_split,'per_gpu_batch':cfg['batch_size'],'world_size':dist.get_world_size(),'max_allocated_gib':memory.item(),'final_loss':loss.item(),'seconds':time.time()-started}
                    (root/'batch_preflight.json').write_text(json.dumps(report,indent=2))
                    print('BATCH PREFLIGHT PASSED',report,flush=True)
                dist.barrier();dist.destroy_process_group();return
        dist.barrier();stop=torch.zeros(1,device=device)
        metrics=evaluate_parallel(model.module,root,selection_split,device)
        diagnostic=None
        if cfg.get('diagnostic_per_subject'):
            diagnostic=evaluate_parallel(model.module,root,'train',device,max_per_subject=cfg['diagnostic_per_subject'],seed=cfg['seed'])
        score=metrics[cfg['selection_group']][cfg['selection_metric']]
        if plateau:scheduler.validation_step(score)
        if rank==0:
            score=metrics[cfg['selection_group']][cfg['selection_metric']]
            row={'epoch':epoch+1,'loss':loss_sum/len(loader),selection_key:metrics,'selection_split':selection_split,'train_diagnostic':diagnostic,'parts':{k:v/len(loader) for k,v in part_sum.items()},'lr':scheduler.get_last_lr(),'epoch_seconds':time.time()-tick,'elapsed_seconds':time.time()-started}
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('PUBLIC TEST SELECTION' if selection_split=='test' else 'VALIDATION',json.dumps(row),flush=True)
            if score>best:
                best=score;bad=0
                checkpoint={'model':model.module.state_dict(),'epoch':epoch+1,'config':cfg,'manifest':manifest,selection_key:metrics,'selection_split':selection_split}
                torch.save(checkpoint,out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
            else:bad+=1
            latest={'model':model.module.state_dict(),'optimizer':optim.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'bad_epochs':bad,'config':cfg,'manifest':manifest,selection_key:metrics,'selection_split':selection_split}
            torch.save(latest,out/'last.tmp');os.replace(out/'last.tmp',out/'last.pt')
            if cfg['patience'] is not None and bad>=cfg['patience']:stop.fill_(1)
        dist.broadcast(stop,0)
        if stop.item():break
    dist.barrier()
    c=torch.load(out/'best.pt',map_location=device,weights_only=False);model.module.load_state_dict(c['model'])
    result=evaluate_parallel(model.module,root,'test',device)
    if rank==0:
        result_file='public_test_selection_metrics.json' if selection_split=='test' else 'test_metrics.json'
        if selection_split=='test':result['evaluation_role']='model_selection_not_independent_test'
        (out/result_file).write_text(json.dumps(result,indent=2))
        (out/'status.json').write_text(json.dumps({'state':'model_selection_complete' if selection_split=='test' else 'test_complete','best_epoch':c['epoch'],'elapsed_seconds':time.time()-started}))
        print('MODEL SELECTION COMPLETE' if selection_split=='test' else 'TEST COMPLETE',result,flush=True)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
