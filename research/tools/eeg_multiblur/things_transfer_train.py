"""Stage two: adapt the completed mixed-data checkpoint to THINGS only."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,ConcatDataset
from data import DatasetEEG
from alljoined_replay_train import MixedModel,evaluate
from neuraltrain.losses.losses import ClipLoss


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    cfg=json.loads(Path(__file__).with_name('things_transfer_config.json').read_text())
    root=Path(cfg['run_root']);root.mkdir(parents=True,exist_ok=True)
    src=Path(cfg['source_checkpoint'])
    source=torch.load(src,map_location='cpu',weights_only=False)
    if not args.preflight:
        assert Path(cfg['stage1_status']).read_text().strip()=='COMPLETED'
        assert source['epoch']>0,'Stage1 retained epoch0 baseline: no learned mixed-data checkpoint available; do not mislabel it as two-stage training'
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
    base=torch.load(cfg['base_checkpoint'],map_location='cpu',weights_only=False)
    head=torch.load(cfg['head_checkpoint'],map_location='cpu',weights_only=False)
    model=MixedModel(base['manifest'],source['channels'],base['model'],head['head']).cuda()
    model.load_state_dict(source['model'],strict=True)
    ds=ConcatDataset([DatasetEEG(cfg['data_root'],s,aux=True,local_aux=False) for s in ['train','val']])
    generator=torch.Generator().manual_seed(cfg['seed'])
    loader=DataLoader(ds,batch_size=256,shuffle=True,generator=generator,num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True)
    encoder=list(model.encoder.parameters());others=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]
    opt=torch.optim.AdamW([{'params':encoder,'lr':cfg['encoder_lr']},{'params':others,'lr':cfg['lr']}],weight_decay=.05)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5,min_lr=[3e-7,3e-6])
    criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False).cuda()
    if args.preflight:
        model.train()
        for step,b in enumerate(loader):
            opt.zero_grad(set_to_none=True);h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),'things')
            loss=criterion(q,b['clean'].cuda())+criterion(h,a);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step==2:break
        (root/'preflight.json').write_text(json.dumps({'passed':True,'steps':3,'source_epoch':source['epoch'],'note':'Smoke updates discarded. Full training reloads completed stage1 best with a fresh optimizer.','final_loss':float(loss.detach()),'training_samples':len(ds),'batch_size':256},indent=2))
        print('STAGE2 PREFLIGHT PASSED',flush=True);return
    assert not (root/'history.jsonl').exists()
    # Preserve the exact source selected by stage1 even if that directory later changes.
    torch.save(source,root/'stage1_selected.pt')
    (root/'provenance.json').write_text(json.dumps({'source':str(src),'source_epoch':source['epoch'],'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'source_selection':source['selection'],'training_samples':len(ds)},indent=2))
    selection=evaluate(model)
    for group in ['single_trial','subject_aggregated']:
        for k in ['top1','top5']:assert abs(selection[group][k]-source['selection'][group][k])<1e-5
    best=selection['single_trial']['top5'];stale=0
    def checkpoint(epoch,selection,name='best.pt'):
        torch.save({'model':model.state_dict(),'epoch':epoch,'stage1_epoch':source['epoch'],'config':cfg,'channels':source['channels'],'manifest':source['manifest'],'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'selection':selection},root/name)
    checkpoint(0,selection)
    for epoch in range(1,cfg['epochs']+1):
        model.train();start=time.time();totals=np.zeros(2)
        for step,b in enumerate(loader):
            opt.zero_grad(set_to_none=True);h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),'things')
            lc=criterion(q,b['clean'].cuda());lb=criterion(h,a);loss=lc+lb
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            totals += [float(lc.detach()),float(lb.detach())]
            if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'clean':float(lc.detach()),'blur':float(lb.detach())}),flush=True)
        selection=evaluate(model);score=selection['single_trial']['top5'];sched.step(score)
        checkpoint(epoch,selection,'last.pt')
        if score>best:best=score;stale=0;checkpoint(epoch,selection)
        else:stale+=1
        row={'epoch':epoch,'clean':totals[0]/len(loader),'blur':totals[1]/len(loader),'selection':selection,'best_top5':best,'stale':stale,'lr':[g['lr'] for g in opt.param_groups],'seconds':time.time()-start}
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        (root/'status.json').write_text(json.dumps({'state':'training','epoch':epoch,'best_top5':best}))
        print(json.dumps(row),flush=True)
        if stale>=cfg['patience']:break
    ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    (root/'result.json').write_text(json.dumps({'best_epoch':ck['epoch'],'stage1_epoch':source['epoch'],'stage1_selection':source['selection'],'stage2_selection':ck['selection'],'role':'public test model selection; not independent test'},indent=2))
    (root/'status.json').write_text(json.dumps({'state':'complete','epochs':epoch,'best_epoch':ck['epoch']}))

if __name__=='__main__':main()
