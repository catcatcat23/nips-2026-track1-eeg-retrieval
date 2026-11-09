"""Fresh end-to-end B architecture training from original pretrained REVE."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,ConcatDataset
from data import DatasetEEG
from alljoined_train import evaluate
from reference_model import ReferenceModel
from recovery_ablation import ProjectedHead
from torch.nn import functional as F

class FreshB(ReferenceModel):
    def __init__(self, manifest):
        super().__init__(manifest)
        self.recovery = ProjectedHead()
    def forward(self, x, blur=None):
        x=self.preprocess(x)
        pos=self.positions.to(x).unsqueeze(0).expand(len(x),-1,-1)
        h=self.blur_head(self.encoder(x,pos=pos,return_output=True)[-1].mean(1))
        q=self.recovery(h)
        return q if blur is None else (h,q,F.normalize(self.image_adapter(blur),dim=-1))
from image_id_loss import ImageIdClipLoss


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    cfg=json.loads(Path(__file__).with_name('fresh_b_config.json').read_text())
    root=Path(cfg['run_root']);root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((Path(cfg['data_root'])/'data/manifest.json').read_text())
    model=FreshB(manifest).cuda()
    assert all(p.requires_grad for p in model.parameters())
    ds=ConcatDataset([DatasetEEG(cfg['data_root'],s,aux=True,local_aux=False) for s in ['train','val']])
    generator=torch.Generator().manual_seed(cfg['seed'])
    loader=DataLoader(ds,batch_size=256,shuffle=True,generator=generator,num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True)
    encoder=list(model.encoder.parameters());others=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]
    opt=torch.optim.AdamW([{'params':encoder,'lr':cfg['encoder_lr']},{'params':others,'lr':cfg['lr']}],weight_decay=.05)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5,min_lr=[1e-6,1e-6])
    criterion=ImageIdClipLoss().cuda()
    if args.preflight:
        model.train()
        for step,b in enumerate(loader):
            opt.zero_grad(set_to_none=True);h,q,a=model(b['eeg'].cuda(),b['blur'].cuda())
            loss=criterion(q,b['clean'].cuda(),b['id'])+criterion(h,a,b['id']);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step==2:break
        (root/'preflight.json').write_text(json.dumps({'passed':True,'steps':3,'note':'Smoke updates discarded. Main run initializes original pretrained REVE and fresh heads again.','final_loss':float(loss.detach()),'training_samples':len(ds),'batch_size':256},indent=2))
        print('FRESH B PREFLIGHT PASSED',flush=True);return
    assert not (root/'history.jsonl').exists()
    (root/'provenance.json').write_text(json.dumps({'initialization':'original brain-bzh/reve-base; all heads random; no retrieval checkpoint loaded','training_samples':len(ds),'config':cfg},indent=2))
    best=-1.;stale=0
    def checkpoint(epoch,selection):
        torch.save({'model':model.state_dict(),'epoch':epoch,'config':cfg,'manifest':manifest,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'selection':selection},root/'best.pt')
    for epoch in range(1,cfg['epochs']+1):
        model.train();start=time.time();totals=np.zeros(2)
        for step,b in enumerate(loader):
            opt.zero_grad(set_to_none=True);h,q,a=model(b['eeg'].cuda(),b['blur'].cuda())
            lc=criterion(q,b['clean'].cuda(),b['id']);lb=criterion(h,a,b['id']);loss=lc+lb
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            totals += [float(lc.detach()),float(lb.detach())]
            if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'clean':float(lc.detach()),'blur':float(lb.detach())}),flush=True)
        selection=evaluate(model);score=selection['single_trial']['top5'];sched.step(score)
        if score>best:best=score;stale=0;checkpoint(epoch,selection)
        else:stale+=1
        row={'epoch':epoch,'clean':totals[0]/len(loader),'blur':totals[1]/len(loader),'selection':selection,'best_top5':best,'stale':stale,'lr':[g['lr'] for g in opt.param_groups],'seconds':time.time()-start}
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        (root/'status.json').write_text(json.dumps({'state':'training','epoch':epoch,'best_top5':best}))
        print(json.dumps(row),flush=True)
        if stale>=cfg['patience']:break
    ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    (root/'result.json').write_text(json.dumps({'best_epoch':ck['epoch'],'selection':ck['selection'],'role':'public test model selection; not independent test'},indent=2))
    (root/'status.json').write_text(json.dumps({'state':'complete','epochs':epoch,'best_epoch':ck['epoch']}))

if __name__=='__main__':main()
