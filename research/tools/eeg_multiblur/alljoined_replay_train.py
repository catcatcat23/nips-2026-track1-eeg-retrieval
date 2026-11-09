"""Shared REVE fine-tuning with native-channel homogeneous batches."""
import argparse,json,math,time
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from channel_contract import register_channels, channel_positions, batch_positions
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from reference_model import ReferenceModel
from recovery_ablation import ProjectedHead,metrics
from data import DatasetEEG
from neuraltrain.losses.losses import ClipLoss
from alljoined_prepare import ROOT,BASE

class MixedData(Dataset):
    def __init__(self):
        self.inv=json.loads((ROOT/'inventory.json').read_text())
        self.records=[];self.channels={};self.domains={};self.maps=OrderedDict()
        manifest=json.loads((BASE/'data/manifest.json').read_text())
        register_channels(self.channels, 'things', manifest['channels'])
        for split in ['train','val']:
            ids=np.load(BASE/'data'/f'{split}_ids.npy')
            self.records.append({'array':str(BASE/'data'/f'{split}_eeg.npy'),'ids':ids,'dataset':'things'})
        for r in json.loads((ROOT/'eeg_complete.json').read_text())['recordings']:
            register_channels(self.channels, r['dataset'], r['channels'])
            self.records.append({'array':r['array'],'ids':np.array([i for _,i in r['events']]),'dataset':r['dataset']})
        self.ends=np.cumsum([len(r['ids']) for r in self.records]);start=0
        for r,end in zip(self.records,self.ends):
            self.domains.setdefault(r['dataset'],[]).append(np.arange(start,end));start=end
        self.domains={k:np.concatenate(v) for k,v in self.domains.items()}
        n=len(self.inv['paths']);self.clean=np.empty((n,1536),np.float32);self.blur=np.empty_like(self.clean)
        base=self.inv['base_images'];self.clean[:base]=np.load(BASE/'data/clean.npy');self.blur[:base]=np.load(BASE/'features/blur.npy')
        for i in range(base,n):
            with np.load(ROOT/'features'/f'{i:05d}.npz') as f:self.clean[i]=f['clean'];self.blur[i]=f['blur']
        for r in self.records:
            assert np.isfinite(self.clean[r['ids']]).all() and np.isfinite(self.blur[r['ids']]).all()
    def __len__(self):return int(self.ends[-1])
    def __getitem__(self,index):
        ri=int(np.searchsorted(self.ends,index,side='right'));local=index-(self.ends[ri-1] if ri else 0);r=self.records[ri]
        key=r['array']
        if key not in self.maps:
            if len(self.maps)>=32:self.maps.popitem(last=False)
            self.maps[key]=np.load(key,mmap_mode='r')
            expected=(len(r['ids']),len(self.channels[r['dataset']]),120)
            if self.maps[key].shape != expected:
                raise ValueError(f'{key}: expected {expected}, got {self.maps[key].shape}')
        self.maps.move_to_end(key);k=int(r['ids'][local])
        return {'eeg':torch.from_numpy(self.maps[key][local].copy()),'clean':torch.from_numpy(self.clean[k].copy()),'blur':torch.from_numpy(self.blur[k].copy()),'id':k,'domain':r['dataset']}

class Batches:
    def __init__(self,ds,batch):self.ds=ds;self.batch=batch;self.epoch=0;self.steps=math.ceil(len(ds.domains['things'])/batch)*2
    def __len__(self):return self.steps
    def __iter__(self):
        rng=np.random.default_rng(33+self.epoch)
        # Explicit deterministic 50/45/5 block balance, random ordering within blocks.
        schedule=np.resize(np.array(['things']*10+['Xu2025Alljoined']*9+['Xu2024Alljoined']),self.steps)
        rng.shuffle(schedule)
        for name in schedule:yield rng.choice(self.ds.domains[name],self.batch,replace=True).tolist()

class MixedModel(ReferenceModel):
    def __init__(self,manifest,channels,base,head):
        super().__init__(manifest);self.load_state_dict(base)
        self.recovery=ProjectedHead();self.recovery.load_state_dict(head)
        self.names=list(channels)
        for name,names in channels.items():self.register_buffer('pos_'+name,channel_positions(self.encoder,names))
    def forward(self,x,blur=None,domain='things'):
        x=self.preprocess(x);pos=batch_positions(self,x,domain)
        h=self.blur_head(self.encoder(x,pos=pos,return_output=True)[-1].mean(1));q=self.recovery(h)
        return q if blur is None else (h,q,F.normalize(self.image_adapter(blur),dim=-1))

@torch.no_grad()
def evaluate(model):
    model.eval();ds=DatasetEEG(BASE,'test',aux=False);parts=[]
    for b in DataLoader(ds,batch_size=256,num_workers=2):parts.append(model(b['eeg'].cuda()).cpu().numpy())
    return metrics(np.concatenate(parts),ds.ids,ds.subjects,np.asarray(ds.clean))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    cfg=json.loads(Path(__file__).with_name('alljoined_replay_config.json').read_text())
    torch.manual_seed(33);np.random.seed(33);torch.backends.cuda.matmul.allow_tf32=False
    ds=MixedData();batcher=Batches(ds,cfg['batch_size'])
    base=torch.load(cfg['base_checkpoint'],map_location='cpu',weights_only=False)
    head=torch.load(cfg['head_checkpoint'],map_location='cpu',weights_only=False)
    model=MixedModel(base['manifest'],ds.channels,base['model'],head['head']).cuda()
    criterion=ClipLoss(norm_kind='y',temperature=False,symmetric=False).cuda()
    encoder=list(model.encoder.parameters());others=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]
    optimizer=torch.optim.AdamW([{'params':encoder,'lr':cfg['encoder_lr']},{'params':others,'lr':cfg['lr']}],weight_decay=.05)
    if args.preflight:
        results={}
        for name,indices in ds.domains.items():
            subset=torch.utils.data.Subset(ds,indices[:256].tolist())
            b=next(iter(DataLoader(subset,batch_size=256)))
            model.train();optimizer.zero_grad(set_to_none=True)
            h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),name)
            loss=criterion(q,b['clean'].cuda())+criterion(h,a);loss.backward()
            for n,p in model.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all(),n
            assert float(sum(p.grad.abs().sum() for p in encoder))>0
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            results[name]={'shape':list(b['eeg'].shape),'loss':float(loss.detach()),'gradients':'all finite including REVE'}
        # Fresh initialization must reproduce B on unchanged current test EEG.
        model=MixedModel(base['manifest'],ds.channels,base['model'],head['head']).cuda()
        actual=evaluate(model)
        for group in ['single_trial','subject_aggregated']:
            for key in ['top1','top5']:assert abs(actual[group][key]-head['selection'][group][key])<1e-5
        (Path(cfg['run_root'])/'preflight.json').write_text(json.dumps({'domains':results,'B_full_test_replay':actual},indent=2));print('PREFLIGHT PASSED',flush=True);return
    out=Path(cfg['run_root'])/'training';out.mkdir(parents=True,exist_ok=True);assert not (out/'history.jsonl').exists()
    loader=DataLoader(ds,batch_sampler=batcher,num_workers=4,pin_memory=True,persistent_workers=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5,min_lr=1e-6)
    best=float('inf');best_epoch=None;stale=0;epoch_loss=None
    def checkpoint(epoch,selection,name="best_loss.pt"):
        torch.save({'model':model.state_dict(),'epoch':epoch,'config':cfg,'channels':ds.channels,'manifest':base['manifest'],'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'selection':selection,'train_loss':epoch_loss,'best_loss':best,'best_epoch':best_epoch},out/name)
    checkpoint(0,head['selection'],'initial.pt')
    for epoch in range(1,cfg['epochs']+1):
        batcher.epoch=epoch;model.train();total=0.;domain_totals={};start=time.time()
        for step,b in enumerate(loader):
            assert len(set(b['domain']))==1
            optimizer.zero_grad(set_to_none=True)
            h,q,a=model(b['eeg'].cuda(),b['blur'].cuda(),b['domain'][0])
            lc=criterion(q,b['clean'].cuda());lb=criterion(h,a);loss=lc+lb
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())
            name=b['domain'][0];stats=domain_totals.setdefault(name,[0.,0.,0]);stats[0]+=float(lc.detach());stats[1]+=float(lb.detach());stats[2]+=1
            if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'domain':b['domain'][0],'clean':float(lc.detach()),'blur':float(lb.detach())}),flush=True)
        epoch_loss=total/len(loader)
        scheduler.step(epoch_loss)
        selection=evaluate(model)  # Monitoring only: no effect on training decisions.
        if epoch_loss<best:
            best=epoch_loss;best_epoch=epoch;stale=0;checkpoint(epoch,selection)
        else:stale+=1
        checkpoint(epoch,selection,'last.pt')
        domain_losses={k:{'clean':v[0]/v[2],'blur':v[1]/v[2],'batches':v[2]} for k,v in domain_totals.items()}
        row={'epoch':epoch,'loss':epoch_loss,'domain_losses':domain_losses,'selection':selection,'best_loss':best,'best_epoch':best_epoch,'stale':stale,'lr':[g['lr'] for g in optimizer.param_groups],'seconds':time.time()-start}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        (out/'status.json').write_text(json.dumps({'state':'training','epoch':epoch,'best_loss':best,'best_epoch':best_epoch}))
        if stale>=cfg['patience']:break
    (out/'status.json').write_text(json.dumps({'state':'complete','epochs':epoch,'best_loss':best,'best_epoch':best_epoch,'public_test_role':'monitoring only; selection and scheduler use mean training loss'}))

if __name__=='__main__':main()
