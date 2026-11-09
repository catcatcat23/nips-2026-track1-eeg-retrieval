"""Last-two-block LoRA; cached frozen prefix; native blur retrieval."""
import argparse,copy,json,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common_target_readout import ROOT,load,write,evaluate,digest
from pooled_residual_calibration import native

class LoRALinear(nn.Module):
    def __init__(self,base,parts=1):
        super().__init__();self.base=base.requires_grad_(False);self.parts=parts
        self.a=nn.ParameterList([nn.Parameter(base.weight.new_empty(4,base.in_features)) for _ in range(parts)])
        self.b=nn.ParameterList([nn.Parameter(base.weight.new_zeros(base.out_features//parts,4)) for _ in range(parts)])
        for a in self.a:nn.init.kaiming_uniform_(a,a=5**.5)
    def forward(self,x):
        return self.base(x)+torch.cat([F.linear(F.linear(x,a),b) for a,b in zip(self.a,self.b)],dim=-1)

class Tail(nn.Module):
    def __init__(self,layers,projection):
        super().__init__();self.layers=layers;self.projection=projection
    def forward(self,x):
        for attn,ff in self.layers:
            x=x+attn(x);x=x+ff(x)
        return self.projection(x.mean(1))

def add_lora(tail):
    tail.requires_grad_(False)
    names=[]
    for i,(attn,_) in enumerate(tail.layers):
        attn.to_qkv=LoRALinear(attn.to_qkv,3);attn.to_out=LoRALinear(attn.to_out)
        names.extend([f'encoder.transformer.layers.{20+i}.0.to_qkv (separate rank4 Q,K,V)',f'encoder.transformer.layers.{20+i}.0.to_out'])
    return names

@torch.no_grad()
def score(model,mem,labels,gallery):
    model.eval();pred=[]
    for i in range(0,len(mem),128):pred.append(model(torch.tensor(mem[i:i+128].copy(),device='cuda')).cpu())
    return evaluate(nn.Identity(),torch.cat(pred),labels,gallery)

def model_for(cfg,meta):
    import extensions as ex
    ex.CFG.update(cfg)
    ck=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    model=(ex.IdentityModel if cfg.get('identity_weight') else ex.AugmentedModel)(ck['manifest'])
    model.load_state_dict(ck['model'],strict=True)
    return model.cuda().eval().requires_grad_(False)

def prepare(fold,arm,out):
    from data import DatasetEEG
    c,_,_,labels,meta=load(fold,arm);cfg=json.loads((ROOT/f'fold_{fold:02d}_{arm}'/'config.json').read_text())
    projection,targets,gallery,expected=native(fold,arm,c,meta)
    model=model_for(cfg,meta);assert len(model.encoder.transformer.layers)==22
    layers=model.encoder.transformer.layers[-2:]
    reuse=os.environ.get('LORA_REUSE_ROOT')
    if reuse:
        dest=Path(reuse)/f's{fold}_{arm}'
        previous=json.loads((dest/'manifest.json').read_text())
        assert previous['checkpoint_sha256']==meta['checkpoint_sha256']
        assert previous['records']==meta['records'] and previous['image_pool']==meta['image_pool']
        trainmem=np.load(dest/'train_tokens.npy',mmap_mode='r');testmem=np.load(dest/'test_tokens.npy',mmap_mode='r')
        assert trainmem.shape==(1000,63,512) and testmem.shape==(len(labels),63,512)
        tail=Tail(copy.deepcopy(layers).cpu(),projection)
        old=torch.load(dest/'tail_base.pt',weights_only=True,map_location='cpu')
        for k,v in tail.state_dict().items():torch.testing.assert_close(v,old[k],rtol=0,atol=0)
        actual=score(tail.cuda(),testmem,labels,gallery)
        for group in ['single_trial','subject_aggregated']:
            for metric in ['top1','top5']:assert abs(actual[group][metric]-expected[group][metric])*expected[group]['count']<=2
        tail.cpu();del model;torch.cuda.empty_cache()
        write(out/f's{fold}_{arm}_reuse.json',dict(cache=str(dest),validated=True,checkpoint_sha256=meta['checkpoint_sha256']))
        return c,labels,targets,gallery,tail,dest
    captured=[]
    hook=layers[0][0].register_forward_pre_hook(lambda m,x:captured.append(x[0].detach()))
    train=[DatasetEEG(cfg['source_root'],s,aux=False) for s in ['train','val']];test=DatasetEEG(cfg['source_root'],'test',aux=False)
    ti=np.flatnonzero(test.subjects.astype(str)==cfg['heldout_subject']);assert len(ti)==len(labels)
    dest=out/f's{fold}_{arm}';dest.mkdir()
    @torch.no_grad()
    def extract(which):
        n=1000 if which=='train' else len(ti);mem=None
        for start in range(0,n,64):
            if which=='train':x=torch.stack([torch.from_numpy(train[d].x[j].copy()) for d,j in c['records'][start:start+64]])
            else:x=torch.tensor(test.x[ti[start:start+64]].copy())
            captured.clear();h,_,_=model.encode_eeg(x.cuda());token=captured[-1]
            if mem is None:mem=np.lib.format.open_memmap(dest/f'{which}_tokens.npy',mode='w+',dtype='float32',shape=(n,*token.shape[1:]))
            mem[start:start+len(x)]=token.cpu().numpy()
            # Verify exact cached-feature correspondence against prior frozen model.
            ref=c['etrain'] if which=='train' else c['etest']
            torch.testing.assert_close(h.cpu(),projection(ref[start:start+len(x)]),rtol=2e-4,atol=2e-4)
            if start%2048==0:print('EXTRACT',fold,arm,which,start,n,flush=True)
        mem.flush()
    extract('train');extract('test');hook.remove()
    tail=Tail(copy.deepcopy(layers).cpu(),projection)
    tok=np.load(dest/'test_tokens.npy',mmap_mode='r');actual=score(tail.cuda(),tok,labels,gallery)
    for group in ['single_trial','subject_aggregated']:
        for k in ['top1','top5']:assert abs(actual[group][k]-expected[group][k])*expected[group]['count']<=2
    torch.save(tail.cpu().state_dict(),dest/'tail_base.pt')
    write(dest/'manifest.json',meta);write(dest/'zero.json',actual)
    del model;torch.cuda.empty_cache()
    return c,labels,targets,gallery,tail,dest

def gate(rows):
    differences={}
    for f in [1,2]:
        for seed in range(33,38):
            b=rows[f'{f}_baseline_{seed}'];i=rows[f'{f}_identity_{seed}'];differences[f'{f}_{seed}']=100*(i['subject_aggregated']['top5']-b['subject_aggregated']['top5'])
    means={str(f):float(np.mean([v for k,v in differences.items() if k.startswith(str(f)+'_')])) for f in [1,2]}
    return dict(differences=differences,subject_means=means,mean=float(np.mean(list(differences.values()))),positive_pairs=sum(v>0 for v in differences.values()),passes=bool(np.mean(list(differences.values()))>=2 and min(means.values())>0 and sum(v>0 for v in differences.values())>=7))

def main(root):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available(),'CUDA required for LoRA; no CPU fallback'
    write(root/'protocol.json',dict(rank=4,alpha=4,dropout=0,blocks=[20,21],qkv='separate rank4 branches within combined QKV',frozen_prefix='cached float32 input to block20',lr=1e-4,steps=200,batch_size=50,weight_decay=.01,clip=1,budgets='200 first; 50 only if200 meets >=2pp bothsubjects positive >=7/10',loss='native blur one-way CE image-normalized temperature1',selection='fixed200; no test checkpoint selection',seeds=list(range(33,38))))
    pairs={}
    for f in [1,2]:
        for arm in ['baseline','identity']:
            write(root/'status.json',dict(state='extracting',fold=f,arm=arm))
            pairs[(f,arm)]=prepare(f,arm,root)
    for f in [1,2]:
        b=pairs[(f,'baseline')][0];i=pairs[(f,'identity')][0]
        assert np.array_equal(b['images'],i['images']) and b['records']==i['records']
    decisions={}
    for n in [200,50]:
        rows={}
        if n==50 and not decisions['200']['passes']:break
        for (f,arm),(cache,labels,targets,gallery,base,dest) in pairs.items():
            mem=np.load(dest/'train_tokens.npy',mmap_mode='r');testmem=np.load(dest/'test_tokens.npy',mmap_mode='r')
            for seed in range(33,38):
                key=f's{f}_{arm}_n{n}_seed{seed}';write(root/'status.json',dict(state='training',current=key))
                subset=np.random.default_rng(seed).permutation(1000)[:n];x=torch.tensor(mem[subset].copy(),device='cuda');y=targets[subset].cuda()
                torch.manual_seed(seed);tail=copy.deepcopy(base).cuda();names=add_lora(tail)
                parameters=[p for p in tail.parameters() if p.requires_grad];assert sum(p.numel() for p in parameters)==32768
                with torch.no_grad():torch.testing.assert_close(tail(x[:2]),base.cuda()(x[:2]),rtol=0,atol=0)
                base.cpu();original={k:v.cpu().clone() for k,v in tail.named_parameters() if not v.requires_grad}
                opt=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01);gen=torch.Generator().manual_seed(seed+10000);losses=[];gn=[]
                for step in range(1,201):
                    ix=torch.randperm(n,generator=gen)[:50].cuda();loss=F.cross_entropy(tail(x[ix])@y[ix].T,torch.arange(len(ix),device='cuda'))
                    opt.zero_grad();loss.backward();assert torch.isfinite(loss)
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                    gn.append(float(nn.utils.clip_grad_norm_(parameters,1)));opt.step();losses.append(float(loss.detach()))
                    if step==2:assert all(p.grad.abs().sum()>0 for p in parameters)
                assert all(torch.equal(p.cpu(),original[k]) for k,p in tail.named_parameters() if not p.requires_grad)
                scores=score(tail,testmem,labels,gallery);rows[f'{f}_{arm}_{seed}']=scores
                state={k:v.detach().cpu() for k,v in tail.named_parameters() if v.requires_grad}
                torch.save(state,root/f'{key}.pt')
                clone=copy.deepcopy(base).cuda();add_lora(clone);result=clone.load_state_dict(state,strict=False);assert not result.unexpected_keys
                with torch.no_grad():torch.testing.assert_close(tail(x[:2]),clone(x[:2]),rtol=0,atol=0)
                write(root/f'{key}.json',dict(scores=scores,zero=json.loads((dest/'zero.json').read_text()),losses=losses,gradient_norm=gn,modules=names,trainable_parameters=32768,support_indices=subset.tolist(),checkpoint=json.loads((dest/'manifest.json').read_text())['checkpoint']))
                del tail,clone;torch.cuda.empty_cache();print('DONE',key,scores,flush=True)
        decisions[str(n)]=gate(rows);write(root/f'decision_n{n}.json',decisions[str(n)])
    if '50' in decisions:
        diffs={k:(decisions['50']['differences'][k]+v)/2 for k,v in decisions['200']['differences'].items()}
        means=[np.mean([v for k,v in diffs.items() if k.startswith(str(f)+'_')]) for f in [1,2]]
        passed=np.mean(list(diffs.values()))>=2 and min(means)>0 and sum(v>0 for v in diffs.values())>=7
    else:passed=False
    write(root/'status.json',dict(state='complete',decisions=decisions,next='locked_extended_validation_s3_s4' if passed else 'stop_identity_adaptation_branch',note='no episodic training automatically'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:main(a.root)
    except Exception as exc:
        write(a.root/'failure.json',dict(error=repr(exc)));write(a.root/'status.json',dict(state='failed',error=repr(exc),time=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()));raise
