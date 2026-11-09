"""Run immutable v64 training loop with narrowly scoped extensions."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
import extensions as ex
import local_ot_train_ema as base


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    cfg=json.loads(args.config.read_text());out=Path(cfg['run_root']);ex.CFG.update(cfg)
    torch.set_num_threads(4);torch.manual_seed(33);np.random.seed(33);random.seed(33)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cls=ex.IdentityModel if cfg.get('identity_weight') else ex.AugmentedModel
    ex.audit(cfg,out)
    if args.preflight:ex.preflight(cfg,out,cls);return
    assert json.loads((out/'preflight.json').read_text())['passed']
    original=ex.loss_fn
    calls=0
    def monitored(model,batch,weights):
        nonlocal calls
        total,parts=original(model,batch,weights)
        if calls<3 or calls%100==0:
            params=[p for p in model.encoder.parameters() if p.requires_grad]
            retrieval=parts['clean']+parts['blur']+.1*parts['local']
            identity=cfg['identity_weight']*parts['identity']
            g=torch.autograd.grad(retrieval,params,retain_graph=True,allow_unused=True)
            h=torch.autograd.grad(identity,params,retain_graph=True,allow_unused=True)
            aa=total.new_zeros(());bb=aa.clone();ab=aa.clone()
            for x,y in zip(g,h):
                if x is not None:aa+=x.detach().float().square().sum()
                if y is not None:bb+=y.detach().float().square().sum()
                if x is not None and y is not None:ab+=(x.detach().float()*y.detach().float()).sum()
            row=dict(step=calls,retrieval_encoder_l2=float(aa.sqrt()),identity_weighted_encoder_l2=float(bb.sqrt()),identity_to_retrieval_ratio=float((bb/aa.clamp_min(1e-20)).sqrt()),cosine=float(ab/(aa*bb).sqrt().clamp_min(1e-20)))
            with (out/'gradient_balance.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('GRADIENT_BALANCE',json.dumps(row),flush=True)
        calls+=1
        return total,parts
    ex.loss_fn=monitored
    base.LocalOTModel=cls;base.MixedTrials=ex.Trials;base.loss_fn=ex.loss_fn;base.evaluate=ex.evaluate
    base.train(cfg,out)
    if cfg.get('heldout_subject'):ex.final_loso(cfg,out,cls)
    (out/'queue_status.txt').write_text('complete\n')

if __name__=='__main__':main()
