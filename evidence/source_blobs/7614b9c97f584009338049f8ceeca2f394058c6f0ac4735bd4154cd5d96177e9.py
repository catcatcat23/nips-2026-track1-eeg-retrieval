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
    assert json.loads((out/'picture_gpu_check.json').read_text())['passed']
    def retrieval_loss(model,batch,weights):
        value,parts=ex.original_loss(model,batch,weights)
        parts['identity']=value.new_zeros(())
        model.pooled_for_identity=None
        return value,parts
    ex.loss_fn=retrieval_loss
    base.LocalOTModel=cls;base.MixedTrials=ex.Trials;base.loss_fn=ex.loss_fn;base.evaluate=ex.evaluate
    base.train(cfg,out)
    if cfg.get('heldout_subject'):ex.final_loso(cfg,out,cls)
    (out/'queue_status.txt').write_text('complete\n')

if __name__=='__main__':main()
