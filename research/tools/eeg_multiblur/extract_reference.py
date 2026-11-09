"""94b3a31 image protocol; new cache only, original clean targets untouched."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoModel,AutoImageProcessor
from upstream_blur import load_blurred,LEVELS
from neuralset.extractors.image import _fix_pixel_values
from data import config

SCHEMA='upstream_per_scale_l2_dino_v1'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cfg=config();root=Path(cfg['run_root']);rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((root/'data/manifest.json').read_text())
    ids=np.union1d(np.load(root/'data/train_ids.npy'),np.load(root/'data/val_ids.npy'))
    if args.smoke:ids=ids[np.linspace(0,len(ids)-1,8,dtype=int)]
    out=root/('reference_smoke' if args.smoke else 'reference_shards');out.mkdir(parents=True,exist_ok=True)
    hf=Path(os.environ['HF_HOME'])/'hub/models--facebook--dinov2-giant/snapshots/611a9d42f2335e0f921f1e313ad3c1b7178d206d'
    model=AutoModel.from_pretrained(hf,local_files_only=True).cuda().eval().requires_grad_(False)
    processor=AutoImageProcessor.from_pretrained(hf,local_files_only=True,do_rescale=False)
    transform=transforms.Compose([transforms.Resize(518),transforms.ToTensor()])
    for count,i in enumerate(ids[rank::world]):
        path=manifest['paths'][int(i)];dest=out/f'{int(i):05d}.npz'
        if dest.exists():
            z=np.load(dest)
            assert str(z['schema'])==SCHEMA and str(z['filepath'])==path
            continue
        levels=[]
        with torch.inference_mode():
            for kernel in LEVELS:
                tensor=transform(load_blurred(path,kernel)).cuda()
                inputs=processor(images=[tensor.float()],return_tensors='pt')
                _fix_pixel_values(inputs)
                result=model(**inputs.to('cuda'),output_hidden_states=True)
                feature=result.hidden_states[int(.6667*(len(result.hidden_states)-1))][0].mean(0)
                levels.append(F.normalize(feature,dim=-1).cpu())
        levels=torch.stack(levels)
        assert levels.shape==(12,1536) and torch.isfinite(levels).all()
        torch.testing.assert_close(levels.norm(dim=-1),torch.ones(12),rtol=1e-5,atol=1e-5)
        with dest.with_suffix('.tmp').open('wb') as f:
            np.savez(f,id=i,filepath=path,levels=levels.numpy(),blur=levels.mean(0).numpy(),schema=SCHEMA)
        dest.with_suffix('.tmp').replace(dest)
        if count%50==0:print(json.dumps({'rank':rank,'completed':count+1,'assigned':len(ids[rank::world])}),flush=True)
    (out/f'complete_rank{rank}.json').write_text(json.dumps({'schema':SCHEMA,'world_size':world,'count':len(ids[rank::world]),'processor':processor.to_dict()},default=str,indent=2))

if __name__=='__main__':main()
