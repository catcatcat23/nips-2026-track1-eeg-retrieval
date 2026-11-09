"""Two independent GPU shards; official clean targets are never overwritten."""
import argparse,json,os,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from transformers import AutoModel,AutoImageProcessor
from data import config


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cfg=config();root=Path(cfg['run_root']);rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(rank)
    folder=root/('smoke_features' if args.smoke else 'feature_shards');folder.mkdir(exist_ok=True)
    m=json.loads((root/'data/manifest.json').read_text());clean=np.load(root/'data/clean.npy',mmap_mode='r')
    ids=sorted(set(np.load(root/'data/train_ids.npy').tolist())|set(np.load(root/'data/val_ids.npy').tolist()))
    if args.smoke:ids=ids[:8]
    ids=ids[rank::world]
    hf=Path(os.environ['HF_HOME'])/'hub/models--facebook--dinov2-giant/snapshots/611a9d42f2335e0f921f1e313ad3c1b7178d206d'
    model=AutoModel.from_pretrained(hf,local_files_only=True).cuda().eval().requires_grad_(False)
    processor=AutoImageProcessor.from_pretrained(hf,local_files_only=True,do_rescale=False)
    tf=transforms.Compose([transforms.Resize(518),transforms.ToTensor()])
    report={'rank':rank,'images':len(ids),'order':'Resize(518),ToTensor,GaussianBlur,official processor','processor':processor.to_dict(),'precision':'float32','checks':[]}
    for start in range(0,len(ids),cfg['extract_batch_size']):
        bi=ids[start:start+cfg['extract_batch_size']];dest=folder/f'rank{rank}_{start:06d}.npz'
        if dest.exists():continue
        images=[tf(Image.open(m['paths'][i]).convert('RGB')).cuda() for i in bi]
        globals_=[];locals_=[]
        with torch.inference_mode():
            for kernel in cfg['kernels']:
                # OpenCV sigma=0 rule, explicitly recorded for reproducibility.
                sigma=.3*((kernel-1)*.5-1)+.8
                blurred=[x if kernel==1 else gaussian_blur(x,[kernel,kernel],[sigma,sigma]) for x in images]
                inputs=processor(images=blurred,return_tensors='pt').to('cuda')
                out=model(**inputs,output_hidden_states=True)
                layer=int(.6667*(len(out.hidden_states)-1));tokens=out.hidden_states[layer]
                global_=tokens.mean(1)
                spatial=tokens[:,1+getattr(model.config,'num_register_tokens',0):]
                side=math.isqrt(spatial.shape[1]);assert side*side==spatial.shape[1]
                local=F.adaptive_avg_pool2d(spatial.transpose(1,2).reshape(len(bi),1536,side,side),(3,3)).flatten(2).transpose(1,2)
                globals_.append(global_.cpu());locals_.append(local.cpu())
                if kernel==1:
                    target=torch.from_numpy(np.asarray(clean[bi]).copy())
                    error=(global_.cpu()-target).abs().max().item()
                    cosine=F.cosine_similarity(global_.cpu(),target).min().item()
                    assert torch.allclose(global_.cpu(),target,rtol=2e-4,atol=2e-4),(error,cosine)
                    if start==0:report['checks'].append({'max_abs_error':error,'min_cosine':cosine,'pixel_shape':list(inputs['pixel_values'].shape),'hidden_layer':layer,'patch_grid':side})
        g=torch.stack(globals_).mean(0).numpy();v=torch.stack(locals_).mean(0).numpy()
        assert np.isfinite(g).all() and np.isfinite(v).all()
        np.savez(dest,ids=np.array(bi),blur=g,local=v)
        print('rank',rank,'images',start+len(bi),'/',len(ids),flush=True)
    (folder/f'complete_rank{rank}.json').write_text(json.dumps(report,indent=2,default=str))

if __name__=='__main__':main()
