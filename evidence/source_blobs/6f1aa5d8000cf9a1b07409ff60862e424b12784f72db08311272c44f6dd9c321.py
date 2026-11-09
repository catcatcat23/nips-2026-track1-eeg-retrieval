"""Extract reference-protocol local tokens; never replace clean/global caches."""
import argparse,json,os,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoModel,AutoImageProcessor
from neuralset.extractors.image import _fix_pixel_values
from upstream_blur import load_blurred,LEVELS

SCHEMA='reference224_clean_layer06667_region_l2_3x3_v68'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=json.loads(args.config.read_text());root=Path(cfg['run_root']);src=Path(cfg['source_root'])
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 paths=json.loads((src/'data/manifest.json').read_text())['paths'];ids=np.union1d(np.load(src/'data/train_ids.npy'),np.load(src/'data/val_ids.npy'));ids=np.union1d(ids,np.load(src/'data/test_ids.npy'));allids=ids.copy()
 if args.smoke:ids=np.union1d(np.load(src/'data/train_ids.npy'),np.load(src/'data/val_ids.npy'))[:8]
 out=root/'local_shards';out.mkdir(exist_ok=True);blur=np.load(src/'features/blur.npy',mmap_mode='r')
 hf=Path(os.environ['HF_HOME'])/'hub/models--facebook--dinov2-giant/snapshots/611a9d42f2335e0f921f1e313ad3c1b7178d206d'
 model=AutoModel.from_pretrained(hf,local_files_only=True).cuda().eval().requires_grad_(False);processor=AutoImageProcessor.from_pretrained(hf,local_files_only=True,do_rescale=False);transform=transforms.Compose([transforms.Resize(518),transforms.ToTensor()]);start=time.time()
 for count,i in enumerate(ids):
  dest=out/f'{int(i):05d}.npz'
  if dest.exists():
   with np.load(dest) as z:assert str(z['schema'])==SCHEMA and str(z['filepath'])==paths[i] and int(z['id'])==i and z['local'].shape==(9,1536)
   continue
  globals_,locals_=[],[]
  with torch.inference_mode():
   for kernel in [1]:
    inputs=processor(images=[transform(load_blurred(paths[i],kernel)).cuda().float()],return_tensors='pt');_fix_pixel_values(inputs);inputs=inputs.to('cuda');res=model(**inputs,output_hidden_states=True)
    tokens=res.hidden_states[int(.6667*(len(res.hidden_states)-1))];ph,pw=inputs['pixel_values'].shape[-2:];gh,gw=ph//model.config.patch_size,pw//model.config.patch_size
    assert tokens.shape[1]-gh*gw==1,'Expected exactly one non-spatial CLS token'
    globals_.append(F.normalize(tokens[0].mean(0),dim=-1).cpu())
    regions=F.adaptive_avg_pool2d(tokens[:,1:].transpose(1,2).reshape(1,1536,gh,gw),(3,3)).flatten(2).transpose(1,2)[0]
    locals_.append(F.normalize(regions,dim=-1).cpu())
  global_=torch.stack(globals_).mean(0).numpy();local=F.normalize(torch.stack(locals_).mean(0),dim=-1).numpy()
  reference=blur[i]
  if not np.isfinite(reference).all():
   with np.load(Path(cfg['test_blur_shards'])/f'{i:05d}.npz') as z:
    assert int(z['id'])==i and str(z['filepath'])==paths[i];reference=z['blur'].copy()
  assert np.isfinite(local).all()
  with dest.with_suffix('.tmp').open('wb') as f:np.savez(f,id=i,filepath=paths[i],local=local,schema=SCHEMA,global_max_error=-1.0)
  dest.with_suffix('.tmp').replace(dest)
  if count%25==0:print('LOCAL',count+1,len(ids),'seconds',time.time()-start,flush=True)
 if args.smoke:
  (root/'local_smoke.json').write_text(json.dumps(dict(passed=True,images=len(ids),schema=SCHEMA)));return
 cache=root/'local.npy';a=np.lib.format.open_memmap(cache,mode='w+',dtype='float32',shape=(len(paths),9,1536));a[:]=np.nan
 for i in allids:
  with np.load(out/f'{i:05d}.npz') as z:
   assert str(z['schema'])==SCHEMA and int(z['id'])==i and str(z['filepath'])==paths[i];assert np.isfinite(z['local']).all();a[i]=z['local']
 a.flush();(root/'local_complete.json').write_text(json.dumps(dict(schema=SCHEMA,count=len(allids),global_cache_unchanged=True,processor=processor.to_dict()),default=str,indent=2))
if __name__=='__main__':main()
