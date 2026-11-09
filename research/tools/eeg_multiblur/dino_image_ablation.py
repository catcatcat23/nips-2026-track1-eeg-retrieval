"""Original MultiBlur image pipeline, replacing RN50 with frozen DINOv2-giant."""
import argparse,json,os,math,sys,time,zipfile,io
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

KERNELS=[1,3,9,15,21,27,33,39,45,51,57,63]
UPSTREAM=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/upstream_multiblur_mean_sub01_seed21/source-0300630d9ec6/EEG_Image_IntraSubject_SelectiveLocalOT/intrasubject_retrieval')
SOURCE=Path('/mnt/DATA-2/Yanhao/Visual_EEG_Decoding_processed_data/things-eeg')
MANIFEST=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/multiblur_v4_compet120/data/manifest.json')


def prepare(root):
    paths=json.loads(MANIFEST.read_text())['paths']
    mapping={}
    for path in paths:
        for split in ['train','test']:
            token=('training_images/' if split=='train' else 'test_images/')
            if token in path:
                key=split+'_images/'+path.split(token,1)[1]
                assert key not in mapping and Path(path).is_file()
                mapping[key]=path
    keys=[]
    for split,expected in [('train',16540),('test',200)]:
        d=torch.load(SOURCE/'Image_feature'/f'MultiBlur_RN50_{split}.pt',map_location='cpu',weights_only=False)
        levels=set(map(str,KERNELS));assert set(d)==levels
        current=sorted(d['1']);assert len(current)==expected
        for level in levels:assert set(d[level])==set(current)
        assert set(current)<=set(mapping)
        keys.extend(current)
        del d
    archive=SOURCE/'Image_set.zip'
    members={}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            for split in ['train','test']:
                token=split+'_images/'
                if token in name and not name.endswith('/'):
                    key=token+name.split(token,1)[1]
                    assert key not in members
                    members[key]=name
    assert set(keys)<=set(members)
    items=[{'key':k,'path':mapping[k],'zip_member':members[k]} for k in keys]
    (root/'items.json').write_text(json.dumps(items,indent=2))
    data=root/'data';(data/'Image_feature').mkdir(parents=True,exist_ok=True)
    link=data/'Preprocessed_data'
    if link.exists():assert link.resolve()==(SOURCE/'Preprocessed_data').resolve()
    else:link.symlink_to(SOURCE/'Preprocessed_data',target_is_directory=True)
    print('PAIRING PASSED',len(items),flush=True)


def extract(root,rank,smoke=False):
    import cv2
    from transformers import AutoModel,AutoImageProcessor
    from torchvision import transforms
    items=json.loads((root/'items.json').read_text())
    ids=list(range(len(items))) if not smoke else [0,1,2,3,len(items)-4,len(items)-3,len(items)-2,len(items)-1]
    ids=ids[rank::2]
    hf=Path(os.environ['HF_HOME'])/'hub/models--facebook--dinov2-giant/snapshots/611a9d42f2335e0f921f1e313ad3c1b7178d206d'
    model=AutoModel.from_pretrained(hf,local_files_only=True).cuda().eval().requires_grad_(False)
    # Encoder-native ImageNet normalization, without a second resize/crop.
    processor=AutoImageProcessor.from_pretrained(hf,local_files_only=True,do_resize=False,do_center_crop=False)
    folder=root/('smoke' if smoke else 'shards');folder.mkdir(exist_ok=True)
    archive=zipfile.ZipFile(SOURCE/'Image_set.zip')
    tick=time.time();batch=16
    for start in range(0,len(ids),batch):
        bi=ids[start:start+batch];dest=folder/f'rank{rank}_{start:06d}.npz'
        if dest.exists():continue
        images=[]
        for i in bi:
            with Image.open(io.BytesIO(archive.read(items[i]['zip_member']))) as im:images.append(im.convert('RGB').resize((224,224)))
        levels=[]
        with torch.inference_mode():
            for kernel in KERNELS:
                blurred=[]
                for im in images:
                    if kernel==1:blurred.append(im)
                    else:
                        a=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
                        a=cv2.GaussianBlur(a,(kernel,kernel),0)
                        blurred.append(Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)))
                inputs=processor(images=blurred,return_tensors='pt').to('cuda')
                assert list(inputs['pixel_values'].shape[1:])==[3,224,224]
                out=model(**inputs,output_hidden_states=True)
                layer=int(.6667*(len(out.hidden_states)-1))
                z=F.normalize(out.hidden_states[layer].mean(1).float(),dim=-1)
                assert z.shape==(len(bi),1536) and torch.isfinite(z).all()
                levels.append(z.cpu().numpy())
        x=np.stack(levels,axis=1);assert np.allclose(np.linalg.norm(x,axis=-1),1,atol=1e-5)
        np.savez(dest,ids=np.array(bi),features=x)
        print(json.dumps({'rank':rank,'images':start+len(bi),'total':len(ids),'seconds':time.time()-tick}),flush=True)
    (folder/f'complete_rank{rank}.json').write_text(json.dumps({'images':len(ids),'passed':True,'seconds':time.time()-tick,'layer':layer,'dtype':'FP32','shape':[12,1536],'input':[3,224,224]}))


def merge(root):
    items=json.loads((root/'items.json').read_text());x=np.empty((len(items),12,1536),np.float32);seen=np.zeros(len(items),bool)
    for rank in [0,1]:assert (root/'shards'/f'complete_rank{rank}.json').is_file()
    for p in sorted((root/'shards').glob('*.npz')):
        with np.load(p) as f:
            ids=f['ids'];assert not seen[ids].any();x[ids]=f['features'];seen[ids]=True
    assert seen.all() and np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x,axis=-1),1,atol=1e-5)
    for split in ['train','test']:
        ids=[i for i,it in enumerate(items) if it['key'].startswith(split+'_images/')]
        # Keep upstream expected filenames solely to run its unmodified loader.
        result={str(k):{items[i]['key']:torch.from_numpy(x[i,j].copy()) for i in ids} for j,k in enumerate(KERNELS)}
        torch.save(result,root/'data/Image_feature'/f'MultiBlur_RN50_{split}.pt')
        del result
    (root/'features_complete.json').write_text(json.dumps({'encoder':'DINOv2-giant','images':len(items),'levels':12,'dim':1536,'normalization':'per-scale L2, before mean; expected RN50 filenames contain DINO features'}))


def train(root):
    sys.path.insert(0,str(UPSTREAM));sys.path.insert(0,str(UPSTREAM/'scripts'))
    import train_global_mean_baseline as upstream
    base=upstream.GlobalMeanEncoder
    class DinoDimEncoder(base):
        def __init__(self,channels,proj_dim,temporal_len):super().__init__(channels,1536,temporal_len)
    upstream.GlobalMeanEncoder=DinoDimEncoder
    args=argparse.Namespace(data_path=str(root/'data'),subject=1,seed=21,epochs=200,batch_size=1024,lr=.001,device='cuda:0',output_dir=str(root/'results'))
    upstream.train(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','merge','train']);p.add_argument('--root',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();root=Path(a.root)
    if a.stage=='prepare':prepare(root)
    elif a.stage=='extract':extract(root,a.rank,a.smoke)
    elif a.stage=='merge':merge(root)
    else:train(root)
