"""Training-only, subject-specific 5-fold predictability; fold-local PCA/scaling."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

def correlation(a,b):
 a=a-a.mean(0);b=b-b.mean(0);den=np.sqrt((a*a).sum(0)*(b*b).sum(0));return np.divide((a*b).sum(0),den,out=np.zeros(a.shape[1]),where=den>1e-12)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());out=Path(cfg['run_root']);src=Path(cfg['source_root'])/'data'
 assert (out/'local_complete.json').exists();dest=out/'predictability';dest.mkdir(exist_ok=True)
 splitdata=[(np.load(src/f'{s}_eeg.npy',mmap_mode='r'),np.load(src/f'{s}_ids.npy'),np.load(src/f'{s}_subjects.npy')) for s in ['train','val']]
 validation_ids=np.array(json.loads(Path(cfg['validation_ids_path']).read_text()));subjects=np.unique(np.concatenate([d[2] for d in splitdata]));testids=np.load(src/'test_ids.npy');local=np.load(out/'local.npy',mmap_mode='r');result={}
 for subject in subjects:
  targetids=np.unique(np.concatenate([ids[subs==subject] for _,ids,subs in splitdata]));targetids=targetids[~np.isin(targetids,validation_ids)];assert not np.intersect1d(targetids,testids).size
  # One sample per subject/image, averaging training trials only; no test EEG used.
  x=np.zeros((len(targetids),63,24),np.float64);count=np.zeros(len(targetids),np.int64)
  for eeg,ids,subs in splitdata:
   indices=np.flatnonzero((subs==subject)&np.isin(ids,targetids))
   for start in range(0,len(indices),512):
    ix=indices[start:start+512];mapped=np.searchsorted(targetids,ids[ix]);pooled=eeg[ix].reshape(-1,63,24,5).mean(-1)
    np.add.at(x,mapped,pooled);np.add.at(count,mapped,1)
  assert (count>0).all();x=(x/count[:,None,None]).reshape(len(x),-1).astype('float32');y=np.asarray(local[targetids]);assert np.isfinite(y).all()
  folds=[]
  for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=50).split(x)):
   scale=StandardScaler().fit(x[tr]);px=PCA(n_components=128,svd_solver='randomized',random_state=33);xt=px.fit_transform(scale.transform(x[tr]));xv=px.transform(scale.transform(x[va]))
   # Image-fold PCA is reusable across subjects only for an identical training image set.
   key=hashlib.sha256(targetids[tr].tobytes()).hexdigest()[:16];vp=dest/f'visual_{key}.npz'
   if vp.exists():
    with np.load(vp) as f:mean=f['mean'];components=f['components']
   else:
    py=PCA(n_components=64,svd_solver='randomized',random_state=33).fit(y[tr].reshape(-1,1536));mean=py.mean_;components=py.components_;np.savez(vp,mean=mean,components=components)
   yt=((y[tr]-mean)@components.T).reshape(len(tr),-1);yv=((y[va]-mean)@components.T).reshape(len(va),-1)
   pred=Ridge(alpha=100.).fit(xt,yt).predict(xv)
   score=np.median(np.maximum(correlation(pred,yv).reshape(9,64),0),axis=1);folds.append(score)
   print('WEIGHTS',subject,'fold',fold+1,'scores',score.tolist(),flush=True)
  scores=np.mean(folds,axis=0);weights=np.clip(scores/max(float(np.percentile(scores,90)),1e-6),.1,1.)**2
  result[str(subject)]={'weights':weights.tolist(),'scores':scores.tolist(),'fold_scores':[f.tolist() for f in folds],'images':len(targetids),'mean_training_repeats':float(count.mean())}
  (dest/'subjects.json').write_text(json.dumps(result,indent=2))
 (dest/'complete.json').write_text(json.dumps(dict(subjects=list(result),folds=5,standardization_and_pca='fit on each training fold only',image_disjoint_folds=True,sample='mean official EEG training trials per subject/image; time-bin 120 to 24 for probe only',score='mean across folds of median positive component correlation per patch',ridge_alpha=100,training_splits=['train','val'],test_used=False,validation_ids_excluded=validation_ids.tolist()),indent=2))
if __name__=='__main__':main()
