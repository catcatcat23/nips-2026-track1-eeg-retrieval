"""Train-only five-fold region predictability; all transforms fit within folds."""
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from data import config

def main():
    cfg=config();root=Path(cfg['run_root']);ids=np.load(root/'data/train_ids.npy');rng=np.random.default_rng(cfg['seed'])
    selected={}
    for i in rng.permutation(len(ids)):
        selected.setdefault(int(ids[i]),int(i))
    image_ids=np.array(sorted(selected));rows=np.array([selected[int(i)] for i in image_ids])
    np.savez(root/'features/ridge_samples.npz',image_ids=image_ids,train_rows=rows)
    eeg=np.load(root/'data/train_eeg.npy',mmap_mode='r')[rows].reshape(len(rows),-1)
    visual=np.load(root/'features/local.npy',mmap_mode='r')[image_ids]
    scores=[]
    for fold,(tr,va) in enumerate(KFold(cfg['ridge_folds'],shuffle=True,random_state=cfg['seed']).split(rows)):
        scaler=StandardScaler().fit(eeg[tr]);xp=PCA(n_components=min(cfg['eeg_pca'],len(tr)-1,eeg.shape[1]),svd_solver='randomized',random_state=21)
        xt=xp.fit_transform(scaler.transform(eeg[tr]));xv=xp.transform(scaler.transform(eeg[va]))
        region=[]
        for p in range(9):
            yp=PCA(n_components=min(cfg['image_pca'],len(tr)-1,1536),svd_solver='randomized',random_state=21)
            yt=yp.fit_transform(visual[tr,p]);yv=yp.transform(visual[va,p])
            pred=Ridge(alpha=cfg['ridge_alpha']).fit(xt,yt).predict(xv)
            # Out-of-fold variance-weighted R², with training-mean baseline.
            region.append(float(1-np.square(yv-pred).sum()/max(np.square(yv).sum(),1e-12)))
        scores.append(region);print('fold',fold,region,flush=True)
    weights=np.maximum(np.mean(scores,0),0)
    fallback=bool(weights.sum()<1e-8)
    if fallback:weights=np.ones(9)
    weights=np.maximum(weights,1e-6);weights/=weights.sum()
    np.save(root/'features/weights.npy',weights.astype('float32'))
    (root/'features/weights.json').write_text(json.dumps({'fold_scores':scores,'weights':weights.tolist(),'uniform_fallback':fallback,'sampling':'one random deterministic training trial per image; seed21','fit_scope':'train only; scaler and both PCAs inside each fold'},indent=2))
if __name__=='__main__':main()
