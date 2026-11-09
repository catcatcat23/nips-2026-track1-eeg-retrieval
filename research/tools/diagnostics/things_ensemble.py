"""Fixed-checkpoint full THINGS ensemble replay; no fitting on test data."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import DatasetEEG
from local_ot_model import LocalOTModel
from recovery_ablation import metrics

BASE = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
OUT = BASE / 'things_ensemble_v64_v69_20261105'
CHOICES = {
    'v64': ('reve_v64_stage1_ema', 'ema0.9998_epoch_014.pt', 14),
    'v69': ('reve_v69_joint_stage1', 'ema0.9998_epoch_015.pt', 15),
}

def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['v64', 'v69', 'compare'])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(33)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ds = DatasetEEG(BASE/'reve_v13_reference', 'test', aux=False)
    if args.mode == 'compare':
        result = {k: json.loads((OUT/f'{k}_result.json').read_text()) for k in CHOICES}
        arrays = []
        for k in CHOICES:
            idx = np.load(OUT/f'{k}_indices.npz')
            assert np.array_equal(idx['ids'], ds.ids)
            assert np.array_equal(idx['subjects'], ds.subjects)
            arrays.append(np.load(OUT/f'{k}_clean.npy', mmap_mode='r'))
        for kind in ['raw_equal_mean', 'unit_equal_mean']:
            pred = np.lib.format.open_memmap(OUT/f'{kind}.npy', mode='w+', dtype='float32', shape=arrays[0].shape)
            for start in range(0, len(ds), 4096):
                a, b = [np.array(x[start:start+4096]) for x in arrays]
                if kind == 'unit_equal_mean':
                    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
                pred[start:start+len(a)] = (a+b)*.5
            pred.flush()
            result[kind] = metrics(pred, ds.ids, ds.subjects, np.asarray(ds.clean))
        write('results.json', result)
        write('status.json', {'state': 'complete', 'protocol': 'Fixed checkpoints, equal weights; public test diagnostic, no weight search. Each trial predicted separately; raw ensemble predictions averaged by subject and image before cosine retrieval. Blur galleries are model-specific, so no cross-model blur-vector ensemble.'})
        print(json.dumps(result), flush=True)
        return
    name = args.mode
    folder, file, epoch = CHOICES[name]
    checkpoint = BASE/folder/'stage1'/file
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = LocalOTModel(state['manifest']).eval().requires_grad_(False)
    sd = dict(state['model'])
    if name == 'v69':
        assert 'alljoined_positions' in sd
        sd.pop('alljoined_positions')  # THINGS path uses the unchanged 63-channel positions.
    model.load_state_dict(sd, strict=True)
    del state
    model.cuda()
    np.savez(OUT/f'{name}_indices.npz', ids=ds.ids, subjects=ds.subjects)
    preds = {s: np.lib.format.open_memmap(OUT/f'{name}_{s}.npy', mode='w+', dtype='float32', shape=(len(ds),1536)) for s in ['clean','blur']}
    start = time.time()
    offset = 0
    with torch.inference_mode():
        for i, batch in enumerate(DataLoader(ds, batch_size=128, num_workers=2, pin_memory=True)):
            h,z,_ = model.encode_eeg(batch['eeg'].cuda())
            for s,t in [('clean',z),('blur',h)]:
                assert torch.isfinite(t).all()
                preds[s][offset:offset+len(t)] = t.cpu().numpy()
            offset += len(z)
            if i % 100 == 0:
                write(f'{name}_status.json', {'state':'inference','trials':offset,'total':len(ds),'elapsed_s':time.time()-start})
                print(name, offset, len(ds), time.time()-start, flush=True)
        assert offset == len(ds)
        for p in preds.values(): p.flush()
        cfg = json.loads((BASE/'reve_v64_stage1_ema/config.json').read_text())
        gallery = np.zeros_like(np.asarray(ds.clean))
        ids = np.unique(ds.ids)
        blur = []
        for ident in ids:
            with np.load(Path(cfg['test_blur_shards'])/f'{ident:05d}.npz') as f:
                assert int(f['id']) == ident
                blur.append(f['blur'].copy())
        gallery[ids] = model.image_adapter(torch.tensor(np.stack(blur),device='cuda')).cpu().numpy()
    result = {'checkpoint':str(checkpoint),'epoch':epoch,'clean':metrics(preds['clean'],ds.ids,ds.subjects,np.asarray(ds.clean)), 'blur':metrics(preds['blur'],ds.ids,ds.subjects,gallery)}
    rows = [json.loads(x) for x in (BASE/folder/'stage1/history.jsonl').read_text().splitlines() if x.strip()]
    expected = next(r for r in rows if r['epoch']==epoch)['metrics']['ema0.9998']
    if name == 'v69': expected = expected['things']
    for space in ['clean','blur']:
        for group in ['single_trial','subject_aggregated']:
            for k in ['top1','top5']:
                assert abs(result[space][group][k]-expected[space][group][k])*result[space][group]['count'] <= 10, (name,space,group,k,'history mismatch')
    result['history_replay_verified'] = True
    write(f'{name}_result.json',result)
    write(f'{name}_status.json',{'state':'complete','elapsed_s':time.time()-start})

if __name__ == '__main__': main()
