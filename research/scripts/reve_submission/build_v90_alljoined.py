"""Export selected joint model and verify real native Alljoined input parity."""
import hashlib, importlib.util, json, shutil, socket, sys
from pathlib import Path
import numpy as np
import torch

BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
PROJECT=Path('/home/Anteng/eeg-retrieval')
RUN=BASE/'reve_v90_alljoined_full_public_joint'
OUT=BASE/'submission_v90_alljoined_ep15_20261107'
sys.path[:0]=[str(RUN/'source'),str(PROJECT/'competition-2026')]
from joint_train import JointModel

def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    package=OUT/'model';package.mkdir(exist_ok=True)
    source=RUN/'stage1/ema0.9998_epoch_015.pt'
    ck=torch.load(source,map_location='cpu',weights_only=False)
    assert ck['epoch']==15
    template=PROJECT/'submissions/reve_b_epoch6_compat_v2'
    text=(template/'submission.py').read_text().replace("name = 'REVE-B-Epoch6-CompatV2'", "name = 'REVE-v90-FullPublic-EMA9998-Epoch15'")
    (package/'submission.py').write_text(text)
    for name in ['positions.json','LICENSE_BRAINDECODE.txt']:shutil.copyfile(template/name,package/name)
    state=ck['model']
    weights={k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')}
    weights.update({'head.'+k.removeprefix('blur_head.'):v for k,v in state.items() if k.startswith('blur_head.')})
    weights.update({k:v for k,v in state.items() if k.startswith('recovery.')})
    torch.save(weights,package/'weights.pt')
    reference=JointModel(ck['manifest'],ck['alljoined_channels'])
    reference.load_state_dict(state,strict=True);reference=reference.cuda().eval().select('alljoined')
    config=json.loads((BASE/'reve_v67_alljoined2025_v64recipe/config.json').read_text())
    manifest=json.loads(Path(config['data_manifest']).read_text())
    channels=manifest['channels'];assert channels==ck['alljoined_channels']
    bank=json.loads((package/'positions.json').read_text());lookup={k.upper():v for k,v in bank.items()}
    positions=torch.tensor([lookup[n.removeprefix('EEG ').split('-',1)[0].upper()] for n in channels])
    torch.testing.assert_close(positions,state['alljoined_positions'],rtol=0,atol=0)
    spec=importlib.util.spec_from_file_location('exported_v90',package/'submission.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    original=socket.socket.connect
    def guard(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('Network forbidden')
        return original(sock,address)
    socket.socket.connect=guard
    meta=dict(n_chans=32,n_times=120,sfreq=120.,ch_names=channels,n_outputs=1536,submission_dir=package,device='cuda')
    model=mod.Solver().load_model(meta)
    # Spread the real parity sample across recordings and subjects.
    arrays=[np.load(r['array'],mmap_mode='r')[:2] for r in manifest['test']]
    x=torch.tensor(np.concatenate(arrays)[:257].copy(),device='cuda')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.inference_mode():
        expected=reference(x);actual=model.predict(x)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        perm=torch.randperm(len(x),device='cuda')
        torch.testing.assert_close(model.predict(x[perm]),actual[perm],rtol=1e-4,atol=1e-4)
        for size in [1,7,33]:torch.testing.assert_close(model.predict(x[:size]),actual[:size],rtol=1e-4,atol=1e-4)
        assert model.predict(x[:0]).shape==(0,1536)
        cpu=mod.Solver().load_model(dict(meta,device='cpu'))
        torch.testing.assert_close(cpu.predict(x[:2].cpu()),actual[:2].cpu(),rtol=1e-4,atol=1e-4)
        order=torch.randperm(32);moved=mod.Solver().load_model(dict(meta,ch_names=[channels[i] for i in order]))
        torch.testing.assert_close(moved.predict(x[:7,order]),actual[:7],rtol=1e-4,atol=1e-4)
    report=dict(source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        weights_sha256=hashlib.sha256((package/'weights.pt').read_bytes()).hexdigest(),
        expected_clean=ck['metrics']['alljoined']['clean'],channels=channels,real_samples=len(x),
        max_abs_error=float((expected-actual).abs().max()),positions_exact=True,
        batch_permutation='passed',variable_batches='passed',channel_permutation='passed',cpu_gpu='passed',network='disabled',
        selection='fixed epoch15 EMA0.9998; full public training including original test; local scores are in-sample diagnostics',
        inference='EEG only -> REVE -> blur projection -> clean recovery; no candidates, image adapter or OT')
    (OUT/'build.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='expected_clean'},indent=2),flush=True)

if __name__=='__main__':main()
