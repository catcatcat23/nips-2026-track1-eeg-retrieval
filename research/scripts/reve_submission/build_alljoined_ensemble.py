"""Export the validated 75/25 raw clean-output ensemble; audit real EEG parity."""
import hashlib, importlib.util, json, shutil, socket, sys
from pathlib import Path
import numpy as np
import torch
BASE=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
PROJECT=Path('/home/Anteng/eeg-retrieval')
OUT=BASE/'submission_alljoined_v69_v85_ensemble_20261106'
CHOICES=[('reve_v69_joint_stage1','ema0.9998_epoch_015.pt'),('reve_v85_alljoined_identity_global','ema0.9998_epoch_008.pt')]
TAIL='''
class OutputEnsemble(nn.Module):
    def __init__(self, models, coefficients):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.coefficients = coefficients

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        output = None
        for coefficient, model in zip(self.coefficients, self.models):
            value = model.predict(X) * coefficient
            output = value if output is None else output + value
        return output
'''
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    OUT.mkdir(exist_ok=True); package=OUT/'model';package.mkdir(exist_ok=True)
    template=PROJECT/'submissions/reve_b_epoch6_compat_v2'
    text=(template/'submission.py').read_text().replace("name = 'REVE-B-Epoch6-CompatV2'", "name = 'REVE-Alljoined-v69-v85-75-25-Ensemble'")
    old="""        model = ReveRetrieval(coordinates,meta['n_times'],meta['sfreq'])
        state = torch.load(root/'weights.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        return model.to(meta['device']).eval()"""
    new="""        bundle = torch.load(root/'weights.pt',map_location='cpu',weights_only=True)
        models = []
        for state in bundle['members']:
            model = ReveRetrieval(coordinates,meta['n_times'],meta['sfreq'])
            model.load_state_dict(state,strict=True)
            models.append(model.to(meta['device']).eval())
        return OutputEnsemble(models, bundle['coefficients']).eval()"""
    assert text.count(old)==1
    (package/'submission.py').write_text(text.replace(old,new)+TAIL)
    for name in ['positions.json','LICENSE_BRAINDECODE.txt']:shutil.copyfile(template/name,package/name)
    members=[];origins=[]
    for run,file in CHOICES:
        path=BASE/run/'stage1'/file
        state=torch.load(path,map_location='cpu',weights_only=False)['model']
        weights={k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')}
        weights.update({'head.'+k.removeprefix('blur_head.'):v for k,v in state.items() if k.startswith('blur_head.')})
        weights.update({k:v for k,v in state.items() if k.startswith('recovery.')})
        members.append(weights);origins.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    torch.save(dict(members=members,coefficients=[.75,.25]),package/'weights.pt')
    spec=importlib.util.spec_from_file_location('exported_ensemble',package/'submission.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cfg=json.loads((BASE/CHOICES[1][0]/'config.json').read_text())
    manifest=json.loads(Path(cfg['data_manifest']).read_text());channels=manifest['channels']
    meta=dict(n_chans=32,n_times=120,sfreq=120.,ch_names=channels,n_outputs=1536,submission_dir=package,device='cuda')
    original=socket.socket.connect
    def guard(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('Network forbidden')
        return original(sock,address)
    socket.socket.connect=guard
    model=mod.Solver().load_model(meta)
    x=torch.tensor(np.concatenate([np.load(r['array'],mmap_mode='r')[:2] for r in manifest['test']])[:257].copy(),device='cuda')
    sys.path.insert(0,str(BASE/CHOICES[1][0]/'source'))
    from alljoined_v67 import LocalOTModel
    reference=LocalOTModel({'channels':channels}).cuda().eval()
    expected=torch.zeros((len(x),1536),device='cuda')
    with torch.inference_mode():
        for coefficient,(run,file) in zip([.75,.25],CHOICES):
            state=torch.load(BASE/run/'stage1'/file,map_location='cpu',weights_only=False)['model']
            if 'alljoined_positions' in state:state=dict(state);state['positions']=state.pop('alljoined_positions')
            reference.load_state_dict(state,strict=True)
            expected+=coefficient*torch.cat([reference(b) for b in x.split(32)])
        actual=model.predict(x)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        perm=torch.randperm(len(x),device='cuda')
        torch.testing.assert_close(model.predict(x[perm]),actual[perm],rtol=1e-4,atol=1e-4)
        for size in [1,7,33]:torch.testing.assert_close(model.predict(x[:size]),actual[:size],rtol=1e-4,atol=1e-4)
        assert model.predict(x[:0]).shape==(0,1536)
        cpu=mod.Solver().load_model(dict(meta,device='cpu'))
        torch.testing.assert_close(cpu.predict(x[:2].cpu()),actual[:2].cpu(),rtol=1e-4,atol=1e-4)
        order=torch.randperm(32);moved=mod.Solver().load_model(dict(meta,ch_names=[channels[i] for i in order]))
        torch.testing.assert_close(moved.predict(x[:7,order]),actual[:7],rtol=1e-4,atol=1e-4)
    result=json.loads((BASE/'alljoined_bayesian_soup_v69_v85_20261106/output_ensemble.json').read_text())
    assert result['selected']['alpha']==.75 and result['selected']['kind']=='raw'
    report=dict(members=origins,coefficients=[.75,.25],real_samples=len(x),max_abs_error=float((actual-expected).abs().max()),batch_permutation='passed',variable_batches='passed',channel_permutation='passed',cpu_gpu='passed',network='disabled',expected_clean=result['full_public_diagnostic'],holdout=result['holdout'],inference='0.75*v69 clean output + 0.25*v85 clean output; no per-member normalization; EEG only')
    (OUT/'build.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
