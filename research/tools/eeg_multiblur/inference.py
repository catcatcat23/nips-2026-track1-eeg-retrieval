"""Inference accepts ONLY preprocessed EEG in the recorded channel order."""
from pathlib import Path
import torch
from torch import nn

class EEGOnly(nn.Module):
    def __init__(self,trained):
        super().__init__()
        self.spatial=trained.spatial;self.bn=trained.bn;self.temporal=trained.temporal
        self.global_head=trained.global_head;self.restore=trained.restore
    def forward(self,eeg):
        shared=self.temporal(self.bn(torch.einsum('kc,bct->bkt',self.spatial.abs(),eeg)))
        h=self.global_head(shared.flatten(1))
        return h+self.restore(h)


class ChannelAdapter(nn.Module):
    """Name-aligned inference; absent training electrodes are zero-filled.

    This makes subset montages executable, not equivalent to training on them.
    Unknown electrodes fail explicitly rather than silently swapping channels.
    """
    def __init__(self, model, trained_channels, meta):
        super().__init__()
        if float(meta['sfreq']) != 120 or int(meta['n_times']) != 120:
            raise ValueError('Expected competition 120 Hz, one-second windows')
        def name(s):return str(s).removeprefix('EEG ').split('-',1)[0].upper()
        lookup={name(s):i for i,s in enumerate(trained_channels)}
        names=[name(s) for s in meta['ch_names']]
        if len(names)!=meta['n_chans'] or len(set(names))!=len(names):
            raise ValueError('Invalid channel metadata')
        unknown=set(names)-set(lookup)
        if unknown:raise ValueError('Untrained electrodes: '+str(sorted(unknown)))
        self.register_buffer('indices',torch.tensor([lookup[s] for s in names]))
        self.channels=len(trained_channels)
        self.model=model.eval()

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        if X.ndim!=3 or X.shape[1]!=len(self.indices) or X.shape[2]!=120:
            raise ValueError('EEG shape disagrees with metadata')
        aligned=X.new_zeros((len(X),self.channels,120),dtype=torch.float32)
        aligned[:,self.indices,:]=X.float()
        return self.model(aligned)


def export(checkpoint,output):
    import json
    from reve_model import build_model
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    m=build_model(c['manifest'],c['config']);m.load_state_dict(c['model']);m.eval()
    if c['config'].get('encoder') in ('reve_frozen','reve_finetune'):
        output=Path(output);output.mkdir(exist_ok=True)
        torch.save(c,output/'eeg_only.pt')
        (output/'input_contract.json').write_text(json.dumps({'input':'competition preprocessed EEG [B,C,120]', 'output':'official-space [B,1536]', 'channels':c['manifest']['channels'], 'encoder':c['config']['encoder']+'; interpolation120to200; channel normalization; mean pooling', 'loader':'reve_model.build_model(manifest, config), then load_state_dict(model)', 'competition_submission_status':'not yet packaged or certified'},indent=2))
        return
    compact=EEGOnly(m).eval()
    x=torch.randn(2,*c['manifest']['input_shape'])
    with torch.no_grad():assert torch.equal(m(x),compact(x))
    output=Path(output);output.mkdir(exist_ok=True)
    torch.save({'state_dict':compact.state_dict(),'input_shape':c['manifest']['input_shape'],'channels':c['manifest']['channels']},output/'eeg_only.pt')
    (output/'input_contract.json').write_text(json.dumps({'scope':c['config']['scope'],'input':'preprocessed float EEG [B,C,T]','output':'unnormalized official-space feature [B,1536]','preprocessing':c['manifest']['neuro_config'],'channels':c['manifest']['channels'],'input_shape':c['manifest']['input_shape'],'start':c['manifest']['start'],'duration':c['manifest']['duration'],'competition_replay':'pending actual competition input specification'},indent=2))
    traced=torch.jit.trace(compact,x);traced.save(str(output/'eeg_only.ts'))
    with torch.no_grad():assert torch.allclose(torch.jit.load(str(output/'eeg_only.ts'))(x),m(x))
