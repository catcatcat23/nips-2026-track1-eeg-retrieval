
class ReveRetrieval(nn.Module):
    """Frozen inference-only copy of the fine-tuned REVE + mean/linear head."""
    def __init__(self, positions, n_times, sfreq):
        super().__init__()
        self.register_buffer('positions', torch.tensor(positions, dtype=torch.float32), persistent=False)
        self.input_times = int(n_times)
        self.input_sfreq = float(sfreq)
        self.resampled_times = round(self.input_times * 200.0 / self.input_sfreq)
        if self.resampled_times < 200:
            raise ValueError('REVE requires at least one second of EEG')
        self.to_patch_embedding = nn.Sequential(nn.Linear(200,512))
        self.fourier4d = FourierEmb4D(512, freqs=4)
        self.mlp4d = nn.Sequential(nn.Linear(4,512,bias=False),nn.GELU(),nn.LayerNorm(512))
        self.ln = nn.LayerNorm(512)
        self.transformer = TransformerBackbone(512,22,8,64,int(512*2.66),True)
        self.head = nn.Linear(512,1536)

    def forward(self, eeg):
        patches = eeg.unfold(2,200,180)
        pos = self.positions.to(eeg).expand(len(eeg),-1,-1)
        pos = FourierEmb4D.add_time_patch(pos,patches.shape[2])
        tokens = self.to_patch_embedding(patches).flatten(1,2)
        tokens = tokens + self.ln(self.fourier4d(pos)+self.mlp4d(pos))
        return self.head(self.transformer(tokens).mean(dim=1))

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        if X.ndim != 3 or X.shape[1] != len(self.positions) or X.shape[-1] != self.input_times:
            raise ValueError('Input shape disagrees with meta')
        if not torch.isfinite(X).all():
            raise ValueError('EEG input contains nonfinite values')
        X = X.to(device=self.positions.device, dtype=torch.float32)
        if X.shape[-1] != self.resampled_times:
            X = F.interpolate(X,size=self.resampled_times,mode='linear',align_corners=False)
        # Same window-only bridge as the official competition REVE probe.
        # This cannot reverse the platform's continuous-recording preprocessing.
        mean = X.mean(dim=-1,keepdim=True)
        std = X.std(dim=-1,keepdim=True,unbiased=False).clamp_min(1e-6)
        X = ((X-mean)/std).clamp(-15.,15.)
        return self(X)


class Solver(CompetSolver):
    name = 'REVE-NeuralBench-Finetuned-Seed33'
    requirements = []

    def load_model(self, meta):
        root = Path(meta['submission_dir'])
        if int(meta['n_outputs']) != 1536:
            raise ValueError('This trained model targets the fixed 1536-D DINOv2 space')
        names = meta.get('ch_names')
        if names is None or len(names) != int(meta['n_chans']):
            raise ValueError('meta must supply channel names in input order')
        bank = json.loads((root/'positions.json').read_text())
        lookup = {name.upper():value for name,value in bank.items()}
        coordinates = []
        for name in names:
            name = str(name).removeprefix('EEG ').split('-',1)[0].upper()
            if name not in lookup:
                raise ValueError('Channel not present in the shipped REVE position bank: '+name)
            coordinates.append(lookup[name])
        model = ReveRetrieval(coordinates,meta['n_times'],meta['sfreq'])
        state = torch.load(root/'weights.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        return model.to(meta['device']).eval()
