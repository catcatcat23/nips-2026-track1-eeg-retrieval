"""Portable B packaging: torch-only math, bounded batches, unchanged FP32 state."""
from pathlib import Path
import ast,hashlib,json,shutil
p=Path('/home/Anteng/eeg-retrieval/submissions');src=p/'reve_b_epoch6';out=p/'reve_b_epoch6_compat_v2';out.mkdir(exist_ok=True)
s=(src/'submission.py').read_text()
s=s.replace('from torch.nn.attention import SDPBackend, sdpa_kernel\n','').replace('from einops import rearrange\n','')
a=s.index('        q, k, v = (');b=s.index('\n        if self.use_sdpa:',a)
s=s[:a]+'''        q, k, v = (
            t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)
            for t in (q, k, v)
        )
''' + s[b:]
a=s.index('        if self.use_sdpa:  # SDPA Implementation');b=s.index('        return out',a)
s=s[:a]+'''        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            scale = q.shape[-1] ** -0.5
            out = ((q @ k.transpose(-1, -2)) * scale).softmax(-1) @ v
        out = out.transpose(1, 2).reshape(qkv.shape[0], qkv.shape[1], -1)
''' +s[b:]
a=s.index('    @torch.inference_mode()\n    def predict(self, X):');b=s.index('\n\nclass Solver',a)
s=s[:a]+'''    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        if X.ndim != 3 or X.shape[1] != len(self.positions) or X.shape[-1] != self.input_times:
            raise ValueError('Input shape disagrees with meta')
        if len(X) == 0:
            return torch.empty(0, 1536, device=self.positions.device)
        results = []
        # Bound activations independently of platform batch size and device.
        chunk_size = 32 if self.positions.device.type == 'cuda' else 16
        for part in X.split(chunk_size):
            part = part.to(device=self.positions.device, dtype=torch.float32)
            if not torch.isfinite(part).all():
                raise ValueError('EEG input contains nonfinite values')
            if part.shape[-1] != self.resampled_times:
                part = F.interpolate(part, size=self.resampled_times, mode='linear', align_corners=False)
            mean = part.mean(dim=-1, keepdim=True)
            std = part.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            part = ((part-mean)/std).clamp(-15.,15.)
            results.append(self(part))
        return torch.cat(results, dim=0)
''' +s[b:]
s=s.replace("name = 'REVE-B-ProjectedRecovery-Epoch6'","name = 'REVE-B-Epoch6-CompatV2'")
ast.parse(s);(out/'submission.py').write_text(s)
for name in ['weights.pt','positions.json','LICENSE_BRAINDECODE.txt']:shutil.copyfile(src/name,out/name)
assert hashlib.sha256((out/'weights.pt').read_bytes()).digest()==hashlib.sha256((src/'weights.pt').read_bytes()).digest()
(out/'build.json').write_text(json.dumps({'source':'reve_b_epoch6','weights_unchanged':True,'changes':['remove einops dependency','remove torch.nn.attention backend selector import','internal chunks32 GPU/16 CPU','empty batch supported'],'original_failure_cause':'unknown: platform provided no logs'},indent=2))
print(out)
