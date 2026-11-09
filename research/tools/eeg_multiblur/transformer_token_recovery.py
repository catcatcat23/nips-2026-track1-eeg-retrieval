"""Frozen REVE tokens -> Transformer encoder/conditional decoder recovery.

Matched to v42's original B base, loss, cache, splits and optimizer. The new
branch adds token self-attention and a two-layer conditional decoder. Not diffusion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import token_recovery as engine
from v33_recovery_stage2 import make_head


class TransformerTokenRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = make_head('B')
        self.query = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 256))
        self.tokens = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256))
        # Token ordering is inherited unchanged from the verified 63-token cache.
        self.position = nn.Parameter(torch.empty(1, 63, 256))
        nn.init.normal_(self.position, std=.02)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(256, 4, dim_feedforward=1024,
                dropout=.1, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(2)
        ])
        self.attention = nn.ModuleList([
            nn.TransformerDecoderLayer(256, 4, dim_feedforward=1024,
                dropout=.1, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(2)
        ])
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256,1536))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, h, tokens):
        assert tokens.shape[1:] == (63,512)
        memory = self.tokens(tokens) + self.position
        for block in self.encoder:
            memory = block(memory)
        query = self.query(h).unsqueeze(1)
        for block in self.attention:
            query = block(query, memory)
        return self.base(h) + self.output(query[:,0])


def check_cache(cfg):
    root=Path(cfg['cache_root']);tc=Path(cfg['token_cache_root']);src=Path(cfg['source_root'])
    for metadata in [root/'complete.json',tc/'complete.json']:
        assert json.loads(metadata.read_text())['checkpoint_sha256']==cfg['checkpoint_sha256']
    assert json.loads((tc/'complete.json').read_text())['all_h_checked']
    for split in ['train','val','test']:
        ids=np.load(root/f'{split}_ids.npy')
        assert np.load(tc/f'{split}.npy',mmap_mode='r').shape==(len(ids),63,512)
        for key in ['ids','subjects']:
            assert np.array_equal(np.load(root/f'{split}_{key}.npy'),np.load(src/'data'/f'{split}_{key}.npy'))


@torch.no_grad()
def diagnose(cfg,out):
    ck=torch.load(out/'B/best.pt',map_location='cpu',weights_only=False)
    head=TransformerTokenRecovery().cuda().eval();head.load_state_dict(ck['head'])
    root=Path(cfg['cache_root']);tc=Path(cfg['token_cache_root'])
    ids=np.load(root/'test_ids.npy');ix=np.random.default_rng(52).choice(len(ids),2048,replace=False)
    h=np.load(root/'test_h.npy',mmap_mode='r')[ix].copy();tokens=np.load(tc/'test.npy',mmap_mode='r')[ix].copy()
    clean=np.load(Path(cfg['source_root'])/'data/clean.npy');gallery_ids=np.unique(ids)
    gallery=torch.nn.functional.normalize(torch.tensor(clean[gallery_ids],device='cuda'),dim=-1)
    labels=torch.tensor(np.searchsorted(gallery_ids,ids[ix]),device='cuda')
    result={'epoch':ck['epoch'],'samples':len(ix),'candidates':len(gallery_ids),'note':'diagnostic subset only; not a trained ablation'}
    for variant in ['normal','base_only','shuffled_tokens','mean_tokens']:
        tt=tokens
        if variant=='shuffled_tokens':tt=tokens[np.random.default_rng(52).permutation(len(ix))]
        if variant=='mean_tokens':tt=np.broadcast_to(tokens.mean(1,keepdims=True),tokens.shape).copy()
        pred=[]
        for start in range(0,len(ix),256):
            x=torch.tensor(h[start:start+256],device='cuda');t=torch.tensor(tt[start:start+256],device='cuda')
            pred.append(head.base(x) if variant=='base_only' else head(x,t))
        q=torch.cat(pred);rank=(torch.nn.functional.normalize(q,dim=-1)@gallery.T).topk(5).indices
        result[variant]={'top1':float((rank[:,0]==labels).float().mean()),'top5':float((rank==labels[:,None]).any(1).float().mean())}
    engine.write(out/'B/token_diagnostics.json',result)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());out=Path(cfg['run_root'])
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    check_cache(cfg)
    engine.TokenRecovery=TransformerTokenRecovery
    (out/'status.txt').write_text('PREFLIGHT\n')
    engine.preflight(cfg,out)
    # The real-source preflight checks cache correspondence, 8-sample fitting,
    # token-swap sensitivity and decoder gradients. Discard its learned weights.
    (out/'status.txt').write_text('TRAINING\n')
    engine.train(cfg,out)
    (out/'status.txt').write_text('DIAGNOSTICS\n')
    diagnose(cfg,out)
    (out/'status.txt').write_text('COMPLETED\n')

if __name__=='__main__':main()
