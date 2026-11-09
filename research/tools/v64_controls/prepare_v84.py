from pathlib import Path
import json,hashlib,shutil,ast
r=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments');p=r/'reve_v80_v64_identity_direct';o=r/'reve_v84_global_identity_weight1'
assert not o.exists();o.mkdir();shutil.copytree(p/'source',o/'source')
for n in ['local.npy','local_complete.json','predictability']:(o/n).symlink_to((p/n).resolve())
c=json.loads((p/'config.json').read_text());c.update(run_root=str(o),identity_weight=1.0,experiment='v84 v80 matched identity weight1',only_change='identity weight 0.1 -> 1.0; diagnostic gradient measurement added')
(o/'config.json').write_text(json.dumps(c,indent=2))
f=o/'source/run_control.py';f.chmod(0o644);s=f.read_text();needle='    base.LocalOTModel=cls;';assert needle in s
s=s.replace(needle,'''    original=ex.loss_fn
    calls=0
    def monitored(model,batch,weights):
        nonlocal calls
        total,parts=original(model,batch,weights)
        if calls<3 or calls%100==0:
            params=[p for p in model.encoder.parameters() if p.requires_grad]
            retrieval=parts['clean']+parts['blur']+.1*parts['local']
            identity=cfg['identity_weight']*parts['identity']
            g=torch.autograd.grad(retrieval,params,retain_graph=True,allow_unused=True)
            h=torch.autograd.grad(identity,params,retain_graph=True,allow_unused=True)
            aa=total.new_zeros(());bb=aa.clone();ab=aa.clone()
            for x,y in zip(g,h):
                if x is not None:aa+=x.detach().float().square().sum()
                if y is not None:bb+=y.detach().float().square().sum()
                if x is not None and y is not None:ab+=(x.detach().float()*y.detach().float()).sum()
            row=dict(step=calls,retrieval_encoder_l2=float(aa.sqrt()),identity_weighted_encoder_l2=float(bb.sqrt()),identity_to_retrieval_ratio=float((bb/aa.clamp_min(1e-20)).sqrt()),cosine=float(ab/(aa*bb).sqrt().clamp_min(1e-20)))
            with (out/'gradient_balance.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
            print('GRADIENT_BALANCE',json.dumps(row),flush=True)
        calls+=1
        return total,parts
    ex.loss_fn=monitored
'''+needle)
ast.parse(s);f.write_text(s)
hashes={str(f.relative_to(o/'source')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (o/'source').rglob('*') if f.is_file() and '__pycache__' not in str(f)}
(o/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
for n in hashes:(o/'source'/n).chmod(0o444)
s=(p/'launch.sh').read_text().replace(str(p),str(o)).replace('CUDA_VISIBLE_DEVICES=2','CUDA_VISIBLE_DEVICES=1')
a=s.index('used=$(nvidia-smi');b=s.index('echo preflight',a)
s=s[:a]+'''echo waiting_for_gpu1 > queue_status.txt
while true; do
 used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
 if [ "$used" -lt 1000 ]; then break; fi
 sleep 60
done
'''+s[b:]
(o/'launch.sh').write_text(s)
(o/'EXPERIMENT_PROTOCOL.md').write_text('v84 is a weight-only training control for v80, identity_weight=1.0. Same original random batch256 for all losses; no second sampled batch. Subject positives unrestricted by image, other subjects negative. Fresh pretrained REVE seed33. Other data, augmentation, optimizer, EMA and public-test development selection settings unchanged. Sparse autograd.grad diagnostics do not modify .grad or update rules. CPU syntax verified, GPU preflight required before training.\n')
print(o)
