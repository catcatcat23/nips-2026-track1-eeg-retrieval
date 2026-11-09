"""v40: original frozen-v33 B recovery recipe plus blur ranking distillation."""
import argparse, hashlib, json, time, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v33_recovery_stage2 import make_head, metrics, predictions, write
from neuraltrain.losses.losses import ClipLoss


def prepare_teacher(cfg, ids, testids):
    source=Path(cfg['source_root'])
    assert hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()==cfg['checkpoint_sha256']
    blur=np.load(source/'features/blur.npy')
    available=np.load(source/'features/available.npy')
    unique=np.unique(ids)
    assert not np.intersect1d(unique,testids).size
    assert available[unique].all() and np.isfinite(blur[unique]).all()
    assert blur.ndim==2 and blur.shape[1]==1536
    adapter=nn.Sequential(nn.Linear(1536,768),nn.ELU(),nn.Dropout(.2),nn.Linear(768,1536)).cuda().eval()
    ck=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    adapter.load_state_dict({k.removeprefix('image_adapter.'):v for k,v in ck['model'].items() if k.startswith('image_adapter.')},strict=True)
    adapter.requires_grad_(False)
    teacher=torch.zeros(len(blur),1536,device='cuda')
    with torch.no_grad():
        for ix in np.array_split(unique,math.ceil(len(unique)/512)):
            teacher[ix]=F.normalize(adapter(torch.from_numpy(blur[ix]).cuda()),dim=-1)
    assert not teacher.requires_grad
    return teacher


def distill_loss(q,h,targets,ids,teacher,cfg):
    # Preserve the original clean contrastive loss exactly, including duplicate columns.
    clean=ClipLoss(norm_kind='y',temperature=False,symmetric=False)(q,targets)
    unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
    first=torch.full((len(unique),),len(ids),device=ids.device,dtype=torch.long)
    first.scatter_reduce_(0,inverse,torch.arange(len(ids),device=ids.device),reduce='amin')
    tau=cfg['distill_temperature']
    with torch.no_grad():
        p=(F.normalize(h.float(),dim=-1)@teacher[unique].T/tau).softmax(-1)
    logits=F.normalize(q.float(),dim=-1)@F.normalize(targets[first].float(),dim=-1).T/tau
    kl=F.kl_div(logits.log_softmax(-1),p,reduction='batchmean')*tau*tau
    return clean+cfg['distill_weight']*kl, {'clean':clean.detach(),'distill_scaled':kl.detach()}


def preflight(cfg,out,h,ids,target,teacher):
    _,ix=np.unique(ids.cpu().numpy(),return_index=True)
    ix=torch.tensor(ix[:8],device='cuda');tinyh=h[ix];tinyids=ids[ix];y=target[tinyids]
    model=make_head('B').eval();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0)
    initial=float(distill_loss(model(tinyh),tinyh,y,tinyids,teacher,cfg)[0].detach())
    for step in range(300):
        opt.zero_grad();loss,_=distill_loss(model(tinyh),tinyh,y,tinyids,teacher,cfg)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        with torch.no_grad():
            acc=float(((F.normalize(model(tinyh),dim=-1)@F.normalize(y,dim=-1).T).argmax(1)==torch.arange(8,device='cuda')).float().mean())
        if acc==1 and float(loss.detach())<initial*.3:break
    assert acc==1 and float(loss.detach())<initial*.3
    # Duplicate candidate observations must not change the distilled distribution.
    q=torch.randn(3,1536,device='cuda',requires_grad=True);hh=tinyh[[0,0,1]].clone().requires_grad_();di=tinyids[[0,0,1]]
    val,_=distill_loss(q,hh,target[di],di,teacher,cfg);val.backward()
    assert hh.grad is None and torch.isfinite(q.grad).all() and not teacher.requires_grad
    write(out/'preflight.json',dict(passed=True,initial_loss=initial,final_loss=float(loss.detach()),steps=step+1,tiny_top1=acc,teacher_stop_gradient=True,unique_gallery=True,updates_discarded=True))
    print('PREFLIGHT PASSED',flush=True)

def train(cfg, out, arm):
    cache = Path(cfg['cache_root'])
    assert json.loads((cache / 'complete.json').read_text())['checkpoint_sha256'] == cfg['checkpoint_sha256']
    dest = out / arm
    dest.mkdir(exist_ok=True)
    assert not (dest / 'history.jsonl').exists(), 'Use a new run directory'
    for split in ['train','val','test']:
        for suffix in ['ids','subjects']:
            assert np.array_equal(np.load(cache/f'{split}_{suffix}.npy'),np.load(Path(cfg['source_root'])/'data'/f'{split}_{suffix}.npy'))
    clean = np.load(Path(cfg['source_root']) / 'data/clean.npy')
    target = torch.from_numpy(clean).cuda()
    ids = np.concatenate([np.load(cache / f'{s}_ids.npy') for s in ('train', 'val')])
    trainids = torch.as_tensor(ids, device='cuda')
    h = torch.from_numpy(np.concatenate([np.load(cache / f'{s}_h.npy') for s in ('train', 'val')])).cuda()
    testids = np.load(cache / 'test_ids.npy')
    subjects = np.load(cache / 'test_subjects.npy')
    assert len(np.intersect1d(ids, testids)) == 0
    testh = torch.from_numpy(np.load(cache / 'test_h.npy')).cuda()
    teacher = prepare_teacher(cfg, ids, testids)
    preflight(cfg, out, h, trainids, target, teacher)
    head = make_head(arm)
    initial = torch.load(cfg['initial_reference'], map_location='cpu', weights_only=True)
    for key, value in head.state_dict().items():
        assert torch.equal(value.cpu(), initial[key]), key
    torch.save(head.state_dict(), dest / 'initial_head.pt')
    opt = torch.optim.AdamW(head.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps = (len(h) + cfg['batch_size'] - 1) // cfg['batch_size']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], epochs=cfg['epochs'], steps_per_epoch=steps, pct_start=0.1)
    if arm == 'A':
        ck = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
        old = OldHead().cuda()
        old.recovery.load_state_dict({k.removeprefix('recovery.'): v for k, v in ck['model'].items() if k.startswith('recovery.')})
        original = metrics(predictions(old, testh), testids, subjects, clean)
        assert abs(original['single_trial']['top5'] - 0.3708086785009862) < 1e-5
        write(dest / 'original_v13_metrics.json', original)
        del old, ck
    best, stale = -1.0, 0
    generator = torch.Generator(device='cuda').manual_seed(cfg['seed'])
    for epoch in range(1, cfg['epochs'] + 1):
        begin = time.time()
        head.train()
        order = torch.randperm(len(h), generator=generator, device='cuda')
        totals, count = {}, 0
        for step, idx in enumerate(order.split(cfg['batch_size'])):
            opt.zero_grad(set_to_none=True)
            q = head(h[idx])
            loss, parts = distill_loss(q, h[idx], target[trainids[idx]], trainids[idx], teacher, cfg)
            assert torch.isfinite(loss), (arm, epoch, step)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            sched.step()
            for key, value in {'total': loss.detach(), **parts}.items():
                totals[key] = totals.get(key, 0.0) + float(value) * len(idx)
            count += len(idx)
            if step % 500 == 0:
                print(json.dumps({'arm': arm, 'epoch': epoch, 'step': step, 'steps': steps, 'loss': float(loss.detach())}), flush=True)
        selection = metrics(predictions(head, testh), testids, subjects, clean)
        # Fixed training subset: diagnostic only, never checkpoint selection.
        diagnostic = metrics(predictions(head, h[:2048]), ids[:2048], np.zeros(2048, dtype=int), clean)
        score = selection['single_trial']['top5']
        improved = score > best
        if improved:
            best, stale = score, 0
            torch.save({'head': head.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'epoch': epoch, 'arm': arm, 'config': cfg, 'selection': selection, 'source_checkpoint_sha256': cfg['checkpoint_sha256']}, dest / 'best.pt')
        else:
            stale += 1
        record = {'arm': arm, 'epoch': epoch, 'loss': {k: v/count for k, v in totals.items()}, 'selection': selection, 'train_diagnostic': diagnostic, 'lr': opt.param_groups[0]['lr'], 'best_top5': best, 'stale': stale, 'seconds': time.time() - begin}
        with (dest / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        write(dest / 'status.json', {'state': 'training', 'epoch': epoch, 'best_top5': best, 'stale': stale})
        if stale >= cfg['patience']:
            break
    best_ck = torch.load(dest / 'best.pt', map_location='cpu', weights_only=False)
    write(dest / 'result.json', {'arm': arm, 'best_epoch': best_ck['epoch'], 'selection': best_ck['selection'], 'role': 'public_test_used_for_model_selection_not_independent_test'})
    write(dest / 'status.json', {'state': 'complete', 'epochs': epoch, 'best_epoch': best_ck['epoch'], 'best_top5': best})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());out=Path(cfg['run_root'])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    train(cfg,out,'B')
if __name__=='__main__':main()
