import json,sys,csv
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/home/Anteng/eeg-retrieval/tools/eeg_multiblur')
from channel_contract import register_channels,channel_positions,batch_positions
from reference_model import ReferenceModel
root=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
m=json.load(open(root/'reve_v13_reference/data/manifest.json'))
rows=json.load(open(root/'reve_v18_alljoined/eeg_complete.json'))['recordings']
registry={'things':m['channels']};counts={}
for r in rows:
 register_channels(registry,r['dataset'],r['channels'])
 x=np.load(r['array'],mmap_mode='r');assert x.shape==(len(r['events']),len(r['channels']),120)
 if r['dataset']=='Xu2025Alljoined':
  with open(r['raw'],'rb') as f:
   header=f.read(256);n=int(header[252:256]);labels=f.read(16*n)
  names=[labels[i*16:(i+1)*16].decode().strip() for i in range(n)]
  assert [s for s in names if s in r['channels']]==r['channels']
  tsv=Path(r['raw'].replace('_eeg.edf','_channels.tsv'))
  with tsv.open() as f: names=[x['name'] for x in csv.DictReader(f,delimiter='\t') if x['type']=='EEG']
  assert names==r['channels']
 counts[r['dataset']]=counts.get(r['dataset'],0)+1
print('CACHE AUDIT',counts,flush=True)
torch.set_num_threads(4)
model=ReferenceModel(m).eval()
for d,n in registry.items():model.register_buffer('pos_'+d,channel_positions(model.encoder,n))
ix=[m['channels'].index(n) for n in registry['Xu2025Alljoined']]
assert torch.equal(model.pos_things[ix],model.pos_Xu2025Alljoined)
shapes={}
with torch.no_grad():
 for d in ['things','Xu2025Alljoined']:
  if d=='things':x=np.load(root/'reve_v13_reference/data/train_eeg.npy',mmap_mode='r')[:1].copy()
  else:x=np.load(next(r['array'] for r in rows if r['dataset']==d),mmap_mode='r')[:1].copy()
  x=model.preprocess(torch.from_numpy(x));pos=batch_positions(model,x,d)
  feat=model.encoder(x,pos=pos,return_output=True)[-1].mean(1);y=model.blur_head(feat)
  assert y.shape==(1,1536) and torch.isfinite(y).all();shapes[d]=list(y.shape)
try:batch_positions(model,torch.zeros(1,63,200),'Xu2025Alljoined')
except ValueError:pass
else:raise AssertionError('wrong channel count accepted')
try:register_channels(registry,'Xu2025Alljoined',registry['Xu2025Alljoined'][::-1])
except ValueError:pass
else:raise AssertionError('wrong channel order accepted')
result={'recordings_checked':counts,'channels':registry,'alljoined2025_indices_in_things':ix,'shared_positions_equal':True,'cpu_forward':shapes,'reject_wrong_count_and_order':True}
out=Path('/home/Anteng/eeg-retrieval/tools/eeg_multiblur/channel_audit.json');out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='channels'}))
